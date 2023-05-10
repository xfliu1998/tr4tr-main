import csv
from visdom import Visdom
from tqdm import tqdm
from apex import amp
from torch.nn.utils import clip_grad_norm_
from experiment.evaluate import evaluate
from utils.experiment_utils import *


def train(epochs, accum_iter, train_loader, val_loader, model, criterion_sup, criterion_ssup, w_mask, drloc_mode, w_drloc, w_rev,
          optimizer, scheduler, use_amp, local_rank, ngpus_per_node, loss_log):
    # set the path to save the results
    project_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(project_path + '/output'):
        os.makedirs(project_path + '/output/')
    output_path = project_path + '/output/'

    viz = Visdom()
    avg_EPE3D_err_list, avg_EPE3D_acc_list, avg_deformation_err_list, avg_geometry_err_list = [], [], [], []
    loss_sum, global_step = 0, 1
    TRAIN, EVALUATE = False, True
    # scaler = amp.GradScaler()
    loss_list = list()

    for epoch in range(epochs):
        total_loss = 0
        if TRAIN:
            model.train()
            for batch_idx, data in tqdm(enumerate(train_loader)):
                try:
                    # source and target use normalized xyz training
                    source = data['source'].cuda(local_rank, non_blocking=True)[..., :6]
                    target = data['target'].cuda(local_rank, non_blocking=True)[..., :6]
                    point_cloud = data['point_cloud'].cuda(local_rank, non_blocking=True)
                    source_rgbd = data['source'].cuda(local_rank, non_blocking=True)[..., 6:]
                    target_rgbd = data['target'].cuda(local_rank, non_blocking=True)[..., 6:]
                    scene_flow_gt = data['scene_flow_gt'].cuda(local_rank, non_blocking=True)
                    scene_flow_mask = data['scene_flow_mask'].cuda(local_rank, non_blocking=True)
                    # mask = data['mask'].cuda(local_rank, non_blocking=True)

                    # source, target, mask_loss = mask_utils(source, target, mask)
                    x = torch.stack([source, target], dim=-1)  # B, H, W, C, T
                    targets = [source_rgbd, target_rgbd, point_cloud, scene_flow_gt, scene_flow_mask]

                    # forward propagation turns on automatic differentiation anomaly detection
                    # torch.autograd.set_detect_anomaly(True)
                    outputs = model(x)  # (b, num_points, 6)
                    loss = criterion_sup(outputs.sup, targets, evaluate=False)
                    if loss.cpu().detach().numpy() == 0.:
                        continue

                    # use self-supervised loss
                    if criterion_ssup != False:
                        loss_ssup, ssup_items = criterion_ssup(outputs, drloc_mode, w_drloc)
                        loss += loss_ssup
                        # if local_rank % ngpus_per_node == 0:
                        #     print('loss_ssup: ', loss_ssup.cpu().detach().numpy())

                    # use the mask of the pretrained model
                    # if mask_loss != None:
                    #     loss += w_mask * mask_loss

                    # use reverse flow loss
                    if w_rev != 0:
                        x_rev = torch.stack([target, source], dim=-1)  # B, H, W, C, T
                        point_cloud_rev = torch.cat((point_cloud[..., 3:], point_cloud[..., :3]), dim=-1)
                        targets_rev = [target_rgbd, source_rgbd, point_cloud_rev, scene_flow_gt, scene_flow_mask]
                        outputs_rev = model(x_rev)  # (b, num_points, 6)
                        loss_rev = criterion_sup(outputs_rev.sup, targets_rev, evaluate=False)
                        loss += w_rev * loss_rev

                    total_loss += loss.item()

                    # backprop
                    if batch_idx % accum_iter == 0:
                        optimizer.zero_grad()

                    # if local_rank % ngpus_per_node == 0:
                    #     for name, parms in model.named_parameters():
                    #         print("parameters beofore the update")
                    #         print('-->name:', name)
                    #         print('-->para:', parms)
                    #         print('-->grad_requirs:', parms.requires_grad)
                    #         print('-->grad_value:', parms.grad)
                    #         break

                    loss_sum += loss.item()
                    loss = loss / accum_iter
                    # back propagation dervative detection
                    with torch.autograd.detect_anomaly():
                        if use_amp:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward(retain_graph=True)

                    # gradient clipping
                    clip_grad_norm_(model.parameters(), max_norm=0.1, norm_type=2)
                    if batch_idx % accum_iter == 0:
                        optimizer.step()
                        scheduler.step()

                    # if local_rank % ngpus_per_node == 0:
                    #     for name, parms in model.named_parameters():
                    #         print("parameters after the update")
                    #         print('-->name:', name)
                    #         print('-->para:', parms)
                    #         print('-->grad_requirs:', parms.requires_grad)
                    #         print('-->grad_value:', parms.grad)

                    # record loss every num_gpu*batch_size*itertion time
                    iteration = 30
                    if global_step % iteration == 0 and local_rank % ngpus_per_node == 0:
                        avg_loss = loss_sum / iteration
                        with open(output_path + loss_log, "a", encoding='utf-8', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([avg_loss])
                        print('epoch: %d,  batch: %d, avg_loss: %f ' % (epoch, batch_idx, avg_loss))
                        # draw train loss every iteration*batch_size*2 time
                        viz.line([avg_loss], [global_step], win='train_loss', update='append',
                                 opts={'showlegend': True, 'title': "Train loss",
                                       'xlabel': "batch index", 'ylabel': "train loss"})
                        # draw learning rate
                        viz.line([optimizer.state_dict()['param_groups'][0]['lr']], [global_step], win='lr', update='append',
                                 opts={'showlegend': True, 'title': "Learning rate",
                                       'xlabel': "batch index", 'ylabel': "learning rate"})
                        loss_sum = 0
                    global_step += 1

                except Exception as e:
                    print('epoch: %d, batch_idx: %d, except: %s' % (epoch, batch_idx, e))

            loss_list.append(total_loss)
            print('total loss per epoch：', loss_list)

        if EVALUATE:
            with torch.no_grad():
                avg_EPE3D_err, avg_EPE3D_acc, avg_source_deformation_err, avg_deformation_err, avg_source_geometry_err, avg_geometry_err = \
                    evaluate(val_loader, model, criterion_sup, local_rank)
            avg_EPE3D_err_list.append(avg_EPE3D_err)
            avg_EPE3D_acc_list.append(avg_EPE3D_acc)
            avg_deformation_err_list.append(avg_deformation_err)
            avg_geometry_err_list.append(avg_geometry_err)
            if local_rank % ngpus_per_node == 0:
                print('epoch: %d,  EPE3D_err: %f(m), EPE3D_acc: %f, '
                      'source_deformation_err: %f(cm), '
                      'target_deformation_err: %f(cm), '
                      'source_geometry_err: %f(cm), '
                      'target_geometry_err: %f(cm)'
                      % (epoch, avg_EPE3D_err, avg_EPE3D_acc,
                         avg_source_deformation_err, avg_deformation_err,
                         avg_source_geometry_err, avg_geometry_err))

    # output the trained model
    checkpoint = {
        # 'model': TR4TR(),
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    checkpoint_path = output_path + 'checkpoint.pkl'
    torch.save(checkpoint, checkpoint_path)

    # plot and output the result pictures
    plot_curve(avg_EPE3D_err_list, 'EPE3D_error', output_path)
    plot_curve(avg_EPE3D_acc_list, 'EPE3D_accuracy', output_path)
    plot_curve(avg_deformation_err_list, 'deformation_error', output_path)
    plot_curve(avg_geometry_err_list, 'geometry_error', output_path)


