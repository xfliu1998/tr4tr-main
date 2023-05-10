import torch
import os
import math
import matplotlib.pyplot as plt
from utils.data_utils import farthestPointDownSample
from utils.visual_utils import *
from visdom import Visdom
import open3d as o3d
import cv2
from scipy.optimize import linear_sum_assignment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def registration2d(source_image, target_image, registration_matrix, valid_matrix, optical_flow_gt):
    h, w = source_image.shape[0], source_image.shape[1]
    h_scale, w_scale = 480 / h, 640 / w
    ndarray_image = np.concatenate([source_image, target_image], axis=1)  # h, 2w, 3
    registration_matrix = registration_matrix[valid_matrix].reshape(-1, 4)

    # calculate the EPE2D error
    error_matrix = np.zeros((registration_matrix.shape[0], ))
    for i in range(len(registration_matrix)):
        u, v = round(registration_matrix[i, 0]/w_scale), round(registration_matrix[i, 1]/h_scale)
        u_error = abs(optical_flow_gt[v, u, 0]) - abs(registration_matrix[i, 2] - registration_matrix[i, 0])
        v_error = abs(optical_flow_gt[v, u, 1]) - abs(registration_matrix[i, 3] - registration_matrix[i, 1])
        error = math.sqrt(u_error ** 2 + v_error ** 2)
        error_matrix[i] = error
    # get the index in order of error from smallest to largest
    sorted_array = np.argsort(error_matrix)
    print('TR4TR EPE2D error: %f pixel' % np.mean(error_matrix))

    # sample n pairs with the least error
    n = 80
    for i in range(n):
        sorted_registration = registration_matrix[sorted_array[i]]
        plt.plot([sorted_registration[0]/w_scale, sorted_registration[2]/w_scale + w],
                 [sorted_registration[1]/h_scale, sorted_registration[3]/h_scale],
                 color=[(10+i)/255, 80/255, 220/255], linewidth=0.5, marker='.', markersize=2)
    plt.ion()
    cv2.imwrite('experiment/output/registration2d_tr4tr.jpg', ndarray_image)
    plt.imshow(ndarray_image.astype('uint8'))
    plt.show()


def registration3d(source_point_all, source_color_all, target_point_all, target_color_all,
                   source_point_gt, target_point_gt, source_point_pred, target_point_pred):
    num_valid = source_point_gt.shape[0]
    source_pcd_all = o3d.geometry.PointCloud()
    source_pcd_all.points = o3d.utility.Vector3dVector(source_point_all.reshape(-1, 3))
    source_pcd_all.colors = o3d.utility.Vector3dVector(source_color_all.reshape(-1, 3)/255)
    target_pcd_all = o3d.geometry.PointCloud()
    target_pcd_all.points = o3d.utility.Vector3dVector(target_point_all.reshape(-1, 3))
    target_pcd_all.colors = o3d.utility.Vector3dVector(target_color_all.reshape(-1, 3)/255)

    source_pcd_gt = o3d.geometry.PointCloud()
    source_pcd_gt.points = o3d.utility.Vector3dVector(source_point_gt)
    source_pcd_gt.paint_uniform_color([255/255, 127/255, 0/255])  # orange
    target_pcd_gt = o3d.geometry.PointCloud()
    target_pcd_gt.points = o3d.utility.Vector3dVector(target_point_gt)
    target_pcd_gt.paint_uniform_color([50/255, 205/255, 50/255])  # green

    source_pcd_pred = o3d.geometry.PointCloud()
    source_pcd_pred.points = o3d.utility.Vector3dVector(source_point_pred)
    source_pcd_pred.paint_uniform_color([205/255, 92/255, 92/255])  # red
    target_pcd_pred = o3d.geometry.PointCloud()
    target_pcd_pred.points = o3d.utility.Vector3dVector(target_point_pred)
    target_pcd_pred.paint_uniform_color([67/255, 110/255, 238/255])  # blue

    align_colors = [[(10+0.1*i)/255, 80/255, 220/255] for i in range(num_valid)]
    source_gt_pred_points = np.concatenate([source_point_gt, source_point_pred], axis=0)
    source_gt_pred_lines = [[i, i + num_valid] for i in range(num_valid)]
    source_gt_pred_align = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(source_gt_pred_points),
        lines=o3d.utility.Vector2iVector(source_gt_pred_lines),
    )
    source_gt_pred_align.colors = o3d.utility.Vector3dVector(align_colors)

    target_gt_pred_points = np.concatenate([target_point_gt, target_point_pred], axis=0)
    target_gt_pred_lines = [[i, i + num_valid] for i in range(num_valid)]
    target_gt_pred_align = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(target_gt_pred_points),
        lines=o3d.utility.Vector2iVector(target_gt_pred_lines),
    )
    target_gt_pred_align.colors = o3d.utility.Vector3dVector(align_colors)

    source_target_gt_points = np.concatenate([source_point_gt, target_point_gt], axis=0)
    source_target_gt_lines = [[i, i + num_valid] for i in range(num_valid)]
    source_target_gt_align = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(source_target_gt_points),
        lines=o3d.utility.Vector2iVector(source_target_gt_lines),
    )
    source_target_gt_align.colors = o3d.utility.Vector3dVector(align_colors)

    source_target_pred_points = np.concatenate([source_point_pred, target_point_pred], axis=0)
    source_target_pred_lines = [[i, i + num_valid] for i in range(num_valid)]
    source_target_pred_align = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(source_target_pred_points),
        lines=o3d.utility.Vector2iVector(source_target_pred_lines),
    )
    source_target_pred_align.colors = o3d.utility.Vector3dVector(align_colors)

    geometry_dict = {
        "source_pcd_all": source_pcd_all,
        "target_pcd_all": target_pcd_all,
        "source_pcd_gt": source_pcd_gt,
        "target_pcd_gt": target_pcd_gt,
        "source_pcd_pred": source_pcd_pred,
        "target_pcd_pred": target_pcd_pred
    }
    alignment_dict = {
        "source_gt_pred_align": source_gt_pred_align,
        "target_gt_pred_align": target_gt_pred_align,
        "source_target_gt_align": source_target_gt_align,
        "source_target_pred_align": source_target_pred_align
    }

    # customize the display results
    manager = CustomDrawGeometryWithKeyCallback(geometry_dict, alignment_dict)
    manager.custom_draw_geometry_with_key_callback()


def predict(model, test_loader, pretrained_model):
    project_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(project_path + '/output/' + pretrained_model):
        print('Model failure to load')
        return
    checkpoint = torch.load(project_path + '/output/' + pretrained_model)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.to(device)
    viz = Visdom()
    # model.eval()
    for batch_idx, data in enumerate(test_loader):
        source_color_all = data['raw_image'][..., :3].cpu().detach().numpy()
        target_color_all = data['raw_image'][..., 6:9].cpu().detach().numpy()
        source_point_all = data['raw_image'][..., 3:6].cpu().detach().numpy()
        target_point_all = data['raw_image'][..., 9:].cpu().detach().numpy()
        source = data['source'].to(device)[..., :6]
        target = data['target'].to(device)[..., :6]
        optical_flow_gt = data['optical_flow_gt'].cpu().detach().numpy()
        scene_flow_mask = data['scene_flow_mask'].cpu().detach().numpy()
        intrinsics = data['intrinsics'].cpu().detach().numpy()
        point_cloud = data['point_cloud'].to(device)
        source_point_gt = point_cloud[..., :3]
        target_point_gt = point_cloud[..., 3:]

        x = torch.stack([source, target], dim=-1)  # B, H, W, C, T
        outputs = model(x)  # (b, num_points, 6)
        outputs = outputs.sup
        source_point_pred = outputs[:, :, :3]
        target_point_pred = outputs[:, :, 3:]

        # calculate the matching point id according to source point
        batch_size = source.shape[0]
        cost_dist_i = torch.cdist(source_point_pred.to(source_point_gt.dtype), source_point_gt, p=2)
        point_gt_id = np.array([list(linear_sum_assignment(cost_dist_i[i].cpu().detach().numpy())[1]) for i in range(batch_size)])
        for i in range(batch_size):
            source_point_gt[i], target_point_gt[i] = source_point_gt[i, point_gt_id[i]], target_point_gt[i, point_gt_id[i]]

        # infer the target point from the scene flow
        # target_point_pred = target_point_pred - source_point_pred + source_point_gt
        num_point = 1280
        idx = 0
        flags = farthestPointDownSample(source_point_pred[idx].cpu().detach().numpy(), num_point)
        source_point_pred = source_point_pred[idx][flags].cpu().detach().numpy()
        target_point_pred = target_point_pred[idx][flags].cpu().detach().numpy()
        source_point_gt = source_point_gt[idx][flags].cpu().detach().numpy()
        target_point_gt = target_point_gt[idx][flags].cpu().detach().numpy()

        # use visdom for visualization
        source_image = data['source'][idx, ..., 6:9].permute(2, 0, 1)
        target_image = data['target'][idx, ..., 6:9].permute(2, 0, 1)
        images = torch.stack([source_image, target_image], dim=0)
        viz.images(images, win='source & target', opts={'title': "source and target image"})
        viz.scatter(source_point_pred, win='source point', opts={'title': "source points", 'markersize': 1})
        viz.scatter(target_point_pred, win='target point', opts={'title': "target points", 'markersize': 1})

        registration_matrix, source_point_proj2D = get_registration_matrix(source_point_pred, target_point_pred, intrinsics[idx])
        # visualize the points within the mask
        valid_matrix = get_valid_matrix(scene_flow_mask[idx], source_point_proj2D)  # n,

        # visual 2d registration: visualizations without and with masks
        # registration2d(source_color_all[idx], target_color_all[idx], registration_matrix, valid_matrix, optical_flow_gt[idx])
        registration2d(data['source'][idx, ..., 6:9], data['target'][idx, ..., 6:9], registration_matrix, valid_matrix, optical_flow_gt[idx])

        valid_mask = True
        if valid_mask:
            valid_matrix = np.tile(valid_matrix.reshape(-1, 1), (1, 3))
            source_point_gt = source_point_gt[valid_matrix].reshape(-1, 3)
            target_point_gt = target_point_gt[valid_matrix].reshape(-1, 3)
            source_point_pred = source_point_pred[valid_matrix].reshape(-1, 3)
            target_point_pred = target_point_pred[valid_matrix].reshape(-1, 3)

        # use open3d visualization
        target_point_error = np.linalg.norm(target_point_gt - target_point_pred, axis=1, ord=2)
        print('TR4TR EPE3D error: %f m' % np.mean(target_point_error))
        registration3d(source_point_all[idx], source_color_all[idx], target_point_all[idx], target_color_all[idx],
                       source_point_gt, target_point_gt, source_point_pred, target_point_pred)

