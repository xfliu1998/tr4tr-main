import torch
from tqdm import tqdm
from copy import deepcopy
import collections


def cal_EPE3D(num_EPE3D, scene_flow_loss, EPE3D_err, EPE3D_acc, err_dict, frames_dict, object, frames, valid_matrix):
    """
    calculate the scene flow error in mask, the scene flow unit is m
    :param num_EPE3D:
    :param scene_flow_loss:
    :param EPE3D_err: cumulative EPE3D error
    :param EPE3D_acc: cumulative EPE3D accuracy
    :param err_dict: cumulative error dict or different object
    :param frames_dict: cumulative error dict of different frames
    :param object:
    :param frames:
    :param valid_matrix: mask of valid
    :return: num_EPE3D, EPE3D_err, EPE3D_acc, err_dict, frames_dict
    """
    batch_size = object.shape[0]
    split_loss = torch.split(scene_flow_loss, len(scene_flow_loss) // batch_size)
    for i in range(batch_size):
        if split_loss[i][valid_matrix[i]].shape[0] == 0:
            continue
        error = torch.mean(split_loss[i][valid_matrix[i]]).item()
        err_dict[object[i].item()][0] += error
        err_dict[object[i].item()][1] += 1
        frames_dict[frames[i].item()][0] += error
        frames_dict[frames[i].item()][1] += 1
        EPE3D_err[0] += error
        EPE3D_err[1] += torch.sum(split_loss[i][valid_matrix[i]]).item()
        num_EPE3D += split_loss[i][valid_matrix[i]].shape[0]
    # scene flow less than 0.05 is included in accuracy
    EPE3D_acc += len(torch.masked_select(scene_flow_loss, scene_flow_loss.le(0.05)))
    return num_EPE3D, EPE3D_err, EPE3D_acc, err_dict, frames_dict


def cal_deformation_err(num_deformation, point_loss, deformation_err, err_dict, frames_dict, object, frames, valid_matrix):
    """
    calculate the deformation error in mask
    :param num_deformation:
    :param point_loss:
    :param deformation_err:
    :param err_dict: cumulative error dict or different object
    :param frames_dict: cumulative error dict of different frames
    :param object:
    :param frames:
    :param valid_matrix: mask of valid
    :return: num_deformation, deformation_err, err_dict, frames_dict
    """
    batch_size = object.shape[0]
    split_loss = torch.split(point_loss, len(point_loss) // batch_size, dim=0)
    for i in range(batch_size):
        if split_loss[i][valid_matrix[i]].shape[0] == 0:
            continue
        error = torch.mean(split_loss[i][valid_matrix[i]]).item()
        err_dict[object[i].item()][0] += error
        err_dict[object[i].item()][1] += 1
        frames_dict[frames[i].item()][0] += error
        frames_dict[frames[i].item()][1] += 1
        deformation_err[0] += error
        deformation_err[1] += torch.sum(split_loss[i][valid_matrix[i]]).item()
        num_deformation += split_loss[i][valid_matrix[i]].shape[0]
    return num_deformation, deformation_err, err_dict, frames_dict


def cal_geometry_error(num_geometry, target_point_depth_loss, geometry_err, object, valid_matrix):
    """
    calculate the geometry error in mask
    :param num_geometry:
    :param target_point_depth_loss:
    :param geometry_err:
    :param object:
    :param valid_matrix: mask of valid
    :return: geometry_err
    """
    batch_size = object.shape[0]
    split_loss = torch.split(target_point_depth_loss, len(target_point_depth_loss) // batch_size, dim=0)
    for i in range(batch_size):
        if split_loss[i][valid_matrix[i]].shape[0] == 0:
            continue
        error = torch.mean(split_loss[i][valid_matrix[i]]).item()
        geometry_err[0] += error
        geometry_err[1] += torch.sum(split_loss[i][valid_matrix[i]]).item()
        num_geometry += split_loss[i][valid_matrix[i]].shape[0]
    return num_geometry, geometry_err


def evaluate(val_loader, model, criterion, local_rank):
    # object id in evaluate set
    object_id_map = {'Jacket': 0, 'shirt': 1, 'gloves': 2, 'towel': 3, 'sweater': 4, 'shorts': 5, 'adult': 6, 'other': 7}
    init_dict = {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0]}
    # initialize the dict to calculate the error of different object, value0 is the cumulative error, value1 is the numbers
    scene_flow_dict, source_point_dict, target_point_dict = deepcopy(init_dict), deepcopy(init_dict), deepcopy(init_dict)
    # initialize the dict to calculate the error of different frames, value0 is the cumulative error, value1 is the numbers
    scene_flow_frames_dict = collections.defaultdict(lambda: [0., 0.])
    source_point_frames_dict = collections.defaultdict(lambda: [0., 0.])
    target_point_frames_dict = collections.defaultdict(lambda: [0., 0.])

    EPE2D_err, EPE3D_err, EPE2D_acc, EPE3D_acc = 0., [0., 0.], 0., 0.
    source_deformation_err, target_deformation_err, source_geometry_err, target_geometry_err = [0., 0.], [0., 0.], [0., 0.], [0., 0]
    num_EPE2D, num_EPE3D, num_source_deformation, num_target_deformation, num_source_geometry, num_target_geometry = 0, 0, 0, 0, 0, 0
    total_err, num_batch = 0, 0

    for batch_idx, data in tqdm(enumerate(val_loader)):
        frames = data['frames'].cuda(local_rank, non_blocking=True)
        object = data['object'].cuda(local_rank, non_blocking=True)
        source = data['source'].cuda(local_rank, non_blocking=True)[..., :6]
        target = data['target'].cuda(local_rank, non_blocking=True)[..., :6]
        point_cloud = data['point_cloud'].cuda(local_rank, non_blocking=True)
        source_rgbd = data['source'].cuda(local_rank, non_blocking=True)[..., 6:]
        target_rgbd = data['target'].cuda(local_rank, non_blocking=True)[..., 6:]
        scene_flow_gt = data['scene_flow_gt'].cuda(local_rank, non_blocking=True)
        scene_flow_mask = data['scene_flow_mask'].cuda(local_rank, non_blocking=True)
        intrinsics = data['intrinsics'].cuda(local_rank, non_blocking=True)

        x = torch.stack([source, target], dim=-1)  # B, H, W, C, T
        targets = [source_rgbd, target_rgbd, point_cloud, scene_flow_gt, scene_flow_mask]
        outputs = model(x)  # (b, num_points, 6)
        outputs = outputs.sup
        batch_size, num_point = outputs.shape[0], outputs.shape[1]
        source_point_loss, target_point_loss, source_point_depth_loss, target_point_depth_loss, \
        scene_flow_loss, total_loss = criterion(outputs, targets, evaluate=True)

        # calculate 2d mapping points
        source_point_pred = outputs[..., :3]
        source_point_rearr = source_point_pred.permute(0, 2, 1)  # (b, 3, num_points)
        source_point_proj2D = torch.div(torch.bmm(intrinsics, source_point_rearr),
                                        source_point_rearr[:, 2:, :])[:, :2, :]  # (b, 2, num_points)
        source_point_proj2D = source_point_proj2D.permute(0, 2, 1)  # (b, num_points, 2)

        # calculate valid_matrix according to source_pred
        h, w = scene_flow_mask.shape[1], scene_flow_mask.shape[2]
        valid_matrix = torch.full((batch_size, num_point), False)
        for i in range(batch_size):
            for j in range(num_point):
                u, v = torch.round(source_point_proj2D[i, j]).cpu().detach().numpy()
                u, v = int(u // (640/w)), int(v // (480/h))
                if 0 <= u < w and 0 <= v < h and scene_flow_mask[i, v, u, 0]:
                    valid_matrix[i, j] = True

        num_EPE3D, EPE3D_err, EPE3D_acc, scene_flow_dict, scene_flow_frames_dict = cal_EPE3D(num_EPE3D, scene_flow_loss,
                                                                                             EPE3D_err, EPE3D_acc,
                                                                                             scene_flow_dict,
                                                                                             scene_flow_frames_dict,
                                                                                             object, frames,
                                                                                             valid_matrix)
        num_source_deformation, source_deformation_err, source_point_dict, source_point_frames_dict = cal_deformation_err(
            num_source_deformation, source_point_loss, source_deformation_err,
            source_point_dict, source_point_frames_dict,
            object, frames, valid_matrix)
        num_target_deformation, target_deformation_err, target_point_dict, target_point_frames_dict = cal_deformation_err(
            num_target_deformation, target_point_loss, target_deformation_err,
            target_point_dict, target_point_frames_dict,
            object, frames, valid_matrix)
        num_source_geometry, source_geometry_err = cal_geometry_error(num_source_geometry, source_point_depth_loss,
                                                                      source_geometry_err, object, valid_matrix)
        num_target_geometry, target_geometry_err = cal_geometry_error(num_target_geometry, target_point_depth_loss,
                                                                      target_geometry_err, object, valid_matrix)
        total_err += total_loss
        num_batch += 1

    avg_EPE3D_err_mean = EPE3D_err[0] / num_batch / batch_size
    avg_EPE3D_err = EPE3D_err[1] / num_EPE3D
    avg_EPE3D_acc = EPE3D_acc / (num_batch * batch_size * num_point)

    avg_source_deformation_err_mean = source_deformation_err[0] / num_batch / batch_size * 100   # m -> cm
    avg_source_deformation_err = source_deformation_err[1] / num_source_deformation * 100   # m -> cm
    avg_target_deformation_err_mean = target_deformation_err[0] / num_batch / batch_size * 100   # m -> cm
    avg_target_deformation_err = target_deformation_err[1] / num_target_deformation * 100   # m -> cm

    avg_source_geometry_err_mean = source_geometry_err[0] / num_batch / batch_size * 100   # m -> cm
    avg_source_geometry_err = source_geometry_err[1] / num_source_geometry * 100   # m -> cm
    avg_target_geometry_err_mean = target_geometry_err[0] / num_batch / batch_size * 100   # m -> cm
    avg_target_geometry_err = target_geometry_err[1] / num_target_geometry * 100   # m -> cm
    avg_total_err = total_err / num_batch
    print('total loss', avg_total_err)

    # summary result: value0 is the mean error, value1 is the numbers
    scene_flow_group, source_point_group, target_point_group = dict(), dict(), dict()
    for k, v in object_id_map.items():
        if scene_flow_dict[v][1] == 0:
            continue
        scene_flow_group[k] = [scene_flow_dict[v][0] / scene_flow_dict[v][1], scene_flow_dict[v][1]]
        source_point_group[k] = source_point_dict[v][0] / source_point_dict[v][1]
        target_point_group[k] = target_point_dict[v][0] / target_point_dict[v][1]
    print('scene flow group', scene_flow_group)
    print('source point group', source_point_group)
    print('target point group', target_point_group)

    for k, v in scene_flow_frames_dict.items():
        scene_flow_frames_dict[k][0] = scene_flow_frames_dict[k][0] / scene_flow_frames_dict[k][1]
        source_point_frames_dict[k][0] = source_point_frames_dict[k][0] / source_point_frames_dict[k][1]
        target_point_frames_dict[k][0] = target_point_frames_dict[k][0] / target_point_frames_dict[k][1]
    print('scene flow frames', scene_flow_frames_dict)
    print('source point frames', source_point_frames_dict)
    print('target point frames', target_point_frames_dict)

    print('EPE3d mean: ', avg_EPE3D_err_mean)
    print('source_def_mean: ', avg_source_deformation_err_mean, 'target_def_mean: ', avg_target_deformation_err_mean)
    print('source_geo_mean: ', avg_source_geometry_err_mean, 'target_geo_mean:', avg_target_geometry_err_mean)

    return avg_EPE3D_err, avg_EPE3D_acc, \
           avg_source_deformation_err, avg_target_deformation_err, \
           avg_source_geometry_err, avg_target_geometry_err


