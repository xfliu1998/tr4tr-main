import os
from skimage import io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import cv2
import re
import struct
import random
import numpy as np
import open3d as o3d
import json
from PIL import Image
from torchvision import transforms


def crop(img, img_size, crop_size, crop_type):
    """
    crop the image from img_size to crop_size with crop type
    :param img: size of (h, w, c)
    :param img_size:
    :param crop_size:
    :param crop_type:
    :return: crop_img
    """
    th, tw = crop_size
    h, w = img_size
    # center crop
    if crop_type == 'center':
        if len(img.shape) == 2:
            crop_img = img[(h - th) // 2:(h + th) // 2, (w - tw) // 2:(w + tw) // 2]
        else:
            crop_img = img[(h - th) // 2:(h + th) // 2, (w - tw) // 2:(w + tw) // 2, :]
    # down sample with the interpolation mode(INTER_NEAREST INTER_AREA0
    elif crop_type == 'inter_nearest':
        crop_img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_NEAREST)
    else:  # INTER_LINEAR
        crop_img = cv2.resize(img, (tw, th))
    return crop_img


def normalize(img, normal_type):
    """
    normalize data: default rgb/255, xyz not normalized
    :param img: size of (h, w, 6)
    :param normal_type:
    :return: img_sca
    """
    h, w = img.shape[0], img.shape[1]
    source_color_img = img[:, :, :3]
    source_depth_img = img[:, :, 3:]
    if normal_type == 'standard_scaler':
        color_img = source_color_img.reshape(h * w, 3)
        depth_img = source_depth_img.reshape(h * w, 3)
        std_sca1 = StandardScaler()
        std_sca2 = StandardScaler()
        color_img_sca = std_sca1.fit_transform(color_img)
        depth_img_sca = std_sca2.fit_transform(depth_img)
        color_img_sca = color_img_sca.reshape(h, w, 3)
        depth_img_sca = depth_img_sca.reshape(h, w, 3)
    elif normal_type == 'minMax_scalar':
        color_img = source_color_img.reshape(h * w, 3)
        depth_img = source_depth_img.reshape(h * w, 3)
        mm_sca1 = MinMaxScaler()
        mm_sca2 = MinMaxScaler()
        color_img_sca = mm_sca1.fit_transform(color_img)
        depth_img_sca = mm_sca2.fit_transform(depth_img)
        color_img_sca = color_img_sca.reshape(h, w, 3)
        depth_img_sca = depth_img_sca.reshape(h, w, 3)
    else:
        color_img_sca = source_color_img / 255
        depth_img_sca = source_depth_img
    img_sca = np.concatenate((color_img_sca, depth_img_sca, source_color_img, source_depth_img), axis=-1)
    return img_sca


def load_intrinsics(intrinsics_path):
    """
    load intrinsic matrix
    :param intrinsics_path:
    :return: K (intrinsics matrix)
    """
    with open(intrinsics_path) as f:
        data = f.readlines()
        k_x = data[0].split(' ')[0]
        k_y = data[1].split(' ')[1]
        u_0 = data[0].split(' ')[2]
        v_0 = data[1].split(' ')[1]
        intrinsics = list(map(float, [k_x, k_y, u_0, v_0]))
    K = np.zeros((3, 3))
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = np.array(intrinsics, dtype=np.float32)
    return K


def backproject_depth(depth_image, intrinsics):
    """
    map the depth image to a 3D point cloud image based on camera intrinsics
    :param depth_image:
    :param intrinsics:
    :return: point_image
    """
    assert len(depth_image.shape) == 2
    k_x, k_y, u_0, v_0 = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    height, width = depth_image.shape
    point_image = np.zeros((height, width, 3), dtype=np.float32)
    if depth_image.dtype != np.float32:
        depth_image = depth_image.astype(np.float32)

    for v in range(height):  # row = y
        for u in range(width):  # col = x
            if depth_image[v, u] == 0:
                continue
            depth = depth_image[v, u]
            z_c = depth / 1000   # unit is m
            x_c = (u - u_0) * z_c / k_x
            y_c = (v - v_0) * z_c / k_y
            point_image[v, u] = np.array([x_c, y_c, z_c], dtype=np.float32)
    return point_image


def load_image(color_image_path, depth_image_path, intrinsics):
    """
    according to the image path to get the picture and processing
    :param color_image_path:
    :param depth_image_path:
    :param intrinsics:
    :return: image
    """
    color_image = io.imread(color_image_path)  # (h, w, 3)  RGB
    depth_image = io.imread(depth_image_path)  # (h, w)
    # the depth image uses Gaussian filtering
    # depth_image = cv2.GaussianBlur(depth_image, (3, 3), 1)
    depth_image = backproject_depth(depth_image, intrinsics)  # (h, w, 3)  xyz
    image = np.concatenate((color_image, depth_image), axis=-1)  # (h, w, 6)
    return image  # (h, w, 6)


def farthestPointDownSample(vertices, num_point_sampled):
    """
    sample num_point_sampled points with fastest point down sample
    :param vertices:
    :param num_point_sampled:
    :return: flags
    """
    # vertices.shape = (N,3) or (N,2)
    N = len(vertices)
    n = num_point_sampled
    assert n <= N, "Num of sampled point should be less than or equal to the size of vertices."
    _G = np.mean(vertices, axis=0)  # centroid of vertices
    _d = np.linalg.norm(vertices - _G, axis=1, ord=2)
    # take the point farthest from the center of gravity as the starting point
    farthest = np.argmax(_d)
    distances = np.inf * np.ones((N,))
    # be selected or not
    flags = np.zeros((N,), np.bool_)
    for i in range(n):
        flags[farthest] = True
        distances[farthest] = 0.
        p_farthest = vertices[farthest]
        dists = np.linalg.norm(vertices[~flags] - p_farthest, axis=1, ord=2)
        distances[~flags] = np.minimum(distances[~flags], dists)
        farthest = np.argmax(distances)
    return flags  # vertices[flags]


def generate_point(src_image, scene_flow_image, scene_flow_mask, num_point):
    """
    get the aligned point cloud
    :param src_image:
    :param scene_flow_image:
    :param scene_flow_mask:
    :param num_point:
    :return: point_cloud
    """
    src_point = src_image[..., 9:][scene_flow_mask]
    tgt_point = src_point + scene_flow_image[scene_flow_mask]
    src_point = src_point.reshape(-1, 3)
    tgt_point = tgt_point.reshape(-1, 3)
    src_pc = o3d.geometry.PointCloud()
    src_pc.points = o3d.utility.Vector3dVector(src_point)

    down_sample_points = 1
    uni_down_src_pc = src_pc.uniform_down_sample(every_k_points=down_sample_points)
    # radius outlier elimination
    cl, ind = uni_down_src_pc.remove_radius_outlier(nb_points=15, radius=0.02)
    # remove zero
    for i, point in enumerate(uni_down_src_pc.points):
        if (point == np.zeros((3, ))).all() and i in ind:
            ind.remove(i)
    inlier_cloud = np.asarray(uni_down_src_pc.points)[ind]
    src_point = src_point[ind]
    tgt_point = tgt_point[ind]

    if len(ind) > num_point:
        # fastest point down sample
        flags = farthestPointDownSample(inlier_cloud, num_point)
        # random sample
        # flags = random.sample(range(len(ind)), num_point)
        src_point = src_point[flags]
        tgt_point = tgt_point[flags]
    elif len(ind) < (num_point // 4):
        return None
    else:
        repeat = num_point // len(ind)
        mod = num_point % len(ind)
        src_point = np.insert(np.repeat(src_point, repeat, axis=0),
                              np.arange(0, (mod - 1) * repeat + 1, repeat), src_point[:mod], axis=0)
        tgt_point = np.insert(np.repeat(tgt_point, repeat, axis=0),
                              np.arange(0, (mod - 1) * repeat + 1, repeat), tgt_point[:mod], axis=0)

    point_cloud = np.concatenate((src_point, tgt_point), axis=-1)
    return point_cloud


def data_transform(src_image, tgt_image):
    """
    data augment
    :param src_image:
    :param tgt_image:
    :return: src_image, tgt_image
    """
    tf = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))
    ])

    h = src_image.shape[0]
    color_image = np.concatenate((src_image[..., :3], tgt_image[..., :3]), axis=0)  # (2h, w, 3)
    image_PIL = Image.fromarray(color_image, mode='RGB')
    transform_image = tf(image_PIL)
    transform_image = np.array(transform_image)
    src_image[..., :3], tgt_image[..., :3] = transform_image[:h], transform_image[h:]
    return src_image, tgt_image


def data_reverse(src_image, tgt_image, point_cloud):
    """
    reverse data for data augment
    :param src_image:
    :param tgt_image:
    :param point_cloud:
    :return:
    """
    reverse = (random.random() < 0.5)  # [0, 1) random
    if reverse:
        src_image, tgt_image = tgt_image, src_image
        point_cloud_copy = np.copy(point_cloud)
        point_cloud[..., :3], point_cloud[..., 3:] = point_cloud_copy[..., 3:], point_cloud_copy[..., :3]
    return src_image, tgt_image, point_cloud


# Copyright (c) 2020 Aljaz Bozic, Pablo Palafox
def load_PFM(file):
    file = open(file, 'rb')
    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def load_flow_binary(filename):
    # Flow is stored row-wise in order [channels, height, width].
    assert os.path.isfile(filename), "File not found: {}".format(filename)
    with open(filename, 'rb') as fin:
        width = struct.unpack('I', fin.read(4))[0]
        height = struct.unpack('I', fin.read(4))[0]
        channels = struct.unpack('I', fin.read(4))[0]
        n_elems = height * width * channels
        flow = struct.unpack('f' * n_elems, fin.read(n_elems * 4))
        flow = np.asarray(flow, dtype=np.float32).reshape([channels, height, width])
    return flow


def load_flow_middlebury(filename):
    f = open(filename, 'rb')
    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')
    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()
    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    return flow.astype(np.float32)


def process_flow(flow_image_path):
    if flow_image_path.endswith('.pfm') or flow_image_path.endswith('.PFM'):
        return load_PFM(flow_image_path)[0][:, :, 0:2]
    elif flow_image_path.endswith('.oflow') or flow_image_path.endswith('.OFLOW'):
        return load_flow_binary(flow_image_path)
    elif flow_image_path.endswith('.sflow') or flow_image_path.endswith('.SFLOW'):
        return load_flow_binary(flow_image_path)
    elif flow_image_path.endswith('.flo') or flow_image_path.endswith('.FLO'):
        return load_flow_middlebury(flow_image_path)
    else:
        print("Wrong flow extension: {}".format(flow_image_path))
        exit()


def load_flow(flow_image_path):
    """
    load flow data
    :param flow_image_path:
    :return: flow_image
    """
    flow_image = process_flow(flow_image_path)  # (2/3, h, w)
    flow_image = np.moveaxis(flow_image, 0, -1)  # (h, w, 2/3)
    return flow_image


def cal_flow_mask(flow_image):
    """
    Compute flow mask.
    :param flow_image:
    :return: flow_mask
    """
    flow_mask = np.isfinite(flow_image)  # (h, w, 2/3)
    if flow_mask.shape[2] == 2:
        flow_mask = np.logical_and(flow_mask[..., 0], flow_mask[..., 1])  # (h, w)
        flow_mask = flow_mask[..., np.newaxis]  # (h, w, 1)
        flow_mask = np.repeat(flow_mask, 2, axis=2)  # (h, w, 2)
    elif flow_mask.shape[2] == 3:
        flow_mask = np.logical_and(flow_mask[..., 0], flow_mask[..., 1], flow_mask[..., 2])  # (h, w)
        flow_mask = flow_mask[..., np.newaxis]  # (h, w, 1)
        flow_mask = np.repeat(flow_mask, 3, axis=2)  # (h, w, 3)

    # set invalid pixels to zero in the flow image
    flow_image[flow_mask is False] = 0.0

    return flow_mask  # (h, w, 2/3)


def mask_src_image(src_image, mask_image):
    """
    mask image
    :param src_image:
    :param mask_image:
    :return: src_image
    """
    mask_image_repeat = np.tile(mask_image, 4)   # (h, w, 3) -> (h, w, 12)
    src_image = np.where(mask_image_repeat, src_image, 0.)  # (h, w, 12)
    return src_image


def load_mask(mask_image_path):
    """
    load mask data
    :param mask_image_path:
    :return: mask_image
    """
    mask_image = cv2.imread(mask_image_path)
    return mask_image / 255.  # (h, w, 3)


def mask_rgbd(src_image, tgt_image, optical_flow_image, scene_flow_mask):
    """
    use flow data to compute target image mask
    :param src_image:
    :param tgt_image:
    :param optical_flow_image:
    :param scene_flow_mask:
    :return: src_image, tgt_image
    """
    # (h, w, 12) normal_RGB+normal_xyz+rgb+xyz
    scene_flow_mask = np.tile(scene_flow_mask, 4)  # (h, w, 3) -> (h, w, 12)
    src_image = np.where(scene_flow_mask, src_image, 0.)
    mask_target = True
    if mask_target:
        height, width = src_image.shape[:2]
        tgt_mask = np.full((height, width, 1), False)

        for x in range(width):
            for y in range(height):
                if scene_flow_mask[y, x, 0]:
                    tgt_y, tgt_x = int(y + optical_flow_image[y, x, 1]), int(x + optical_flow_image[y, x, 0])
                    if 0 <= tgt_y < height - 1 and 0 <= tgt_x < width - 1:
                        tgt_mask[tgt_y, tgt_x] = True
                        tgt_mask[tgt_y + 1, tgt_x] = True
                        tgt_mask[tgt_y, tgt_x + 1] = True
                        tgt_mask[tgt_y + 1, tgt_x + 1] = True
        tgt_mask = np.tile(tgt_mask, 12)  # (h, w, 3) -> (h, w, 12)
        tgt_image = np.where(tgt_mask, tgt_image, 0.)
    return src_image, tgt_image


def display_inlier_outlier(cloud, ind):
    """
    visualization of outliers points
    :param cloud:
    :param ind:
    :return:
    """
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    # unselected points are red
    outlier_cloud.paint_uniform_color([1, 0, 0])
    # selected points are grey
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def remove_noisy_point(src_image, tgt_image):
    """
    remove noisy point
    :param src_image:
    :param tgt_image:
    :return: src_image, tgt_image
    """
    height, width = src_image.shape[:2]
    src_point, tgt_point = src_image[..., 3:6], tgt_image[..., 3:6]
    src_point = src_point.reshape(height * width, 3)
    tgt_point = tgt_point.reshape(height * width, 3)
    src_pc = o3d.geometry.PointCloud()
    tgt_pc = o3d.geometry.PointCloud()
    src_pc.points = o3d.utility.Vector3dVector(src_point)
    tgt_pc.points = o3d.utility.Vector3dVector(src_point)

    down_sample_points = 1
    uni_down_src_pc = src_pc.uniform_down_sample(every_k_points=down_sample_points)
    uni_down_tgt_pc = tgt_pc.uniform_down_sample(every_k_points=down_sample_points)
    # cl, src_ind = uni_down_src_pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
    # cl, tgt_ind = uni_down_tgt_pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
    cl, src_ind = uni_down_src_pc.remove_radius_outlier(nb_points=20, radius=0.04)
    cl, tgt_ind = uni_down_tgt_pc.remove_radius_outlier(nb_points=20, radius=0.04)
    # visualization
    # display_inlier_outlier(src_pc, src_ind)
    # display_inlier_outlier(tgt_pc, tgt_ind)

    for i in range(height * width):
        if i not in src_ind and i + down_sample_points - 1 < height * width:
            src_point[i: i+down_sample_points] = np.zeros((down_sample_points, 3), dtype=np.float32)
        if i not in tgt_ind and i + down_sample_points - 1 < height * width:
            tgt_point[i: i+down_sample_points] = np.zeros((down_sample_points, 3), dtype=np.float32)
    src_point = src_point.reshape(height, width, 3)
    tgt_point = tgt_point.reshape(height, width, 3)
    src_image[..., 3:6] = src_point
    tgt_image[..., 3:6] = tgt_point
    return src_image, tgt_image


def load_matches(data, data_mode):
    """
    load matches pairs from data
    :param data:
    :param data_mode:
    :return: expand_array
    """
    array_len = 130 if data_mode == 'train' else 40
    expand_array = np.zeros((array_len, 4))
    if len(data) > 0:
        expand_array[:len(data)] = np.array([[data[i]['source_x'], data[i]['source_y'],
                                              data[i]['target_x'], data[i]['target_y']] for i in range(len(data))])
    return expand_array


def load_occlusions(data, data_mode):
    """
    load occlusion pairs from data
    :param data:
    :param data_mode:
    :return: expand_array
    """
    array_len = 130 if data_mode == 'train' else 20
    expand_array = np.zeros((array_len, 2))
    if len(data) > 0:
        expand_array[:len(data)] = np.array([[data[i]['source_x'], data[i]['source_y']] for i in range(len(data))])
    return expand_array


if __name__ == '__main__':
    # train/val json data
    data_path = '/media/small/HDD11/PycharmProjects/Data/Deepdeform/'

    data_mode_list = ['/train', '/val']
    for data_mode in data_mode_list:
        file_list = ['_alignments', '_masks', '_matches', '_occlusions', '_selfsupervised']

        # read file
        filename = data_path + data_mode
        for i in range(len(file_list)):
            with open(filename + file_list[i] + '.json', 'r', encoding='utf-8', newline='\n') as f:
                if i == 0: alignments = json.load(f)
                elif i == 1: masks = json.load(f)
                elif i == 2: matches = json.load(f)
                elif i == 3: occlusions = json.load(f)
                elif i == 4: selfsupervised = json.load(f)

        # train: 4540 3816 5827 4690 7930  val: 683 308 714 235 0
        print(len(alignments), len(masks), len(matches), len(occlusions), len(selfsupervised))

        # 1. alignments + self-supervied
        data = alignments + selfsupervised

        # 2. add matches data
        for i in range(len(matches)):
            for j in range(len(data)):
                if 'matches' not in data[j]:
                    data[j]['matches'] = []
                if matches[i]['seq_id'] == data[j]['seq_id'] \
                    and matches[i]['source_id'] == data[j]['source_id'] \
                    and matches[i]['target_id'] == data[j]['target_id'] \
                    and matches[i]['object_id'] == data[j]['object_id']:
                    data[j]['matches'] += matches[i]['matches']
                    # continue

        # 3. add occlusions data
        for i in range(len(occlusions)):
            for j in range(len(data)):
                if 'occlusions' not in data[j]:
                    data[j]['occlusions'] = []
                if occlusions[i]['seq_id'] == data[j]['seq_id'] \
                    and occlusions[i]['source_id'] == data[j]['source_id'] \
                    and occlusions[i]['target_id'] == data[j]['target_id'] \
                    and occlusions[i]['object_id'] == data[j]['object_id']:
                    data[j]['occlusions'] += occlusions[i]['occlusions']
                    # continue

        # 4. add masks data
        for i in range(len(masks)):
            for j in range(len(data)):
                if 'mask' not in data[j]:   # initialization
                    # data[j]['mask'] = []   # multiple mask
                    data[j]['mask'] = ''
                    data[j]['num_mask'] = 0
                if masks[i]['seq_id'] == data[j]['seq_id'] and masks[i]['frame_id'] == data[j]['source_id']:
                    data[j]['num_mask'] += 1
                    # data[j]['mask'].append(masks[i]['mask'])
                    if masks[i]['object_id'] == data[j]['object_id']:
                        data[j]['mask'] = masks[i]['mask']
                        # continue

        #  max length of the matches and occlusions
        max_matches_len, max_occu_len = 0, 0
        for i in range(len(data)):
            max_matches_len = max(max_matches_len, len(data[i]['matches']))
            max_occu_len = max(max_occu_len, len(data[i]['occlusions']))
        print(max_matches_len, max_occu_len)  # train: 126 120    val: 36 16

        # write file
        with open(data_path + data_mode +'.json', "w", encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(data, ensure_ascii=False, indent='\t'))
            print('%s totals %d data' % (data_mode[1:], len(data)))    # train 12470  val 683


































