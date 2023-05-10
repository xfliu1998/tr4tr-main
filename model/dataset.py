from torch.utils.data import Dataset
from utils.data_utils import *
import numpy as np


class DeformDataset(Dataset):

    def __init__(self, data_path, data_mode, input_size, crop_type, normal_type,
                 transform, flow_reverse, is_mask, num_point):
        super(DeformDataset, self).__init__()
        self.data_path = data_path
        self.data_mode = data_mode
        self.input_height = input_size[0]
        self.input_width = input_size[1]
        self.crop_type = crop_type
        self.normal_type = normal_type
        self.transform = transform
        self.flow_reverse = flow_reverse
        self.is_mask = is_mask
        self.num_point = num_point
        self.load_json()

    def load_json(self):
        with open(os.path.join(self.data_path, self.data_mode + '.json')) as f:
            self.labels = json.loads(f.read())

    def __len__(self):
        return len(self.labels)

    def get_metadata(self, index):
        return self.labels[index]

    def __getitem__(self, index):
        data = self.labels[index]
        # object id of the evaluate set
        object_id_map = {'Jacket': 0, 'shirt': 1, 'gloves': 2, 'towel': 3, 'sweater': 4, 'shorts': 5, 'adult': 6}
        object_id = object_id_map.get(data["object_id"], 7)
        frames = abs(int(data["target_id"]) - int(data["source_id"]))

        # load the color and depth images
        src_color_image_path = os.path.join(self.data_path, data["source_color"])
        src_depth_image_path = os.path.join(self.data_path, data["source_depth"])
        tgt_color_image_path = os.path.join(self.data_path, data["target_color"])
        tgt_depth_image_path = os.path.join(self.data_path, data["target_depth"])
        # load the intrinsics of the camera
        intrinsics_path = os.path.join(self.data_path,
                                       data["source_color"].split('/')[0] + '/' + data["source_color"].split('/')[1]
                                       + '/intrinsics.txt')
        intrinsics = load_intrinsics(intrinsics_path)

        src_image = load_image(src_color_image_path, src_depth_image_path, intrinsics)
        tgt_image = load_image(tgt_color_image_path, tgt_depth_image_path, intrinsics)
        if self.transform:
            src_image, tgt_image = data_transform(src_image, tgt_image)

        # crop and normalize
        img_size = src_image.shape[0], src_image.shape[1]
        crop_size = self.input_height, self.input_width
        src_image = crop(src_image, img_size, crop_size, self.crop_type)
        tgt_image = crop(tgt_image, img_size, crop_size, self.crop_type)
        raw_image = np.concatenate([src_image.copy(), tgt_image.copy()], axis=2)
        # use the normalized data during training
        src_image = normalize(src_image, normal_type=self.normal_type)
        tgt_image = normalize(tgt_image, normal_type=self.normal_type)

        if self.data_mode == 'test':
            return {
                "object": np.array(object_id, dtype=np.int32),
                "frames": np.array(frames, dtype=np.int32),
                "raw_image": raw_image,  # (h, w, 6) raw_source+raw_target
                "source": src_image,     # (h, w, 12)  normal_RGB+normal_xyz+rgb+xyz
                "target": tgt_image,     # (h, w, 12)  normal_RGB+normal_xyz+rgb+xyz
                "intrinsics": intrinsics,  # (3, 3) [k_x, k_y, u_0, v_0]
                "index": np.array(index, dtype=np.int32)
            }
        else:  # 'train' or 'valuate'
            optical_flow_image_path = os.path.join(self.data_path, data["optical_flow"])
            scene_flow_image_path = os.path.join(self.data_path, data["scene_flow"])
            # the unit of the 2d data is pixel
            optical_flow_image = load_flow(optical_flow_image_path)
            # the unit of the 3d data is m
            scene_flow_image = load_flow(scene_flow_image_path)
            # optical flow data also need to crop
            optical_flow_image = crop(optical_flow_image, img_size, crop_size, self.crop_type)
            optical_flow_image = optical_flow_image / (480 / self.input_height)
            scene_flow_image = crop(scene_flow_image, img_size, crop_size, self.crop_type)

            # calculate mask
            optical_flow_mask = cal_flow_mask(scene_flow_image)
            scene_flow_mask = cal_flow_mask(scene_flow_image)

            # Check that flow mask is valid for at least one pixel.
            assert np.sum(optical_flow_mask) > 0, "Zero flow mask for sample: " + json.dumps(data)

            # self-supervised data without mask, matches and occlusion
            if data["mask"] == '':
                mask_image = np.ones((self.input_height, self.input_width, 3))
            else:
                mask_image_path = os.path.join(self.data_path, data["mask"])
                mask_image = load_mask(mask_image_path)
                mask_image = crop(mask_image, img_size, crop_size, self.crop_type)
            # only predict the single target if there are multiple masks
            if data["num_mask"] > 1:
                src_image = mask_src_image(src_image, mask_image)

            # mask the original rgbd image
            if self.is_mask:
                src_image, tgt_image = mask_rgbd(src_image, tgt_image, optical_flow_image, scene_flow_mask)

            # it will take lots of time to remove the noise of the point cloud
            # src_image, tgt_image = remove_noisy_point(src_image, tgt_image)

            # get the aligned point cloud ground truth according to the scene flow gt
            point_cloud = generate_point(src_image, scene_flow_image, scene_flow_mask, self.num_point)

            # skip the exception sample
            if point_cloud is None:  # 跳过异常样本
                # print('error ', index, src_color_image_path, tgt_color_image_path)
                return self.__getitem__(index + 1)

            # flow augment
            if self.flow_reverse:
                src_image, tgt_image, point_cloud = data_reverse(src_image, tgt_image, point_cloud)

            matches_array = load_matches(data['matches'], self.data_mode)
            occlusions_array = load_occlusions(data['occlusions'], self.data_mode)

            return {
                "object": np.array(object_id, dtype=np.int32),
                "frames": np.array(frames, dtype=np.int32),
                "raw_image": raw_image,                  # (h, w, 12)   raw_source+raw_target
                "source": src_image,                     # (h, w, 12)  normal_RGB+normal_xyz+rgb+xyz
                "target": tgt_image,                     # (h, w, 12)  normal_RGB+normal_xyz+rgb+xyz
                "point_cloud": point_cloud,              # (num_point, 6) source+target
                "optical_flow_gt": optical_flow_image,   # (h, w, 2)
                "optical_flow_mask": optical_flow_mask,  # (h, w, 2)
                "scene_flow_gt": scene_flow_image,       # (h, w, 3)
                "scene_flow_mask": scene_flow_mask,      # (h, w, 3)
                "intrinsics": intrinsics,                # [k_x, k_y, u_0, v_0]
                "mask": mask_image,                      # (h, w, 3)
                "matches": matches_array,                # [[srcx, srcy, tgtx, tgty],...]  (130/40, 4)
                "occlusions": occlusions_array,          # [[srcx, srcy],...]  (130/20, 2)
                "index": np.array(index, dtype=np.int32)
            }
