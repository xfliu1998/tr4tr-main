import open3d as o3d
import numpy as np


def get_valid_matrix(scene_flow_mask, source_point_proj2D):
    """
    get mask matrix of valid
    :param scene_flow_mask:
    :param source_point_proj2D:
    :return: valid_matrix
    """
    num_point = source_point_proj2D.shape[0]
    h, w = scene_flow_mask.shape[0], scene_flow_mask.shape[1]
    valid_matrix = np.full((num_point, ), False)
    for i in range(num_point):
        u, v = np.round(source_point_proj2D[i])
        u, v = int(u // (640/w)), int(v // (480/h))
        if 0 <= u < w and 0 <= v < h and scene_flow_mask[v, u, 0]:
            valid_matrix[i] = True
    return valid_matrix  # nx1


def get_registration_matrix(source_point, target_point, intrinsics):
    """
    get 2d registration matrix
    :param source_point:
    :param target_point:
    :param intrinsics:
    :return: registration_matrix, source_point_proj2D
    """
    # map 3d data to 2d data, x = KX / z_c
    source_point_rearr = source_point.transpose(1, 0)  # nx3 -> 3xn
    source_point_proj2D = np.divide(np.matmul(intrinsics, source_point_rearr),
                                    np.tile(source_point_rearr[2, :], (3, 1)))[:2, :]  # 2xn
    target_point_rearr = target_point.transpose(1, 0)  # nx3 -> 3xn
    target_point_proj2D = np.divide(np.matmul(intrinsics, target_point_rearr),
                                    np.tile(target_point_rearr[2, :], (3, 1)))[:2, :]  # 2xn
    registration_matrix = np.concatenate([source_point_proj2D.transpose(1, 0),
                                          target_point_proj2D.transpose(1, 0)], axis=1)
    source_point_proj2D = source_point_proj2D.transpose(1, 0)
    return registration_matrix, source_point_proj2D  # nx4


class CustomDrawGeometryWithKeyCallback:

    def __init__(self, geometry_dict, alignment_dict):
        self.added_source_pcd_all = True
        self.added_target_pcd_all = False
        self.added_source_pcd_gt = False
        self.added_target_pcd_gt = False
        self.added_source_pcd_pred = False
        self.added_target_pcd_pred = False

        self.added_source_gt_pred_align = False
        self.added_target_gt_pred_align = False
        self.added_source_target_gt_align = False
        self.added_source_target_pred_align = False
        self.added_source_gt_target_pred_align = False

        self.source_pcd_all = geometry_dict["source_pcd_all"]
        self.target_pcd_all = geometry_dict["target_pcd_all"]
        self.source_pcd_gt = geometry_dict["source_pcd_gt"]
        self.target_pcd_gt = geometry_dict["target_pcd_gt"]
        self.source_pcd_pred = geometry_dict["source_pcd_pred"]
        self.target_pcd_pred = geometry_dict["target_pcd_pred"]

        self.source_gt_pred_align = alignment_dict["source_gt_pred_align"]
        self.target_gt_pred_align = alignment_dict["target_gt_pred_align"]
        self.source_target_gt_align = alignment_dict["source_target_gt_align"]
        self.source_target_pred_align = alignment_dict["source_target_pred_align"]
        self.source_gt_target_pred_align = alignment_dict["source_gt_target_pred_align"]

    def custom_draw_geometry_with_key_callback(self):
        def view_source_all(vis):
            print('view source all')
            if not self.added_source_pcd_all:
                vis.add_geometry(self.source_pcd_all)
                self.added_source_pcd_all = True
            else:
                vis.remove_geometry(self.source_pcd_all)
                self.added_source_pcd_all = False
            return False

        def view_target_all(vis):
            print('view target all')
            if not self.added_target_pcd_all:
                vis.add_geometry(self.target_pcd_all)
                self.added_target_pcd_all = True
            else:
                vis.remove_geometry(self.target_pcd_all)
                self.added_target_pcd_all = False
            return False

        def view_source_gt(vis):
            print('view source gt')
            if not self.added_source_pcd_gt:
                vis.add_geometry(self.source_pcd_gt)
                self.added_source_pcd_gt = True
            else:
                vis.remove_geometry(self.source_pcd_gt)
                self.added_source_pcd_gt = False
            return False

        def view_target_gt(vis):
            print('view target gt')
            if not self.added_target_pcd_gt:
                vis.add_geometry(self.target_pcd_gt)
                self.added_target_pcd_gt = True
            else:
                vis.remove_geometry(self.target_pcd_gt)
                self.added_target_pcd_gt = False
            return False

        def view_source_pred(vis):
            print('view source pred')
            if not self.added_source_pcd_pred:
                vis.add_geometry(self.source_pcd_pred)
                self.added_source_pcd_pred = True
            else:
                vis.remove_geometry(self.source_pcd_pred)
                self.added_source_pcd_pred = False
            return False

        def view_target_pred(vis):
            print('view target pred')
            if not self.added_target_pcd_pred:
                vis.add_geometry(self.target_pcd_pred)
                self.added_target_pcd_pred = True
            else:
                vis.remove_geometry(self.target_pcd_pred)
                self.added_target_pcd_pred = False
            return False

        def align_source_gt_pred(vis):
            print('align source gt pred')
            if not self.added_source_pcd_gt:
                vis.add_geometry(self.source_pcd_gt)
                self.added_source_pcd_gt = True
            if not self.added_source_pcd_pred:
                vis.add_geometry(self.source_pcd_pred)
                self.added_source_pcd_pred = True
            if not self.added_source_gt_pred_align:
                vis.add_geometry(self.source_gt_pred_align)
                self.added_source_gt_pred_align = True
            else:
                vis.remove_geometry(self.source_gt_pred_align)
                self.added_source_gt_pred_align = False
                vis.remove_geometry(self.source_pcd_gt)
                self.added_source_pcd_gt = False
                vis.remove_geometry(self.source_pcd_pred)
                self.added_source_pcd_pred = False
            return False

        def align_target_gt_pred(vis):
            print('align target gt pred')
            if not self.added_target_pcd_gt:
                vis.add_geometry(self.target_pcd_gt)
                self.added_target_pcd_gt = True
            if not self.added_target_pcd_pred:
                vis.add_geometry(self.target_pcd_pred)
                self.added_target_pcd_pred = True
            if not self.added_target_gt_pred_align:
                vis.add_geometry(self.target_gt_pred_align)
                self.added_target_gt_pred_align = True
            else:
                vis.remove_geometry(self.target_gt_pred_align)
                self.added_target_gt_pred_align = False
                vis.remove_geometry(self.target_pcd_gt)
                self.added_target_pcd_gt = False
                vis.remove_geometry(self.target_pcd_pred)
                self.added_target_pcd_pred = False
            return False

        def align_source_target_gt(vis):
            print('align source target gt')
            if not self.added_source_pcd_gt:
                vis.add_geometry(self.source_pcd_gt)
                self.added_source_pcd_gt = True
            if not self.added_target_pcd_gt:
                vis.add_geometry(self.target_pcd_gt)
                self.added_target_pcd_gt = True
            if not self.added_source_target_gt_align:
                vis.add_geometry(self.source_target_gt_align)
                self.added_source_target_gt_align = True
            else:
                vis.remove_geometry(self.source_target_gt_align)
                self.added_source_target_gt_align = False
                vis.remove_geometry(self.source_pcd_gt)
                self.added_source_pcd_gt = False
                vis.remove_geometry(self.target_pcd_gt)
                self.added_target_pcd_gt = False
            return False

        def align_source_target_pred(vis):
            print('align source target pred')
            if not self.added_source_pcd_pred:
                vis.add_geometry(self.source_pcd_pred)
                self.added_source_pcd_pred = True
            if not self.added_target_pcd_pred:
                vis.add_geometry(self.target_pcd_pred)
                self.added_target_pcd_pred = True
            if not self.added_source_target_pred_align:
                vis.add_geometry(self.source_target_pred_align)
                self.added_source_target_pred_align = True
            else:
                vis.remove_geometry(self.source_target_pred_align)
                self.added_source_target_pred_align = False
                vis.remove_geometry(self.source_pcd_pred)
                self.added_source_pcd_pred = False
                vis.remove_geometry(self.target_pcd_pred)
                self.added_target_pcd_pred = False
            return False

        def align_source_gt_target_pred(vis):
            print('align source gt target pred')
            if not self.added_source_pcd_gt:
                vis.add_geometry(self.source_pcd_gt)
                self.added_source_pcd_gt = True
            if not self.added_target_pcd_pred:
                vis.add_geometry(self.target_pcd_pred)
                self.added_target_pcd_pred = True
            if not self.added_source_gt_target_pred_align:
                vis.add_geometry(self.source_gt_target_pred_align)
                self.added_source_gt_target_pred_align = True
            else:
                vis.remove_geometry(self.source_gt_target_pred_align)
                self.added_source_gt_target_pred_align = False
                vis.remove_geometry(self.source_pcd_gt)
                self.added_source_pcd_gt = False
                vis.remove_geometry(self.target_pcd_pred)
                self.added_target_pcd_pred = False
            return False

        def remove_all(vis):
            if not self.added_source_pcd_all:
                vis.add_geometry(self.added_source_pcd_all)
                self.added_source_pcd_all = True
            if self.added_target_pcd_all:
                vis.remove_geometry(self.added_target_pcd_all)
                self.added_target_pcd_all = False
            if self.added_source_pcd_gt:
                vis.remove_geometry(self.added_source_pcd_gt)
                self.added_source_pcd_gt = False
            if self.added_target_pcd_gt:
                vis.remove_geometry(self.added_target_pcd_gt)
                self.added_target_pcd_gt = False
            if self.added_source_pcd_pred:
                vis.remove_geometry(self.added_source_pcd_pred)
                self.added_source_pcd_pred = False
            if self.added_target_pcd_pred:
                vis.remove_geometry(self.added_target_pcd_pred)
                self.added_target_pcd_pred = False

            if self.added_source_gt_pred_align:
                vis.remove_geometry(self.added_source_gt_pred_align)
                self.added_source_gt_pred_align = False
            if self.added_target_gt_pred_align:
                vis.remove_geometry(self.added_target_gt_pred_align)
                self.added_target_gt_pred_align = False
            if self.added_source_target_gt_align:
                vis.remove_geometry(self.added_source_target_gt_align)
                self.added_source_target_gt_align = False
            if self.added_source_target_pred_align:
                vis.remove_geometry(self.added_source_target_pred_align)
                self.added_source_target_pred_align = False

            return False

        key_to_callback = {}
        key_to_callback[ord("A")] = view_source_all
        key_to_callback[ord("B")] = view_target_all
        key_to_callback[ord("C")] = view_source_gt
        key_to_callback[ord("D")] = view_target_gt
        key_to_callback[ord("E")] = view_source_pred
        key_to_callback[ord("F")] = view_target_pred
        key_to_callback[ord("G")] = align_source_gt_pred
        key_to_callback[ord("H")] = align_target_gt_pred
        key_to_callback[ord("I")] = align_source_target_gt
        key_to_callback[ord("J")] = align_source_target_pred
        key_to_callback[ord("K")] = align_source_gt_target_pred
        key_to_callback[ord("L")] = remove_all
        o3d.visualization.draw_geometries_with_key_callbacks([self.source_pcd_all], key_to_callback)





