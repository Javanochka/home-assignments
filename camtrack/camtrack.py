#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import sortednp as snp
import cv2
import click

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4,
    pose_to_view_mat3x4
)

INTERVAL = 20
MAX_PROJ_ERROR = 30
MIN_TRIANG_ANG = 0.1
MIN_DEPTH = 0.001


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    triang_params = TriangulationParameters(max_reprojection_error=MAX_PROJ_ERROR, 
                                            min_triangulation_angle_deg=MIN_TRIANG_ANG, 
                                            min_depth=MIN_DEPTH)

    # TODO: implement
    correspondences = build_correspondences(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]])
    view_mat_1 = pose_to_view_mat3x4(known_view_1[1])
    view_mat_2 = pose_to_view_mat3x4(known_view_2[1])
    points, ids, median_cos = triangulate_correspondences(correspondences, 
                                                          view_mat_1, view_mat_2,
                                                          intrinsic_mat, 
                                                          triang_params)

    view_mats, point_cloud_builder = [], PointCloudBuilder(ids, points)
    for i, (frame, corners) in enumerate(zip(rgb_sequence, corner_storage)):
        _, (idx_1, idx_2) = snp.intersect(point_cloud_builder.ids.flatten(), 
                                          corners.ids.flatten(), 
                                          indices=True)
        try:
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(point_cloud_builder.points[idx_1],
                                                             corners.points[idx_2],
                                                             intrinsic_mat,
                                                             distCoeffs=None)
            inliers = np.array(inliers, dtype=int)
            point_cloud_builder.update_points(inliers, point_cloud_builder.points[idx_1][inliers.flatten()])
            view_mats.append(rodrigues_and_translation_to_view_mat3x4(rvec, tvec))
            print(f"\rFrame {i} out of {len(rgb_sequence)}, inliners: {len(inliers)}", end="")
        except Exception:
            if i == 0:
                print("\rPlease, try other frames")
                break
            view_mats.append(view_mats[-1])
            print(f"\rFrame {i} out of {len(rgb_sequence)}, inliners: {0}", end="")
        if i >= INTERVAL:
            j = i - INTERVAL
            correspondences = build_correspondences(corner_storage[j], corner_storage[i])
            points, ids, median_cos = triangulate_correspondences(correspondences, 
                                                                  view_mats[j], view_mats[i],
                                                                  intrinsic_mat, 
                                                                  triang_params)
            point_cloud_builder.add_points(ids, points)
            
    print()
    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
