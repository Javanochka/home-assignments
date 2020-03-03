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
from _corners import filter_frame_corners
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


MAX_PROJ_ERROR = 1.0
MIN_TRIANG_ANG = 1.0
MIN_DEPTH = 0.1


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

    correspondences = build_correspondences(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]])
    view_mat_1 = pose_to_view_mat3x4(known_view_1[1])
    view_mat_2 = pose_to_view_mat3x4(known_view_2[1])
    points, ids, median_cos = triangulate_correspondences(correspondences, 
                                                          view_mat_1, view_mat_2,
                                                          intrinsic_mat, 
                                                          triang_params)
    if len(points) < 10:
        print("\rPlease, try other frames")
        exit(0)

    view_mats = [None] * len(rgb_sequence)
    view_mats[known_view_1[0]] = view_mat_1
    view_mats[known_view_2[0]] = view_mat_2

    point_cloud_builder = PointCloudBuilder(ids, points)
    was_update = True    
    while was_update:
        was_update = False
        for i, (frame, corners) in enumerate(zip(rgb_sequence, corner_storage)):
            if view_mats[i] is not None:
                continue
            _, (idx_1, idx_2) = snp.intersect(point_cloud_builder.ids.flatten(), 
                                              corners.ids.flatten(), 
                                              indices=True)
            try:
                retval, rvec, tvec, inliers = cv2.solvePnPRansac(point_cloud_builder.points[idx_1],
                                                                 corners.points[idx_2],
                                                                 intrinsic_mat,
                                                                 distCoeffs=None)
                inliers = np.array(inliers, dtype=int)
                if len(inliers) > 0:
                    view_mats[i] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
                    was_update = True
                print(f"\rFrame {i} out of {len(rgb_sequence)}, inliners: {len(inliers)}", end="")
            except Exception:
                print(f"\rFrame {i} out of {len(rgb_sequence)}, inliners: {0}", end="")
            if view_mats[i] is None:
                continue
            cur_corner = filter_frame_corners(corner_storage[i], inliers.flatten())
            
            for j in range(len(rgb_sequence)):
                if view_mats[j] is None:
                    continue
                correspondences = build_correspondences(corner_storage[j], cur_corner)
                if len(correspondences.ids) == 0:
                    continue
                points, ids, median_cos = triangulate_correspondences(correspondences, 
                                                                      view_mats[j], view_mats[i],
                                                                      intrinsic_mat, 
                                                                      triang_params)
                point_cloud_builder.add_points(ids, points)
                
        print()

    first_mat = next((mat for mat in view_mats if mat is not None), None)
    if first_mat is None:
        print("\rFail")
        exit(0)

    view_mats[0] = first_mat
    for i in range(1, len(view_mats)):
        if view_mats[i] is None:
            view_mats[i] = view_mats[i - 1]
    view_mats = np.array(view_mats)

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
