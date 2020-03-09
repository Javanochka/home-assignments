#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import sortednp as snp
import cv2
import click
import pims

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
    pose_to_view_mat3x4,
    eye3x4
)


MAX_PROJ_ERROR = 3.0
MIN_TRIANG_ANG = 5
MIN_DEPTH = 0.00001


def init_tracking(intrinsic_mat: np.ndarray,
                  corner_storage: CornerStorage,
                  rgb_sequence: pims.FramesSequence):
    triang_params = TriangulationParameters(max_reprojection_error=MAX_PROJ_ERROR, 
                                            min_triangulation_angle_deg=5, 
                                            min_depth=MIN_DEPTH)
    #print(np.cos(np.deg2rad(5)))
    frame_1 = 0
    view_mat_1 = eye3x4()
    min_cos = 1
    max_len_ps = 0
    for i in range(1, len(rgb_sequence)): # Ok, I would like to choose the best pair
        # First, find the similar points in the two pictures
        correspondences = build_correspondences(corner_storage[frame_1], corner_storage[i])
        if len(correspondences.ids) < 10:
            continue
        # Next, compute fundamental matrix
        E, mask = cv2.findEssentialMat(correspondences.points_1,
                                       correspondences.points_2,
                                       intrinsic_mat,
                                       threshold=MAX_PROJ_ERROR) # RANSAC inside by default
        retval, R, t, mask = cv2.recoverPose(E, correspondences.points_1, correspondences.points_2)
        view_mat_2 = np.hstack((R, -t))
        points, ids, median_cos = triangulate_correspondences(correspondences, 
                                                              view_mat_1, view_mat_2,
                                                              intrinsic_mat, 
                                                              triang_params)
        min_cos = min(min_cos, median_cos)
        max_len_ps = max(max_len_ps, len(points))
        print(f"\r{i} {retval} {len(points)} {median_cos}", end="")
        if len(points) > 10:
            print()
            return (frame_1, view_mat3x4_to_pose(view_mat_1)), (i, view_mat3x4_to_pose(view_mat_2)) 
    print("\n", min_cos, max_len_ps)
    return None, None


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = init_tracking(intrinsic_mat, corner_storage, rgb_sequence)
        if known_view_1 is None:
            print("Something went wrong.")
            exit(0)

    print(f"Got start frames: {known_view_1[0]} {known_view_2[0]}")
    

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
                                                                 distCoeffs=None,
                                                                 reprojectionError=MAX_PROJ_ERROR)
                inliers = np.array(inliers, dtype=int)
                if len(inliers) > 0:
                    view_mats[i] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
                    was_update = True
                print(f"\rFrame {i} out of {len(rgb_sequence)}, inliners: {len(inliers)}", end="")
            except Exception:
                print(f"\rFrame {i} out of {len(rgb_sequence)}, inliners: {0}", end="")
            if view_mats[i] is None:
                continue
            #cur_corner = filter_frame_corners(corner_storage[i], inliers.flatten())
            
            for j in range(len(rgb_sequence)):
                if view_mats[j] is None:
                    continue
                correspondences = build_correspondences(corner_storage[j], corner_storage[i])
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
