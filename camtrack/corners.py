#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


MIN_DISTANCE=30

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=20,
                      qualityLevel=0.1,
                      minDistance=MIN_DISTANCE,
                      blockSize=5,
                      useHarrisDetector=False)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

class CornerUpdater:
    def __init__(self, ids, points, sizes):
        self.ids = ids
        self.points = points
        self.sizes = sizes

    def get(self):
        return self.ids, self.points, self.sizes

    def extend(self, id, point):
        self.ids = np.concatenate([self.ids, [id]])
        self.points = np.concatenate([self.points, [point]])
        self.sizes = np.concatenate([self.sizes, [10]])

    def update(self, extra_points, start_id):
        cur_id = start_id
        for p in extra_points:
            flag = False
            for pp in self.points:
                if np.linalg.norm(p - pp) < MIN_DISTANCE:
                    flag = True
                    break
            if not flag:
                self.extend(cur_id, p)
                cur_id += 1
        return cur_id


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]
    p0 = cv2.goodFeaturesToTrack(image_0, mask = None, **feature_params).squeeze(axis=1)
    p0_ids = np.array(list(range(len(p0))))
    corners = CornerUpdater(
        p0_ids,
        p0,
        np.array([10] * len(p0))
    )
    last_id = len(p0)
    builder.set_corners_at_frame(0, FrameCorners(*corners.get()))
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        p1, st, err = cv2.calcOpticalFlowPyrLK(to_int_gray(image_0), 
                                               to_int_gray(image_1), 
                                               corners.points, None, **lk_params)
        corners = CornerUpdater(corners.ids[st.flatten()==1], 
                                p1[st.flatten()==1], 
                                corners.sizes[st.flatten()==1])
        update_with = cv2.goodFeaturesToTrack(image_1, mask = None, **feature_params).squeeze(axis=1)
        last_id = corners.update(update_with, last_id)
        builder.set_corners_at_frame(frame, FrameCorners(*corners.get()))
        image_0 = image_1


def to_int_gray(image):
    return np.uint8(image * 255)


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
