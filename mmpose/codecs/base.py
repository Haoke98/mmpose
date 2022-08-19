# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np


class BaseKeypointCodec(metaclass=ABCMeta):
    """The base class of the keypoint codec.

    A keypoint codec is a module to encode keypoint coordinates to specific
    representation (e.g. heatmap) and vice versa. A subclass should implement
    the methods :meth:`encode` and :meth:`decode`.
    """

    @abstractmethod
    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> Any:
        """Encode keypoints.

        Note:

            - instance number: N
            - keypoint number: K
            - keypoint dimension: D

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibility in shape
                (N, K, D)
        """

    @abstractmethod
    def decode(self, encoded: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoints.

        Args:
            encoded (any): Encoded keypoint representation using the codec

        Returns:
            tuple:
            - keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            - keypoints_visible (np.ndarray): Keypoint visibility in shape
                (N, K, D)
        """

    def keypoints_bbox2img(self, keypoints: np.ndarray,
                           bbox_centers: np.ndarray,
                           bbox_scales: np.ndarray) -> np.ndarray:
        """Convert decoded keypoints from the bbox space to the image space.
        Topdown codecs should override this method.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D).
                The coordinate is in the bbox space
            bbox_centers (np.ndarray): BBox centers in shape (N, 2).
                See `pipelines.GetBBoxCenterScale` for details
            bbox_scale (np.ndarray): BBox scales in shape (N, 2).
                See `pipelines.GetBBoxCenterScale` for details

        Returns:
            np.ndarray: The transformed keypoints in shape (N, K, D).
            The coordinate is in the image space.
        """
        raise NotImplementedError