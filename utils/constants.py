"""
Constants.

Authors: Hongjie Fang.
"""

import numpy as np


LOSS_INF = 1e18
DILATION_KERNEL = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.uint8)
