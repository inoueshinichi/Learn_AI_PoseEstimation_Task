"""MS_COCOからOpenPose用のアノテーションデータを作成する
"""
import os
import sys

# os.sepはプラットフォーム固有の区切り文字(Windows: `\`, Unix: `/`)
module_parent_dir = os.sep.join([os.path.dirname(__file__), '..'])
# print("module_parent_dir", module_parent_dir)
sys.path.append(module_parent_dir)

from log_conf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

import copy
import csv
import functools
import glob
import math
import random
import json
from collections import namedtuple

# 人体の各ポイント(MS_COCOのアノテーションに対して, "neck": 0を追加している)
openpose_keypoints: dict = {
   "neck": 0,
   "nose": 1,
   "left_eye": 2,
   "right_eye": 3,
   "left_ear": 4, 
   "right_ear": 5,
   "left_shoulder": 6,
   "right_shoulder": 7,
   "left_elbow": 8,
   "right_elbow": 9,
   "left_wrist": 10,
   "right_wrist": 11,
   "left_hip": 12,
   "right_hip": 13,
   "left_knee": 14,
   "right_knee": 15,
   "left_ankle": 16,
   "right_angle": 17,
} # 18個

# 人体の各ポイントのリンク
openpose_skeltons: dict = {
    "neck_2_left_hip": 0,
    "left_hip_2_neck": 1,
    "left_hip_2_left_knee": 2,
    "left_knee_2_left_hip": 3,
    "left_knee_2_left_ankle": 4,
    "left_ankle_2_left_knee": 5,
    "neck_2_right_hip": 6,
    "right_hip_2_neck": 7,
    "right_hip_2_right_knee": 8,
    "right_knee_2_right_hip": 9,
    "right_knee_2_right_ankle": 10,
    "right_ankle_2_right_knee": 11,
    "neck_2_left_shoulder": 12,
    "left_shoulder_2_neck": 13,
    "left_shoulder_2_left_elbow": 14,
    "left_elbow_2_left_shoulder": 15,
    "left_elbow_2_left_wrist": 16,
    "left_wrist_2_left_elbow":17,
    "neck_2_right_shoulder": 20,
    "right_shoulder_2_neck": 21,
    "right_shoulder_2_right_elbow": 22,
    "right_elbow_2_right_shoulder": 23,
    "right_elbow_2_right_wrist": 24,
    "right_wrist_2_right_elbow": 25,
    "neck_2_nose": 28,
    "nose_2_neck": 29,
    "nose_2_left_eye": 30,
    "left_eye_2_nose": 31,
    "nose_2_right_eye": 32,
    "right_eye_2_nose": 33,
    "left_eye_2_left_ear": 36,
    "left_ear_2_left_eye": 37,
    "right_eye_2_right_ear": 38,
    "right_ear_2_right_eye": 39,
} # 38個?

# アノテーション
OpenPoseAnnoTuple = namedtuple(
    'OpenPoseAnnoTuple',
    [
        "index", # Optional[int]
        "id", # Optional[int]
        "dataset", # Optional[str]
        "phase", # Optional[str]
        "img_id", # Optional[int]
        "rel_img_path", # Optional[str]
        "width", # Optional[float]
        "height", # Optional[float]
        "obj_pos", # Optional[List[float]]
        "bbox", # Optioanl[List[float]]
        "segment_area", # Optional[float]
        "num_keypoints", # Optional[int]
        "joint_self", # Optional[List[float]]
        "scaled", # Optional[float]
        "iscrowd", # Optional[float]]
        "segmentation", # Optional[Union[List[float], Dict[List[float], List[float]]]]
        "skelton", # Optional[List[List[float]]]
    ]
)