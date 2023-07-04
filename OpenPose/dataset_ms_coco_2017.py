"""OpenPose用MS_COCOのデータセット
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

from diskcache import FanoutCache
import numpy as np
from PIL import Image

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

from type_hint import *
from disk import make_disk_cache

dataset: str = "MS_COCO"
year: str = '2017'
scope: str = "keypoints"
task: str = f"{dataset}_{year}_{scope}"
version: str = "1.0"
disk_cache_tag: str = f"{task}_{version}"

# 生データセットのディスクキャッシュ
gzip_cache: FanoutCache = make_disk_cache(
    cache_dir="F:/OpenPose/Cache",                                       
    target=task,
    version=version,
    )

# 人体の各ポイント(MS_COCOのアノテーションに対して, "neck": 0を追加している)
keypoints: dict = {
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
skeltons: dict = {
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
OpenPoseKeypintsAnnoTuple = namedtuple(
    'OpenPoseKeypintsAnnoTuple',
    [
        "index", # Optional[int]
        "id", # Optional[int]
        "dataset", # Optional[str]
        "phase", # Optional[str]
        "img_id", # Optional[int]
        "img_path", # Optional[str]
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

# データレコード
OpenPoseRecordTuple = namedtuple(
    'OpenPoseRecordTuple',
    [
        'annotation',
        'heatmap', 
        'heatmap_mask', 
        'pafs', 
        'pafs_mask'
    ]
)

# データセットの取得
@functools.lru_cache(1)
def get_anno_tuple_list(
    dataset_dir: str = "",
    phase: str = "trn",
    is_require_on_disk: bool = True,
    ):

    year: str = dataset_dir.split(os.sep)[-1]
    anno_file: str = "{}{}_anno.json".format(phase, year)

    """データセットへのアクセス"""
    anno_tuple_list: list = []
    if is_require_on_disk:

        # JSON形式のアノテーションファイルを読み込む
        anno_json_file = os.sep.join([
            dataset_dir, 
            "openpose_keypoints_anno",
            anno_file,
            ])
        
        with open(anno_json_file) as f:
            data_this = json.load(f)
        
        data_info = data_this["info"]
        data_license = data_this["license"]
        data_anno_array = data_this["annotations"]
        
        num_samples: int = len(data_anno_array)


        for ndx in range(num_samples):
            anno_tuple = OpenPoseKeypintsAnnoTuple(
                index=ndx,
                id=data_anno_array[ndx]['id'],
                dataset=data_anno_array[ndx]['dataset'],
                phase=data_anno_array[ndx]['phase'],
                img_id=data_anno_array[ndx]['img_id'],
                img_path=data_anno_array[ndx]['img_path'],
                width=data_anno_array[ndx]['width'],
                height=data_anno_array[ndx]['height'],
                obj_pos=data_anno_array[ndx]['obj_pos'],
                bbox=data_anno_array[ndx]['bbox'],
                segment_area=data_anno_array[ndx]['segment_area'],
                num_keypoints=data_anno_array[ndx]['num_keypoints'],
                joint_self=data_anno_array[ndx]['joint_self'],
                scaled=data_anno_array[ndx]['scaled'],
                iscrowd=data_anno_array[ndx]['iscrowd'],
                segmentation=data_anno_array[ndx]['segmentation'],
                skelton=data_anno_array[ndx]['skelton'],
            )

            anno_tuple_list.append(anno_tuple)
        
    return anno_tuple_list


class OpenPoseKeypointsData:
    def __init__(self,
                 dataset_dir: str,
                 anno_tuple: OpenPoseKeypintsAnnoTuple,
                 ):
        self.dataset_dir = dataset_dir
        self.anno_tuple = anno_tuple
        self.pil_img: Optional[Image] = None
        self.np_img: Optional[np.ndarray] = None
        self.np_heatmap: Optional[np.ndarray] = None
        self.np_heatmap_mask: Optional[np.ndarray] = None
        self.np_pafs: Optional[np.ndarray] = None
        self.np_pafs_mask: Optional[np.ndarray] = None

    def _get_raw_img(self):
        img_path: str = os.sep.join([
            self.dataset_dir,
            self.anno_tuple.img_path,
            ])
                       
        self.pil_img= Image.open(img_path)
        self.np_img = np.array(self.pil_img) # (C,H,W)

    def _build_heatmap(self):
        pass

    def _build_heatmap_mask(self):
        pass

    def _build_pafs(self):
        pass

    def _build_pafs_mask(self):
        pass

    def get_record(self) -> OpenPoseRecordTuple:
        self._get_raw_img()
        self._build_heatmap()
        self._build_pafs()
        self._build_pafs_mask()

        return OpenPoseRecordTuple(
            annotation=self.anno_tuple,
            heatmap=self.np_heatmap,
            heatmap_mask=self.np_heatmap_mask,
            pafs=self.np_pafs,
            pafs_mask=self.np_pafs_mask,
            )
    

@functools.lru_cache(1, typed=True)
def get_openpose_keypoints_data(dataset_dir: str,
                                anno_tuple: OpenPoseKeypintsAnnoTuple,
                                ) -> OpenPoseKeypointsData:
    return OpenPoseKeypointsData(dataset_dir, anno_tuple)


# ディスクキャッシュ
@gzip_cache.memoize(typed=True, tag=disk_cache_tag)
def get_openpose(dataset_dir: str,
                 anno_tuple: OpenPoseKeypintsAnnoTuple,
                 ) -> OpenPoseRecordTuple:
    
    keypoints_data: OpenPoseKeypointsData = \
        get_openpose_keypoints_data(dataset_dir=dataset_dir,
                                    anno_tuple=anno_tuple,
                                    )
    
    return keypoints_data.get_record()



class OpenPoseKeypointsDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 mode: str = 'trn',
                 sortby: str = 'random',
                 ):
        pass

    def __len__(self):
        pass

    def __getitem__(self, ndx):
        pass

    

