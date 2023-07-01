"""OpenPose用MS_COCOのデータセット
"""
import os
import sys

from diskcache import FanoutCache

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

import numpy as np
import PIL

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

from type_hint import *
from disk import make_disk_cache


scope: str = "MS_COCO"
version: str = '2014'
disk_cache_tag: str = f"{scope}_{version}"

# 生データセットのディスクキャッシュ
gzip_cache: FanoutCache = make_disk_cache(
    cache_dir="F:/Cache/OpenPose",                                         
    scope=scope,
    version=version,
    )

# データレコード
OpenPoseRecordInfoTuple = namedtuple(
    'OpenPoseRecordInfoTuple',
    ['index', 'img_path', 'mask_path', 'coco_annotation']
)

# 訓練データセットの取得
@functools.lru_cache(1)
def get_trn_record_info_list(
    dataset_dir: str = "",
    is_require_on_disk: bool = True,
    ):

    """データセットへのアクセス"""
    record_info_list: list = []
    if is_require_on_disk:

    # JSON形式のアノテーションファイルを読み込む
        json_file = os.sep.join([dataset_dir, "COCO.json"])
        with open(json_file) as f:
            data_this = json.load(f)
            data_json = data_this['root']
        
        num_samples: int = len(data_json)
        trn_indexes: list = []
        val_indexes: list = []
        for ndx in range(num_samples):
            if data_json[ndx]['isValidation'] != 0.:
                val_indexes.append(ndx)
            else:
                trn_indexes.append(ndx)

        assert len(trn_indexes) !=0, f"No training indexes about {json_file}"

        del val_indexes

        # 生データ(画像)のパスを取得
        trn_img_list: list = []
        for ndx in trn_indexes:
            img_path: str = os.path.join(dataset_dir, data_json[ndx]['img_paths'])
            trn_img_list.append(img_path)
        
        
        # マスクデータのパスを取得
        trn_mask_list: list = []
        for ndx in trn_indexes:
            img_ndx = data_json[ndx]['img_paths'][-16:-4]
            anno_path = os.sep.join([dataset_dir,
                                    'mask',
                                    f'train{version}',
                                    f'mask_COCO_train{version}_{img_ndx}.jpg'])
            trn_mask_list.append(anno_path)

        # アノテーションデータの取得
        trn_anno_list: list = []
        for ndx in trn_indexes:
            trn_anno_list.append(data_json[ndx])

        # 名前付きタプルのリストを作成
       
        records = zip(trn_indexes, 
                    trn_img_list, 
                    trn_mask_list, 
                    trn_anno_list)
        for ndx, img_path, mask_path, coco_anno in records:
            record_info_list.append(
                OpenPoseRecordInfoTuple(
                    ndx,
                    img_path,
                    mask_path,
                    coco_anno,
                )
            )

    return record_info_list



# 検証データセットの取得
@functools.lru_cache(1)
def get_val_record_info_list(
    dataset_dir: str = "",
    is_require_on_disk: bool = True,
    ):
    """データセットへのアクセス"""
    record_info_list: list = []
    if is_require_on_disk:

        # JSON形式のアノテーションファイルを読み込む
        json_file = os.sep.join([dataset_dir, "COCO.json"])
        with open(json_file) as f:
            data_this = json.load(f)
            data_json = data_this['root']
        
        num_samples: int = len(data_json)
        trn_indexes: list = []
        val_indexes: list = []
        for ndx in range(num_samples):
            if data_json[ndx]['isValidation'] != 0.:
                val_indexes.append(ndx)
            else:
                trn_indexes.append(ndx)

        assert len(val_indexes) !=0, f"No validation indexes about {json_file}"

        del trn_indexes

        # 生データ(画像)のパスを取得
        val_img_list: list = []
        for ndx in val_indexes:
            img_path: str = os.path.join(dataset_dir, data_json[ndx]['img_paths'])
            val_img_list.append(img_path)

        # マスクデータの取得
        val_mask_list: list = []
        for ndx in val_indexes:
            img_ndx = data_json[ndx]['img_paths'][-16:-4]
            anno_path = os.sep.join([dataset_dir,
                                    'mask',
                                    f'val{version}',
                                    f'mask_COCO_val{version}_{img_ndx}.jpg'])
            val_mask_list.append(anno_path)

        # アノテーションデータの取得
        val_anno_list: list = []
        for ndx in val_indexes:
            val_anno_list.append(data_json[ndx])

        # 名前付きタプルのリストを作成
        
        records = zip(val_indexes, 
                    val_img_list, 
                    val_mask_list, 
                    val_anno_list)
        for ndx, img_path, mask_path, coco_anno in records:
            record_info_list.append(
                OpenPoseRecordInfoTuple(
                    ndx,
                    img_path,
                    mask_path,
                    coco_anno,
                )
            )

    return record_info_list


class MS_COCO_Image:
    def __init__(self,
                 img_path: str,
                 ):
        self.pil_img: PIL.Image = PIL.Image.open(img_path)
        self.np_img: np.ndarray = np.array(self.pil_img, dtype=np.float32) # (C,H,W)

    def get_np_img(self) -> np.ndarray:
        return self.np_img
    
    def get_pil_img(self) -> PIL.Image:
        return self.pil_img
    

@functools.lru_cache(1, typed=True)
def get_ms_coco_img(img_path: str) -> MS_COCO_Image:
    return MS_COCO_Image(img_path)


@gzip_cache.memoize(typed=True, tag=disk_cache_tag)
def get_ms_coco_img_record(img_path: str) -> np.ndarray:
        ms_coco_img: MS_COCO_Image = get_ms_coco_img(img_path)
        np_img: np.ndarray = ms_coco_img.get_np_img()
        return np_img


class MSCOCOKeyPointsDataset(Dataset):
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

    

