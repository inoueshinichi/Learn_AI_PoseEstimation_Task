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
import torchvision

from type_hint import *
from disk import make_disk_cache
from openpose_annotation import OpenPoseAnnoTuple

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

# データレコード
OpenPoseRecordTuple = namedtuple(
    'OpenPoseRecordTuple',
    [
        'annotation',
        'img_t',
        'heatmap_t', 
        'heatmap_mask_t', 
        'pafs_t', 
        'pafs_mask_t'
    ]
)

# データセットの取得
@functools.lru_cache(1)
def get_anno_tuple_list(
    dataset_dir: str = "",
    phase: str = "trn",
    is_require_on_disk: bool = True,
    ) -> Tuple[List[OpenPoseAnnoTuple], str]:

    year: str = dataset_dir.split(os.sep)[-1]
    anno_file: str = "{}{}_anno.json".format(phase, year)

    # JSON形式のアノテーションファイルを読み込む
    anno_json_file = os.sep.join([
        dataset_dir, 
        "openpose_keypoints_anno",
        anno_file,
        ])

    anno_tuple_list: list = []
    if is_require_on_disk:
        if not os.path.exists(anno_json_file):
            raise ValueError(f"No exist {anno_json_file}")
        
        with open(anno_json_file) as f:
            data_this = json.load(f)
        
        data_info = data_this["info"]
        data_license = data_this["license"]
        data_anno_array = data_this["annotations"]
        
        num_samples: int = len(data_anno_array)


        for ndx in range(num_samples):
            anno_tuple = OpenPoseAnnoTuple(
                index=ndx,
                id=data_anno_array[ndx]['id'],
                dataset=data_anno_array[ndx]['dataset'],
                phase=data_anno_array[ndx]['phase'],
                img_id=data_anno_array[ndx]['img_id'],
                rel_img_path=data_anno_array[ndx]['img_path'],
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
        
    return anno_tuple_list, anno_json_file


class OpenPoseKeypointsData:
    def __init__(self,
                 dataset_dir: str,
                 annotation: OpenPoseAnnoTuple,
                 ):
        self.dataset_dir = dataset_dir
        self.annotation = annotation

        self.id: int = self.annotation.id
        self.img_id: int = self.annotation.img_id
        self.img_path: str = os.sep.join([self.dataset_dir, self.annotation.rel_img_path])
        self.pil_img: Image = Image.open(self.img_path)
        self.np_img: np.ndarray = np.array(self.pil_img) # (C,H,W)
        self.np_mask: np.ndarray = np.ones_like(self.np_img) # [[1,1,...,1],[1,1,...,1],[1,1,...,1]]

    def get_anno_id(self) -> int:
        return self.id
    
    def get_img_id(self) -> int:
        return self.img_id
    
    def get_pil_img(self) -> Image:
        return self.pil_img
    
    def get_np_img(self) -> np.ndarray:
        return self.np_img
    
    def get_raw_record(self) -> Tuple[np.ndarray, np.ndarray, OpenPoseAnnoTuple]:
        return self.np_img, self.np_mask, self.annotation
    

@functools.lru_cache(1, typed=True)
def get_openpose_keypoints_data(dataset_dir: str,
                                anno_tuple: OpenPoseAnnoTuple,
                                ) -> OpenPoseKeypointsData:
    return OpenPoseKeypointsData(dataset_dir, anno_tuple)


# ディスクキャッシュ
@gzip_cache.memoize(typed=True, tag=disk_cache_tag)
def get_openpose_raw_record(dataset_dir: str,
                        anno_tuple: OpenPoseAnnoTuple,
                        ) -> Tuple[np.ndarray, np.ndarray, OpenPoseRecordTuple]:
    
    keypoints_data: OpenPoseKeypointsData = \
        get_openpose_keypoints_data(dataset_dir=dataset_dir,
                                    anno_tuple=anno_tuple,
                                    )
    
    return keypoints_data.get_raw_record()



class OpenPoseKeypointsDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 phase: str = 'train',
                 sortby: str = 'random', # 'img_id'
                 img_id: Optional[int] = None,
                 anno_tuple_list: Optional[List[OpenPoseAnnoTuple]] = None,
                 transform: Optional[torch.nn.Sequential] = None,
                 ):
        super(OpenPoseKeypointsDataset, self).__init__()

        if phase != 'train' or phase != 'validation' or phase != 'test':
            raise ValueError(f"Invalid {phase}. Given must be `train`, `validation` or `test`")

        self.dataset_dir: str = dataset_dir
        self.phase: str = 'trn' if phase == 'train' else 'val' if phase == 'validation' else 'tst'
        self.use_cache: Optional[bool] = None
        self.anno_file_path: Optional[str] = None
        self.anno_tuple_list: Optional[List[OpenPoseAnnoTuple]] = None
        self.img_id_list: Optional[List[int]] = None
        self.transform: Optional[torch.nn.Sequential] = None

        if anno_tuple_list is not None:
            self.anno_tuple_list = copy.copy(anno_tuple_list)
            self.use_cache = False
        else:
            # (キャッシュ済み)アノテーションデータを使用する
            self.anno_tuple_list, self.anno_file_path = copy.copy(get_anno_tuple_list(dataset_dir=self.dataset_dir, phase=self.phase))
            self.use_cache = True

        if img_id is not None:
            self.img_id_list = [img_id]
            if not img_id in set(anno_tuple.img_id for anno_tuple in self.anno_tuple_list):
                raise ValueError(f"No exist img_id[{img_id}] in {self.anno_file_path}")
        else:
            self.img_id_list = sorted(set(anno_tuple.img_id for anno_tuple in self.anno_tuple_list))

        # img_idによるアノテーションデータのフィルタ
        img_id_set = set(self.img_id_list)
        self.anno_tuple_list = [x for x in self.anno_tuple_list if x.img_id in img_id_set]

        # アノテーションデータの順序
        if sortby == "random":
            random.shuffle(self.anno_tuple_list)
        elif sortby == "img_id":
            self.anno_tuple_list.sort(key = lambda x: (x.img_id, x.id)) # (画像ID, アノテーションID)
        else:
            raise ValueError("Unkown sort: " + repr(sortby))
        
        # データセット情報
        log.info(
            "{!r}: {} {} anno samples".format(
                self,
                len(self.anno_tuple_list),
                phase,
            )
        )

    def preprocess(self, 
                   annotation: OpenPoseAnnoTuple, 
                   img_t: torch.Tensor, 
                   mask_t: torch.Tensor,
                   ) -> Tuple[OpenPoseAnnoTuple, torch.Tensor, torch.Tensor]:
        # 前処理
        if self.transform is not None:
            pass
        else:
            pass

        return annotation, img_t, mask_t
    
    def augmentation(self, 
                     annotation: OpenPoseAnnoTuple, 
                     img_t: torch.Tensor,
                     mask_t: torch.Tensor,
                     ) -> Tuple[OpenPoseAnnoTuple, torch.Tensor, torch.Tensor]:
        # データ拡張
        if self.transform is not None:
            pass
        else:
            pass

        return annotation, img_t, mask_t
    
    def get_ground_truth(self, 
                         annotation: OpenPoseAnnoTuple,
                         img_t: torch.Tensor,
                         mask_t: torch.Tensor,
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # OpenPose固有のheatmapとpafsを作成する(maskも).
        pass
        
    def __len__(self):
        return len(self.anno_tuple_list)

    def __getitem__(self, ndx) -> OpenPoseRecordTuple:
        # OpenPose Network Modelに入力するデータ

        # 生のデータ
        img, mask, annotation = get_openpose_raw_record(self.dataset_dir, self.anno_tuple_list[ndx])

        img_t: torch.Tensor = torch.from_numpy(img) # (C,H,W)
        mask_t: torch.Tensor = torch.from_numpy(mask) # (C,H,W)

        # 前処理
        annotation, img_t, mask_t = self.preprocess(annotation, img_t, mask_t)

        # データ拡張
        annotation, img_t, mask_t = self.augmentation(annotation, img_t, mask_t)

        # OpenPose用の回帰値(heatmapとpafs)を作成
        heatmap, heatmap_mask, pafs, pafs_mask = self.get_ground_truth(annotation, img_t, mask_t)

        # numpy -> torch.Tensor
        heatmap_t: torch.Tensor = torch.from_numpy(heatmap)
        heatmap_mask_t: torch.Tensor = torch.from_numpy(heatmap_mask)
        pafs_t: torch.Tensor = torch.from_numpy(pafs)
        pafs_mask_t: torch.Tensor = torch.from_numpy(pafs_mask)

        # heatmapのマスク処理とアノテーションデータの修正
        if not torch.all(heatmap_mask_t):
            pass

        # pafsのマスク処理とアノテーションデータの修正
        if not torch.all(pafs_mask_t):
            pass


        # マスクの不必要な要素を除外
        heatmap_mask_t = heatmap_mask_t[0,:,:].squeeze() # (C,H,W) -> (H,W)
        pafs_mask_t = pafs_mask_t[0,:,:].squeeze() # (C,H,W) -> (H,W)

        input_record = OpenPoseRecordTuple(
            annotation=annotation,
            img_t=img_t,
            heatmap_t=heatmap_t,
            heatmap_mask_t=heatmap_mask_t,
            pafs_t=pafs_t,
            pafs_mask_t=pafs_mask_t,
        )


        return input_record


        

