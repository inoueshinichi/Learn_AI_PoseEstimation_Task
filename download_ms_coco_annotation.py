"""MS_COCOのデータセットをダウンロードする
    | Year | Dataset | Annotation |
    | 2014 | trn,val,test | class_trn-anno, class_val-anno, test-img-info |
    | 2015 | test | test-img-info |
    | 2017 | trn,val,test,unlabeled | class_trn_anno, class_val-anno, stuff_trn-anno, stuff_val-anno, panoptic_trn-anno, panoptic_val-anno, test-img_info, unlabeled-img_info |
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

import urllib.request
import zipfile
import tarfile

ms_coco_zip_base_url: str = "http://images.cocodataset.org/zips/"

def download_method(dataset_dir: str, file_basename: str) -> str:

    download_path: str = ""
    
    filename: str = file_basename + '.zip'
    target_path = os.path.join(dataset_dir, filename)

    if not os.path.exists(target_path):
        # download
        url: str = ms_coco_zip_base_url + filename
        urllib.request.urlretrieve(url, target_path)
    
        # zip
        zip = zipfile.ZipFile(target_path)
        zip.extractall(target_path)  # ZIPを解凍
        zip.close()  # ZIPファイルをクローズ
        download_path = target_path
        
    return download_path


def download_ms_coco_anno_2014(dataset_dir: str):
    pass

def download_ms_coco_anno_2015(dataset_dir: str):
    pass

def download_ms_coco_anno_2017(dataset_dir: str): 
    type_list: list[str] = ['annotations_trainval2017']
    for data_type in type_list:
        download_path: str = download_method(dataset_dir, data_type)
        if download_path:
            print(f'Download {download_path}')


if __name__ == "__main__":
    dataset_dir = "F:/MS_COCO/2017"
    download_ms_coco_anno_2017(dataset_dir)