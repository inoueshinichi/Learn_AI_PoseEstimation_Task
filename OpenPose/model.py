"""OpenPoseモデル
"""
import os
from re import I, S
import sys

from numpy import require
from sympy import DiagMatrix

# os.sepはプラットフォーム固有の区切り文字(Windows: `\`, Unix: `/`)
module_parent_dir = os.sep.join([os.path.dirname(__file__), '..'])
# print("module_parent_dir", module_parent_dir)
sys.path.append(module_parent_dir)

from log_conf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

import pickle
from collections import namedtuple

import torch
import torch.nn as nn
import torch.functional as F
import torch.utils.data as data
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.nn import init

# 画像特徴量抽出器
class FeatureBackborn(nn.Module):
    """OpenPose Original VGG19

    VGG19の上流側から10個のパラメータブロックに
    CMPオリジナルブロックを連結したモデル
    (下流タスク:Stage用)
    """
    def __init__(self):
        super(FeatureBackborn, self).__init__()

        blocks = [
            # conv2d : [in_channels, out_channels, kernel, stride, padding, (dilattion=1, groups=1)]
            { 'conv2d_1_1': [3, 64, 3, 1, 1] },
            { 'conv2d_1_2': [64, 64, 3, 1, 1] },
            # maxpool2d : [kernel_size, stride, padding]
            { 'maxpool2d_1': [2, 2, 0] },
            { 'conv2d_2_1': [64, 128, 3, 1, 1] },
            { 'conv2d_2_2': [64, 128, 3, 1, 1] },
            { 'maxpool2d_2': [2, 2, 0] },
            { 'conv2d_3_1': [128, 256, 3, 1, 1] },
            { 'conv2d_3_2': [256, 256, 3, 1, 1] },
            { 'conv2d_3_3': [256, 256, 3, 1, 1] },
            { 'conv2d_3_4': [256, 256, 3, 1, 1] },
            { 'maxpool2d_3': [2, 2, 0] },
            { 'conv2d_4_1': [256, 512, 3, 1, 1] },
            { 'conv2d_4_2': [512, 512, 3, 1, 1] }, # PReLU
            # OpenPose original (CPM)
            { 'conv2d_4_3_CPM': [512, 256, 3, 1, 1] }, # PReLU
            { 'conv2d_4_4_CPM': [256, 128, 3, 1, 1] }, # PReLU
        ]

        layers = []
        for i in range(len(blocks)):
            block = blocks[i]
            for k, v in block.items():
                if 'maxpool' in k:
                    layers += [
                        nn.MaxPool2d(kernel_size=v[0],
                                    stride=v[1],
                                    padding=v[2],
                                    )]
                
                elif k in ['conv2d_4_2', 'conv2d_4_3_CPM', 'conv2d_4_4_CPM']:
                    conv2d = nn.Conv2d(
                                    in_channels=v[0],
                                    out_channels=v[1],
                                    kernel_size=v[2],
                                    stride=v[3],
                                    padding=v[4],
                                    )
                    layers += [
                        conv2d, 
                        nn.PReLU(num_parameters=v[1])
                        ]
                else:
                    conv2d = nn.Conv2d(
                                    in_channels=v[0],
                                    out_channels=v[1],
                                    kernel_size=v[2],
                                    stride=v[3],
                                    padding=v[4],
                                    )
                    layers += [
                        conv2d,
                        nn.ReLU(inplace=True)
                    ]
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.layers(x)
        return out
    

# Heatmapストリームブロック
class StreamBlockHeatmap(nn.Module):
    """Heatmap for keypoints
    """
    def __init__(self, 
                 stage,
                 in_channels,
                 inner_channels,
                 out_channels=38,
                 ):
        super(StreamBlockHeatmap, self).__init__()

        stream = None
        if stage > 1:
            stream = [
                # conv2d : [in_channels, out_channels, kernel, stride, padding, (dilattion=1, groups=1)]
                { f'st{stage}_heat_conv2d_1': [in_channels, inner_channels, 7, 1, 3] },
                { f'st{stage}_heat_conv2d_2': [inner_channels, inner_channels, 7, 1, 3] },
                { f'st{stage}_heat_conv2d_3': [inner_channels, inner_channels, 7, 1, 3] },
                { f'st{stage}_heat_conv2d_4': [inner_channels, inner_channels, 7, 1, 3] },
                { f'st{stage}_heat_conv2d_5': [inner_channels, inner_channels, 7, 1, 3] },
                { f'st{stage}_heat_conv2d_6': [inner_channels, inner_channels, 1, 1, 0] },
                { f'st{stage}_heat_conv2d_7': [inner_channels, out_channels, 1, 1, 0] },
            ]
        else:
            stream = [
                # conv2d : [in_channels, out_channels, kernel, stride, padding, (dilattion=1, groups=1)]
                { f'st{stage}_heat_conv2d_1': [in_channels, inner_channels, 3, 1, 1] },
                { f'st{stage}_heat_conv2d_2': [inner_channels, inner_channels, 3, 1, 1] },
                { f'st{stage}_heat_conv2d_3': [inner_channels, inner_channels, 3, 1, 0] },
                { f'st{stage}_heat_conv2d_4': [inner_channels, inner_channels * 3, 1, 1, 0] },
                { f'st{stage}_heat_conv2d_5': [inner_channels * 3, out_channels, 1, 1, 0] },
            ]

        layers = []
        for i in range(len(stream)):
            for k, v in stream[i].items():
                if 'conv2d' in k:
                    # 畳み込み関数
                    conv = nn.Conv2d(
                        in_channels=v[0],
                        out_channels=v[1],
                        kernel_size=v[2],
                        stride=v[3],
                        padding=v[4],
                    )

                    # 活性化関数
                    act = nn.PReLU(
                        num_parameters=out_channels,
                        )
                    layers += [ conv, act ]

        # 最後の活性化関数を除外
        self.layers = nn.Sequential(*layers[:-1])

    def forward(self, x):
        stream_out = self.layers(x)
        return stream_out
        

# PAFsストリームブロック
class StreamBlockPAFs(nn.Module):
    """Part Affinity Fields(PAFs) for limbs
    """
    def __init__(self, 
                 stage,
                 in_channels,
                 inner_channels,
                 out_channels = 19,
                 ):
        super(StreamBlockPAFs, self).__init__()

        stream = None
        if stage > 1:
            stream = [
                # conv2d : [in_channels, out_channels, kernel, stride, padding, (dilattion=1, groups=1)]
                { f'st{stage}_heat_conv2d_1': [in_channels, inner_channels, 7, 1, 3] },
                { f'st{stage}_heat_conv2d_2': [inner_channels, inner_channels, 7, 1, 3] },
                { f'st{stage}_heat_conv2d_3': [inner_channels, inner_channels, 7, 1, 3] },
                { f'st{stage}_heat_conv2d_4': [inner_channels, inner_channels, 7, 1, 3] },
                { f'st{stage}_heat_conv2d_5': [inner_channels, inner_channels, 7, 1, 3] },
                { f'st{stage}_heat_conv2d_6': [inner_channels, inner_channels, 1, 1, 0] },
                { f'st{stage}_heat_conv2d_7': [inner_channels, out_channels, 1, 1, 0] },
            ]
        else:
            stream = [
                # conv2d : [in_channels, out_channels, kernel, stride, padding, (dilate=1, groups=1)]
                { f'st{stage}_heat_conv2d_1': [in_channels, inner_channels, 3, 1, 1] },
                { f'st{stage}_heat_conv2d_2': [inner_channels, inner_channels, 3, 1, 1] },
                { f'st{stage}_heat_conv2d_3': [inner_channels, inner_channels, 3, 1, 0] },
                { f'st{stage}_heat_conv2d_4': [inner_channels, inner_channels * 3, 1, 1, 0] },
                { f'st{stage}_heat_conv2d_5': [inner_channels * 3, out_channels, 1, 1, 0] },
            ]

        layers = []
        for i in range(len(stream)):
            for k, v in stream[i].items():
                if 'conv2d' in k:
                    # 畳み込み関数
                    conv = nn.Conv2d(
                        in_channels=v[0],
                        out_channels=v[1],
                        kernel_size=v[2],
                        stride=v[3],
                        padding=v[4],
                    )

                    # 活性化関数
                    act = nn.PReLU(
                        num_parameters=out_channels,
                        )
                    layers += [ conv, act ]

        # 最後の活性化関数を除外
        self.layers = nn.Sequential(*layers[:-1])

    def forward(self, x):
        stream_out = self.layers(x)
        return stream_out
    
HeatApfsTuple = namedtuple(
    "HeatApfsTuple",
    [
        'heatmap_out',
        'apfs_out',
    ]
)

# カスケード接続を行うステージブロック
# ステージを重ねるごとに回帰精度が洗練される
class CascadeStage(nn.Module):
    """カスケード接続用ステージ
    """
    def __init__(self,
                 stage,
                 in_channels,
                 inner_channels,
                 heatmap_channels,
                 pafs_channels,
                 ):
        super(CascadeStage, self).__init__()

        # Heatmap stream block
        self.heatmap_stream_block = StreamBlockHeatmap(
            stage=stage,
            in_channels=in_channels,
            inner_channels=inner_channels,
            out_channels=heatmap_channels,
        )

        # Pafs stream block
        self.pafs_stream_block = StreamBlockPAFs(
            stage=stage,
            in_channels=in_channels,
            inner_channels=inner_channels,
            out_channels=pafs_channels,
        )

    def forward(self, x):
        heatmap_out = self.heatmap_stream_block(x) # (49,48,48)
        pafs_out = self.pafs_stream_block(x) # (38,48,48)

        # 名前付きタプル
        return HeatApfsTuple(heatmap_out, pafs_out)


class OpenPoseNet(nn.Module):
    def __init__(self,
                 img_shape,
                 num_cascade_stages = 6,
                 inner_channels = 128,
                 human_keypoints = 38,
                 human_limbs = 19,
                 ):
        super(OpenPoseNet, self).__init__()

        # FeatureBackborn
        self.feature_backborn = FeatureBackborn()

        # Stage数: ステージ数を増やすに従って, 関節点と四肢の回帰推定精度が向上する
        self.num_cascade_stages = num_cascade_stages

        # 人体のキーポイント(関節)数
        self.keypoints = human_keypoints

        # 人体の肢(関節と関節の連結部)数
        self.limbs = human_limbs

        # 入力画像の形状(C,H,W)
        channels, height, width = img_shape

        # カスケード接続のステージ
        self.stages = []
        # Stage1
        stage = CascadeStage(
            stage=1,
            in_channels=channels,
            inner_channels=inner_channels,
            heatmap_channels=self.keypoints,
            pafs_channels=self.limbs,
        )
        self.stages.append(stage)
        # Stage > 2 [2,6]
        for ndx in range(2, num_cascade_stages + 1):
            stage = CascadeStage(
                stage=ndx,
                in_channels= inner_channels + self.keypoints + self.limbs,
                inner_channels=inner_channels,
                heatmap_channels=self.keypoints,
                pafs_channels=self.limbs,
            )
            self.stages.append(stage)

        # 損失計算用の各ステージ出力
        self.saved_stage_out_for_loss = []

    def forward(self, x):
        # 1) 画像特徴量抽出
        feature_map = self.feature_backborn(x)

        # 2) ステージング(heatmap + pafs)
        cat_out = feature_map
        final_heatmap_out = None
        final_pafs_out = None
        for ndx in range(1, len(self.stages) + 1):
            # ステージ出力
            heatmap_out, pafs_out = self.stages[ndx](cat_out)

            # 損失計算用バッファに保存
            self.saved_stage_out_for_loss.append(
                HeatApfsTuple(heatmap_out, pafs_out)
                )

            # (N,C,H,W) C:第1次元で連結 (feature_mapのスキップ接続)
            cat_out = torch.cat([heatmap_out, pafs_out, feature_map], dim=1)

            if ndx == len(self.stages):
                final_heatmap_out = heatmap_out
                final_pafs_out = pafs_out
        
        return final_heatmap_out, final_pafs_out, self.saved_stage_out_for_loss
    
    






    

    

