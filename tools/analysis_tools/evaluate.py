__author__ = 'Thaiph99'
import argparse
import matplotlib.pyplot as plt
import os
from mmcv.ops import nms_rotated
from mmdet.registry import DATASETS
from mmengine import Config, DictAction
from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from typing import List, Optional, Union

from mmrotate.evaluation import DOTAMetric
from mmrotate.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from detection results')
    # parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test .pkl result')
    parser.add_argument(
        'save_dir', help='directory where confusion matrix will be saved')
    parser.add_argument(
        '--show', action='store_true', help='show confusion matrix')
    parser.add_argument(
        '--color-theme',
        default='plasma',
        help='theme of the matrix color map')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='score threshold to filter detection bboxes')
    parser.add_argument(
        '--tp-iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold to be considered as matched')
    parser.add_argument(
        '--nms-iou-thr',
        type=float,
        default=None,
        help='nms IoU threshold, only applied when users want to change the'
        'nms IoU threshold.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def create_dummy_results(results: List[dict]) -> List[dict]:
    dummy_results = []

    for per_img_res in results:
        dummy_results.append(dict(
            img_id=per_img_res['img_id'],
            gt_instances=per_img_res['gt_instances'],
            ignored_instances=per_img_res['ignored_instances'],
            pred_instances=per_img_res['pred_instances']))

    return dummy_results


def main():
    register_all_modules()
    args = parse_args()

    pkl_res = load(args.prediction_path)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print('checking pkl_res len: ', len(pkl_res))
    print('checking pkl_res first type: ', type(pkl_res[0]))

    metric = DOTAMetric(iou_thrs=[0.1, 0.5])
    metric.dataset_meta = {
        'classes': ('paper', 'metal', 'plastic',
                    'nilon', 'glass', 'fabric', 'other')}
    metric.process({}, create_dummy_results(pkl_res))
    eva_res = metric.evaluate(size=1)
    print('evaulation results: ', eva_res)


if __name__ == '__main__':
    main()
