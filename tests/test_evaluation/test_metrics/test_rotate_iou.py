# Copyright (c) OpenMMLab. All rights reserved.
import math
import os.path as osp
import tempfile
import unittest

import numpy as np
import torch
from mmcv.ops import box_iou_rotated

from mmrotate.evaluation import DOTAMetric


class TestDOTAMetric(unittest.TestCase):

    def _create_dummy_data_sample(self):
        bboxes = np.array([[23, 31, 10.0, 20.0, 0.0],
                           [100, 120, 10.0, 20.0, 0.1],
                           [150, 160, 10.0, 20.0, 0.2]])
        labels = np.array([0] * 3)
        bboxes_ignore = np.array([[0] * 5])
        labels_ignore = np.array([0])
        pred_bboxes = np.array([[23, 31, 10.0, 20.0, 0.0],
                                # [100, 120, 10.0, 20.0, 0.1],
                                # [150, 160, 10.0, 20.0, 0.2],
                                [250, 260, 10.0, 20.0, 0.3]])
        pred_scores = np.array([1.0, 0.95])
        pred_labels = np.array([0, 0])
        return [
            dict(
                img_id='P2805__1024__0___0',
                gt_instances=dict(
                    bboxes=torch.from_numpy(bboxes),
                    labels=torch.from_numpy(labels)),
                ignored_instances=dict(
                    bboxes=torch.from_numpy(bboxes_ignore),
                    labels=torch.from_numpy(labels_ignore)),
                pred_instances=dict(
                    bboxes=torch.from_numpy(pred_bboxes),
                    scores=torch.from_numpy(pred_scores),
                    labels=torch.from_numpy(pred_labels)))
        ]

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_init(self):
        # test invalid iou_thrs
        with self.assertRaises(AssertionError):
            DOTAMetric(iou_thrs={'a', 0.5})

        metric = DOTAMetric(iou_thrs=0.6)
        self.assertEqual(metric.iou_thrs, [0.6])

    def test_eval(self):
        metric = DOTAMetric(metric='mAP', iou_thrs=[0.5, 0.55, 0.6])
        metric.dataset_meta = {'classes': ('plane', )}
        metric.process({}, self._create_dummy_data_sample())
        results = metric.evaluate(size=1)
        targets = {'dota/AP50': 1.0, 'dota/mAP': 1.0}
        print('checking results: ', results)
        # self.assertDictEqual(results, targets)

        # test multi-threshold
        # metric = DOTAMetric(iou_thrs=[0.1, 0.5])
        # metric.dataset_meta = dict(classes=('plane', ))
        # metric.process({}, self._create_dummy_data_sample())
        # results = metric.evaluate(size=1)
        # targets = {'dota/AP10': 1.0, 'dota/AP50': 1.0, 'dota/mAP': 1.0}
        # self.assertDictEqual(results, targets)


def calculate_rotated_box_intersection(box1, box2):
    """
    Calculates the intersection of two rotated boxes in PyTorch tensors.

    Args:
      box1: A PyTorch tensor of shape (5,), where the elements represent the x-center, y-center, width, height, and angle of rotation of the box.
      box2: A PyTorch tensor of shape (5,), where the elements represent the x-center, y-center, width, height, and angle of rotation of the box.

    Returns:
      A PyTorch tensor of shape (5,), where the elements represent the x-center, y-center, width, height, and angle of rotation of the intersection box, or None if there is no intersection.
    """

    # Convert the bounding boxes to a format that is compatible with the algorithm.
    box1 = box1.view(1, 5)
    box2 = box2.view(1, 5)

    # Initialize the intersection box.
    intersection_box = torch.zeros_like(box1)

    # Calculate the intersection of each edge of the second box with the first box.
    for i in range(4):
        edge1 = torch.tensor([[-box1[:, 2] / 2, -box1[:, 3] / 2], [box1[:, 2] / 2, -box1[:, 3] / 2],
                             [box1[:, 2] / 2, box1[:, 3] / 2], [-box1[:, 2] / 2, box1[:, 3] / 2]])
        edge2 = torch.tensor([box2[:, 2] / 2 * math.cos(box2[:, 4]), box2[:, 2] / 2 * math.sin(box2[:, 4]),
                             box2[:, 3] / 2 * math.cos(box2[:, 4]), box2[:, 3] / 2 * math.sin(box2[:, 4])])
        print(edge2.shape)

        # Calculate the intersection point of the two edges.
        intersection_point = calculate_line_intersection(edge1, edge2)

        # If the intersection point is on the edge of the first box, add it to the intersection box.
        if intersection_point is not None and is_point_on_line(intersection_point, edge1):
            intersection_box = torch.cat([intersection_box, intersection_point.unsqueeze(0)], dim=0)

    # If there is no intersection, return None.
    if intersection_box.shape[0] == 1:
        return None

    # Calculate the center of the intersection box.
    intersection_center = intersection_box.mean(dim=0)

    # Calculate the width and height of the intersection box.
    intersection_width = torch.max(intersection_box[:, 0]) - torch.min(intersection_box[:, 0])
    intersection_height = torch.max(intersection_box[:, 1]) - torch.min(intersection_box[:, 1])

    # Calculate the angle of rotation of the intersection box.
    intersection_angle = math.atan2(
        intersection_box[:, 1] - intersection_center[1], intersection_box[:, 0] - intersection_center[0])

    # Return the intersection box.
    return torch.cat([intersection_center, intersection_width.unsqueeze(0), intersection_height.unsqueeze(0), intersection_angle.unsqueeze(0)], dim=0)


def calculate_line_intersection(line1, line2):
    """
    Calculates the intersection point of two lines in PyTorch tensors.

    Args:
      line1: A PyTorch tensor of shape (2, 2), where each row represents a point on the line.
      line2: A PyTorch tensor of shape (2, 2), where each row represents a point on the line.

    Returns:
      A PyTorch tensor of shape (2,), where the elements represent the x-coordinate and y-coordinate of the intersection point, or None if there is no intersection.
    """

    # Calculate the determinant of the two lines.
    determinant = line1[:, 0] * line2[:, 1] - line1[:, 1] * line2[:, 0]

    # If the determinant is zero, the lines are parallel and there is no intersection.
    if determinant == 0:
        return None


def calculate_iou(boxes1, boxes2):
    """
    Calculates the IoU of oriented bounding boxes in PyTorch tensors.

    Args:
      boxes1: A PyTorch tensor of shape (N, 5), where each row represents a bounding box.
      boxes2: A PyTorch tensor of shape (N, 5), where each row represents a bounding box.

    Returns:
      A PyTorch tensor of shape (1,), where the element represents the mean IoU of the bounding boxes in boxes1 and boxes2.
    """

    # Convert the bounding boxes to a format that is compatible with PyTorch.
    boxes1 = boxes1.view(-1, 5)
    boxes2 = boxes2.view(-1, 5)

    # Calculate the area of each bounding box.
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]

    # Calculate the intersection area of the two bounding boxes.
    intersection_area = torch.min(boxes1[:, :2] + boxes1[:, 2:4] / 2, boxes2[:, :2] + boxes2[:, 2:4] / 2) - \
        torch.max(boxes1[:, :2] - boxes1[:, 2:4] / 2, boxes2[:, :2] - boxes2[:, 2:4] / 2)
    intersection_area = torch.clamp(intersection_area, min=0) * torch.min(boxes1[:, 2:4], boxes2[:, 2:4])

    # Calculate the union area of the two bounding boxes.
    union_area = area1 + area2 - intersection_area

    # Calculate the IoU of the two bounding boxes.
    iou = intersection_area / union_area

    # Calculate the mean IoU.
    mean_iou = iou.mean()

    return mean_iou

def box_iou_rotated_custom(bboxes1: torch.Tensor,
                    bboxes2: torch.Tensor,
                    mode: str = 'iou',
                    aligned: bool = False,
                    clockwise: bool = True) -> torch.Tensor:
 
    assert mode in ['iou', 'iof']
    mode_dict = {'iou': 0, 'iof': 1}
    mode_flag = mode_dict[mode]
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if aligned:
        ious = bboxes1.new_zeros(rows)
    else:
        if bboxes1.device.type == 'mlu':
            ious = bboxes1.new_zeros([rows, cols])
        else:
            ious = bboxes1.new_zeros(rows * cols)
    if not clockwise:
        flip_mat = bboxes1.new_ones(bboxes1.shape[-1])
        flip_mat[-1] = -1
        bboxes1 = bboxes1 * flip_mat
        bboxes2 = bboxes2 * flip_mat
    bboxes1 = bboxes1.contiguous()
    bboxes2 = bboxes2.contiguous()
    # ext_module.box_iou_rotated(
    #     bboxes1, bboxes2, ious, mode_flag=mode_flag, aligned=aligned)
    if not aligned:
        ious = ious.view(rows, cols)
    return ious


if __name__ == '__main__':
    # testdota = TestDOTAMetric()
    # testdota.test_eval()
    pi = 22/7
    # Format: (center_x, center_y, width, height, angle)
    box1 = torch.tensor([[100.0, 100.1, 50.2, 30.3, 45.4 * (pi/180)]])
    # box2 = torch.tensor([[100.0, 100.1, 50.2, 30.3, 45.4 * (pi/180)]])
    box2 = torch.tensor([[100.5, 110.6, 40.6, 40.5, 30.4 * (pi/180)]])
    iou_bard = calculate_iou(box1, box2)
    # iou_gpt = calculate_rotated_box_intersection(box1[0], box2[0])

    ious = box_iou_rotated(box1, box2)

    print(iou_bard)
    # print(iou_gpt)
    print(ious)
