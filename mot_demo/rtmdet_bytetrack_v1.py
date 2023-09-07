
__author__ = "Thaiph99"

import time
from collections import defaultdict
from mmdet.apis import inference_detector, init_detector
from mmengine.runner import load_checkpoint
from mmdet.registry import VISUALIZERS
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import mmdet
import mmcv
import mmrotate
import mot.mot_utils as utils
from mot.jde_tracker import JDETracker
from mot.reid.feature_extractor import Extractor

import cv2
import numpy as np

print(mmrotate.__version__)


# init detection model
# config = "configs/rotated_rtmdet/rotated_rtmdet_m-3x-dota_ms_custom_v1.py"
config = "rotated_rtmdet_l_3x_v4/rotated_rtmdet_l-3x-dota_ms_custom_v4.py"

checkpoint = "rotated_rtmdet_l_3x_v4/epoch_96.pth"

device = 'cuda:0'
model = init_detector(config, checkpoint, device)
labels_name = ("paper", "metal", "plastic", "nilon", "glass", "fabric")
tracker = JDETracker(use_byte=True,
                     num_classes=len(labels_name))

extractor = Extractor(model_path='mot/reid/weights/ckpt_epoch_30.t7')
# cv2 config params
font_face = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_thickness = 2
font_color = (255, 255, 255)
text_offset = np.array([0, -10]).reshape((1, 1, -1)).astype(np.int32)
pi = 22/7

# video writer

# cap = cv2.VideoCapture('/home/thaipham/Videos/2023-07-24-094422.webm')
cap = cv2.VideoCapture('/home/thaipham/Videos/output2.avi')
# cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
res_video = cv2.VideoWriter('demo/output.mp4',
                            fourcc,
                            fps,
                            (1428, 1024))
prev_frame_time, new_frame_time = 0, 0

if (cap.isOpened() == False):
    print("Error opening video file")

results = defaultdict(list)
frame_id = 0

# run video/camera
while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    outs = inference_detector(model, frame)

    bboxes = outs.pred_instances.bboxes.cpu().numpy()
    pred_scores = outs.pred_instances.scores.cpu().numpy()
    labels = outs.pred_instances.labels.cpu().numpy()

    confidence_threshold = 0.55
    filtered_indices = np.where(pred_scores > confidence_threshold)[0]

    bboxes = bboxes[filtered_indices]
    pred_scores = pred_scores[filtered_indices]
    labels = labels[filtered_indices]

    if bboxes.shape[0] == 0:
        frame = cv2.resize(frame, (1428, 1024))
        # res_video.write(frame)
        frame_id += 1
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        continue

    # draw prediction rotated bounding box
    for bbox in bboxes:
        xc, yc, width, height, radian = bbox
        degrees = radian * (180/pi)
        points = cv2.boxPoints(([xc, yc], (width, height), degrees))
        pts = np.array(points).reshape((1, -1, 1, 2)).astype(np.int32)
        cv2.polylines(frame, pts, True, (0, 255, 0), 2)

    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    fps_str = "fps: {}".format(int(fps))
    prev_frame_time = new_frame_time

    cv2.putText(frame, fps_str,
                (7, int(9/10 * frame_height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (100, 0, 255),
                2, cv2.FILLED)

    # convert xywh to tlbr
    pred_bboxes = np.asarray(bboxes[:, 0:4]).copy()
    pred_bboxes = np.concatenate(
        (bboxes[:, 0:2] - bboxes[:, 2:4]/2, bboxes[:, 0:2] + bboxes[:, 2:4]/2),
        axis=1
    )

    # Use embedding reid
    im_crops = []
    for pred_bbox in pred_bboxes:
        x1, y1, x2, y2 = map(int, pred_bbox)
        x1 = max(min(x1, frame.shape[1]), 0)
        x2 = max(min(x2, frame.shape[1]), 0)
        y1 = max(min(y1, frame.shape[0]), 0)
        y2 = max(min(y2, frame.shape[0]), 0)

        im = frame[y1:y2, x1:x2]
        print('checking pred_bbox: ', pred_bbox)
        print("checking im shape: ", im.shape)
        im_crops.append(im)

    pred_embs = extractor(im_crops)

    # reshape numpy array from (n,0) to (n,1)
    labels = labels.reshape(-1, 1)
    pred_scores = pred_scores.reshape(-1, 1)
    pred_dets_old = np.concatenate(
        (labels, pred_scores, pred_bboxes), axis=1
    )

    pred_dets_dict = defaultdict(list)
    for cls_id in range(len(labels_name)):
        cls_idx = (pred_dets_old[:, 0:1] == cls_id).squeeze(-1)
        pred_dets_dict[cls_id] = pred_dets_old[cls_idx]

    online_targets_dict = tracker.update(pred_dets_old, pred_embs)
    online_tlwhs = defaultdict(list)
    online_scores = defaultdict(list)
    online_ids = defaultdict(list)
    for cls_id in range(tracker.num_classes):
        online_targets = online_targets_dict[cls_id]
        for idx_det, t in enumerate(online_targets):
            tlwh = t.tlwh
            tid = t.track_id
            tscore = t.score

            # visualize detector and tracker

            # convert tlwh to tlbr
            points = np.asarray(tlwh).copy()
            points[2:4] = points[0:2] + points[2:4]
            intbox = points.astype(int)
            cv2.rectangle(frame, intbox[0:2], intbox[2:4], (0, 0, 255), 2)

            # visualize object id
            obj_id = tid
            text = f"#{obj_id}"
            frame = cv2.putText(frame, text, (intbox[0], intbox[1] - 10), font_face,
                                font_scale, (0, 200, 255), font_thickness, cv2.LINE_4,)

            # save online
            if tlwh[2] * tlwh[3] <= tracker.min_box_area:
                continue
            if tracker.vertical_ratio > 0 and tlwh[2] / tlwh[
                    3] > tracker.vertical_ratio:
                continue
            online_tlwhs[cls_id].append(tlwh)
            online_ids[cls_id].append(tid)
            online_scores[cls_id].append(tscore)

        # save results
        results[cls_id].append((frame_id + 1,
                                online_tlwhs[cls_id],
                                online_scores[cls_id],
                                online_ids[cls_id]))

    # res_video.write(frame)
    frame = cv2.resize(frame, (1428, 1024))

    res_video.write(frame)
    frame_id += 1
    cv2.imshow('Frame', frame)

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
# res_video.release()
cv2.destroyAllWindows()
