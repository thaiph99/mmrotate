
__author__ = "Thaiph99"
import cv2
import numpy as np
from collections import defaultdict
from mmdet.apis import inference_detector, init_detector
from mot.jde_tracker import JDETracker
from mot.reid.feature_extractor import Extractor


# init detection model
config = "rotated_rtmdet_l_3x_v4/rotated_rtmdet_l-3x-dota_ms_custom_v4.py"
checkpoint = "rotated_rtmdet_l_3x_v4/epoch_96.pth"
device = 'cuda:0'
model = init_detector(config, checkpoint, device)
labels_name = ("paper", "metal", "plastic", "nilon", "glass", "fabric")
tracker = JDETracker(use_byte=True, num_classes=len(labels_name))
extractor = Extractor(model_path='mot/reid/weights/ckpt_epoch_30.t7')

# cv2 config params
font_face = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_thickness = 2
font_color = (255, 255, 255)
text_offset = np.array([0, -10]).reshape((1, 1, -1)).astype(np.int32)
pi = 22/7

# video_path = "/home/thaipham/Videos/2023-07-24-094907.webm"
video_path = "/home/thaipham/Videos/2023-07-24-094422.webm"
# video_path = "/home/thaipham/Videos/output2.avi"
cap = cv2.VideoCapture(video_path)

id_set = set()
frame_id = 0

while (cap.isOpened()):
    # time.sleep(0.1)
    ret, frame = cap.read()
    fake_frame = np.copy(frame)

    frame_id += 1
    if not ret:
        break

    outs = inference_detector(model, frame)

    bboxes = outs.pred_instances.bboxes.cpu().numpy()
    pred_scores = outs.pred_instances.scores.cpu().numpy()
    labels = outs.pred_instances.labels.cpu().numpy()

    confidence_threshold = 0.65
    filtered_indices = np.where(pred_scores > confidence_threshold)[0]

    bboxes = bboxes[filtered_indices]
    pred_scores = pred_scores[filtered_indices]
    labels = labels[filtered_indices]

    if bboxes.shape[0] == 0:
        continue

    over_bboxes = []
    for bbox in bboxes:
        xc, yc, width, height, radian = bbox
        degrees = radian * (180/pi)
        points = cv2.boxPoints(([xc, yc], (width, height), degrees))
        bbox_tlbr = np.array([np.min(points[:, 0]), np.min(points[:, 1]),
                              np.max(points[:, 0]), np.max(points[:, 1])])
        over_bboxes.append(bbox_tlbr)
        bbox_tlbr = bbox_tlbr.astype(int)

    over_bboxes = np.array(over_bboxes)

    im_crops = []
    for bb in over_bboxes:
        t, l, b, r = bb[0:4]
        t = max(0, min(int(t), frame.shape[1]))
        b = max(0, min(int(b), frame.shape[1]))
        l = max(0, min(int(l), frame.shape[0]))
        r = max(0, min(int(r), frame.shape[0]))
        im = frame[l:r, t:b]
        im_crops.append(im)

    pred_embs = extractor(im_crops)

    labels = labels.reshape(-1, 1)
    pred_scores = pred_scores.reshape(-1, 1)

    pred_dets_old = np.concatenate((labels, pred_scores, over_bboxes), axis=1)

    pred_dets_dict = defaultdict(list)
    for cls_id in range(len(labels)):
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
            det = t.det

            # convert tlwh to tlbr
            points = np.asarray(tlwh).copy()
            if len(det) != 0:
                points = np.asarray(det).copy()
            else:
                points = np.asarray(tlwh).copy()

            points[2:4] = points[0:2] + points[2:4]
            intbox = points.astype(int)
            cv2.rectangle(frame, intbox[0:2], intbox[2:4], (0, 0, 255), 2)

            # visualize object id
            obj_id = tid
            id_set.add(obj_id)
            text = f"#{obj_id}"
            frame = cv2.putText(frame, text, (intbox[0], intbox[1] - 10), font_face,
                                font_scale, (0, 200, 255), font_thickness, cv2.LINE_4,)

    cv2.imshow('Frame', frame)

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('total id: ', len(id_set))
