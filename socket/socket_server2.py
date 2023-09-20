import os
import socket
import cv2
import numpy as np
import numpy
import base64
import glob
import sys
import time
import threading
from datetime import datetime
import math

from similari import Universal2DBox, Sort, SpatioTemporalConstraints, PositionalMetricType, BoundingBox
from mmdet.apis import inference_detector, init_detector


config = "../rotated_rtmdet_l_3x_v4/rotated_rtmdet_l-3x-dota_ms_custom_v4.py"
checkpoint = "../rotated_rtmdet_l_3x_v4/epoch_96.pth"

device = 'cuda:0'
model = init_detector(config, checkpoint, device)

font_face = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_thickness = 2
font_color = (255, 255, 255)
text_offset = np.array([0, -10]).reshape((1, 1, -1)).astype(np.int32)

labels_name = ("paper", "metal", "plastic", "nilon", "glass", "fabric")
pi = 22/7

constraints = SpatioTemporalConstraints()
constraints.add_constraints([(1, 1.0)])

iou_metric = PositionalMetricType.iou(threshold=0.09)
maha_metric = PositionalMetricType.maha()

tracker = Sort(
    shards=4,
    bbox_history=2*4,
    max_idle_epochs=4,
    method=iou_metric,
    spatio_temporal_constraints=constraints,
)


class ServerSocket:

    def __init__(self, ip, port):
        self.TCP_IP = ip
        self.TCP_PORT = port
        self.createImageDir()
        self.folder_num = 0
        self.socketOpen()
        self.receiveImages()
        # self.receiveThread = threading.Thread(target=self.receiveImages)
        # self.receiveThread.start()

    def socketClose(self):
        self.sock.close()
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is close')

    def socketOpen(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.TCP_IP, self.TCP_PORT))
        self.sock.listen(1)
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is open')
        self.conn, self.addr = self.sock.accept()
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' +
              str(self.TCP_PORT) + ' ] is connected with client')

    def receiveImages(self):
        cnt = 0
        try:
            while True:
                if cnt > 100:
                    cnt = 0
                cnt += 1
                length = self.recvall(self.conn, 64)
                if not length:
                    print("End processing")
                    break

                length1 = length.decode('utf-8')
                stringData = self.recvall(self.conn, int(length1))
                stime = self.recvall(self.conn, 64)
                print('send time: ' + stime.decode('utf-8'))
                now = time.localtime()
                print('receive time: ' + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))
                data = numpy.frombuffer(base64.b64decode(stringData), numpy.uint8)
                decimg = cv2.imdecode(data, 1)
                # cv2.imshow("image", decimg)
                # cv2.imwrite('./' + str(self.TCP_PORT) + '_images' +
                #             str(self.folder_num) + '/img' + cnt_str + '.jpg', decimg)
                new_img, vision_resutls = self.handle_input(decimg)

                # send back to socket client
                print("sending back")
                stringData = base64.b64encode(vision_resutls)
                length = str(len(stringData))
                self.conn.sendall(length.encode('utf-8').ljust(64))
                self.conn.send(stringData)
                self.conn.send(base64.b64encode(np.array(vision_resutls.shape)).ljust(64))
                print("send done")
                print("================================================================")

                cv2.imwrite('./' + str(self.TCP_PORT) + '_images' +
                            str(self.folder_num) + '/img' + str(cnt) + '.jpg', new_img)

        except Exception as e:
            print(e)

    def createImageDir(self):

        folder_name = str(self.TCP_PORT) + "_images0"
        try:
            if not os.path.exists(folder_name):
                os.makedirs(os.path.join(folder_name))
        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create " + folder_name + " directory")
                raise

    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def handle_input(self, frame):

        result = inference_detector(model, frame)

        bboxes = result.pred_instances.bboxes.cpu().numpy()
        pred_scores = result.pred_instances.scores.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()
        confidence_threshold = 0.80
        filtered_indices = np.where(pred_scores > confidence_threshold)[0]
        frame_detections = []
        vision_results = []

        for idx in filtered_indices:
            bbox = bboxes[idx]
            xc, yc, width, height, radian = bbox

            # with rotated bounding box
            frame_detections.append((
                Universal2DBox.new_with_confidence(
                    xc=xc,
                    yc=yc,
                    angle=1.5708,
                    aspect=width/height,
                    height=height,
                    confidence=1),
                None))

        res = tracker.predict(frame_detections)

        for i, idx in enumerate(filtered_indices):
            bbox = bboxes[idx]
            # draw bbox
            xc, yc, width, height, radian = bbox
            degree = radian * (180/pi)
            points = cv2.boxPoints(([xc, yc], (width, height), degree))
            pts = np.array(points).reshape((1, -1, 1, 2)).astype(np.int32)
            cv2.polylines(frame, pts, True, (0, 255, 0), 2)

            # draw class
            label_index = labels[idx]
            label = labels_name[label_index]
            point_text_orig = pts[0, 1, :] + text_offset

            frame = cv2.putText(frame, label,
                                point_text_orig.ravel(),
                                font_face, font_scale,
                                font_color, font_thickness,
                                cv2.LINE_8,)

            pred_score = pred_scores[idx]
            obj = res[i]
            obj_id = obj.id
            # text = f"#{obj_id}"

            # Accept conditions
            (dy, dx) = frame.shape[1], frame.shape[0]
            if ((3 * dy/4 > yc > dx/4) and label in ['plastic', 'paper']):

                # degree = cal_degree_from_box(bbox)
                vision_results.append([xc, yc, width, height, degree,
                                       obj_id, label_index, pred_score])

            point_text_orig = pts[0, 0, :] + text_offset

            ps = np.array(points).astype(np.int32)
            frame = cv2.circle(frame, ps[0], 10, (0, 0, 255), -1)
            frame = cv2.circle(frame, ps[1], 10, (0, 255, 255), -1)
            frame = cv2.circle(frame, ps[2], 10, (255, 255, 255), -1)
            frame = cv2.circle(frame, ps[3], 10, (0, 0, 0), -1)
            dis1 = math.dist(ps[0], ps[1])
            dis2 = math.dist(ps[0], ps[3])
            print(dis1, height, dis2, width)
            if width <= height:
                p1, p2 = points[0], points[1]
            else:
                p1, p2 = points[0], points[3]

            print(p1, p2)

            # frame = cv2.circle(frame, (int(p1[0] + 50), int(p1[1])), 10, (0, 0, 255), -1)
            frame = cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p1[0] + 100), int(p1[1])), (255, 255, 255), 2)

            angle = get_angle(p2, p1, (p1[0] + 50, p1[1]))
            print('checking1')
            angle = convert_angle(angle)
            print('checking2')
            text = '{:.2f}*{}'.format(angle, label_index)

            frame = cv2.putText(frame, text,
                                point_text_orig.ravel(),
                                font_face, font_scale,
                                (0, 200, 255), font_thickness,
                                cv2.LINE_4,)

        vision_results = np.array(vision_results)
        return frame, vision_results


def convert_angle(a):
    if 270 >= a > 90:
        return a - 180
    elif 360 >= a > 270:
        return a - 360
    else:
        return a


def get_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang


def main(TCP_IP, TCP_PORT):
    # TCP_IP = '0.0.0.0'
    # TCP_PORT = 8080
    server = ServerSocket(TCP_IP, TCP_PORT)


if __name__ == "__main__":

    TCP_IP = '0.0.0.0'
    TCP_PORT = 9080
    while True:
        main(TCP_IP, TCP_PORT)
        TCP_PORT = 9080 if TCP_PORT == 9081 else 9081
