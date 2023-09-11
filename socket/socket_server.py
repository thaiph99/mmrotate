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
        cnt_str = ''
        cnt = 0

        try:
            while True:
                if (cnt < 10):
                    cnt_str = '000' + str(cnt)
                elif (cnt < 100):
                    cnt_str = '00' + str(cnt)
                elif (cnt < 1000):
                    cnt_str = '0' + str(cnt)
                else:
                    cnt_str = str(cnt)
                if cnt == 0:
                    startTime = time.localtime()
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

                cv2.imwrite('./' + str(self.TCP_PORT) + '_images' +
                            str(self.folder_num) + '/img' + cnt_str + '.jpg', new_img)
                # cv2.waitKey(1)
                # if (cnt == 60 * 10):
                #     # for multi processing
                #     cnt = 0
                #     convertThread = threading.Thread(target=self.convertImage(str(self.folder_num), 600, startTime))
                #     convertThread.start()
                #     self.folder_num = (self.folder_num + 1) % 2

                ####################
                # code robot in here
                ####################

        except Exception as e:
            print(e)
            # self.convertImage(str(self.folder_num), cnt, startTime)
            # self.socketClose()
            # cv2.destroyAllWindows()
            # self.socketOpen()
            # self.receiveThread = threading.Thread(target=self.receiveImages)
            # self.receiveThread.start()

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

    def convertImage(self, fnum, count, now):
        img_array = []
        cnt = 0
        for filename in glob.glob('./' + str(self.TCP_PORT) + '_images' + fnum + '/*.jpg'):
            if (cnt == count):
                break
            cnt = cnt + 1
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        file_date = self.getDate(now)
        file_time = self.getTime(now)
        name = 'video(' + file_date + ' ' + file_time + ').mp4'
        file_path = './videos/' + name
        out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'.mp4'), 20, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        print(u'complete')

    def getDate(self, now):
        year = str(now.tm_year)
        month = str(now.tm_mon)
        day = str(now.tm_mday)

        if len(month) == 1:
            month = '0' + month
        if len(day) == 1:
            day = '0' + day
        return (year + '-' + month + '-' + day)

    def getTime(self, now):
        file_time = (str(now.tm_hour) + '_' + str(now.tm_min) + '_' + str(now.tm_sec))
        return file_time

    def handle_input(self, frame):

        result = inference_detector(model, frame)

        bboxes = result.pred_instances.bboxes.cpu().numpy()
        pred_scores = result.pred_instances.scores.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()
        confidence_threshold = 0.65
        pi = 22/7
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
            degrees = radian * (180/pi)
            points = cv2.boxPoints(([xc, yc], (width, height), degrees))
            pts = np.array(points).reshape((1, -1, 1, 2)).astype(np.int32)
            cv2.polylines(frame, pts, True, (0, 255, 0), 2)

            # draw class
            label = labels_name[labels[idx]]
            point_text_orig = pts[0, 1, :] + text_offset

            frame = cv2.putText(frame, label,
                                point_text_orig.ravel(),
                                font_face, font_scale,
                                font_color, font_thickness,
                                cv2.LINE_8,)

            obj = res[i]
            obj_id = obj.id
            text = f"#{obj_id}"
            vision_results.append([xc, yc, width, height, degrees, obj_id])

            point_text_orig = pts[0, 0, :] + text_offset
            frame = cv2.putText(frame, text,
                                point_text_orig.ravel(),
                                font_face, font_scale,
                                (0, 200, 255), font_thickness,
                                cv2.LINE_4,)

        vision_results = np.array(vision_results)
        print("================================================================")
        print(vision_results.shape)
        return frame, vision_results


def main(TCP_IP, TCP_PORT):
    # TCP_IP = '0.0.0.0'
    # TCP_PORT = 8080
    server = ServerSocket(TCP_IP, TCP_PORT)


if __name__ == "__main__":

    TCP_IP = '0.0.0.0'
    TCP_PORT = 8080
    while True:
        main(TCP_IP, TCP_PORT)
        TCP_PORT = 8080 if TCP_PORT == 8081 else 8081
