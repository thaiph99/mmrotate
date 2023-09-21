import socket
import cv2
import numpy as np
import time
import base64
import sys
from datetime import datetime


class ClientSocket:
    def __init__(self, ip, port):
        self.TCP_SERVER_IP = ip
        self.TCP_SERVER_PORT = port
        self.connectCount = 0
        self.connectServer()

    def connectServer(self):
        try:
            self.sock = socket.socket()
            self.sock.connect((self.TCP_SERVER_IP, self.TCP_SERVER_PORT))
            print(u'Client socket is connected with Server socket [ TCP_SERVER_IP: ' +
                  self.TCP_SERVER_IP + ', TCP_SERVER_PORT: ' + str(self.TCP_SERVER_PORT) + ' ]')
            self.connectCount = 0
            self.sendImages()
        except Exception as e:
            print(e)
            self.connectCount += 1
            if self.connectCount == 10:
                print(u'Connect fail %d times. exit program' % (self.connectCount))
                sys.exit()
            print(u'%d times try to connect with server' % (self.connectCount))
            self.TCP_SERVER_PORT = 8080 if self.TCP_SERVER_PORT == 8081 else 8081
            self.connectServer()

    def sendImages(self):
        cnt = 0
        capture = cv2.VideoCapture('/home/thaiph/data/1280x_240723/output2.avi')
        # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 315)
        try:
            while capture.isOpened():
                ret, frame = capture.read()
                if not ret:
                    break
                cv2.imshow('origin_frame.jpg', frame)
                # frame = cv2.imread('/home/thaipham/Documents/img_labeled/images2/frame150.jpg')
                x_scale = 315
                y_scale = 480
                x_rate = frame.shape[0]/x_scale
                y_rate = frame.shape[1]/y_scale
                print("check frame size before: ", frame.shape)
                resize_frame = cv2.resize(frame, dsize=(x_scale, y_scale), interpolation=cv2.INTER_AREA)
                print("check frame size after: ", resize_frame.shape)

                now = time.localtime()
                stime = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')

                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                # result, imgencode = cv2.imencode('.jpg', resize_frame, encode_param)
                result, imgencode = cv2.imencode('.jpg', frame, encode_param)
                data = np.array(imgencode)
                stringData = base64.b64encode(data)
                length = str(len(stringData))
                self.sock.sendall(length.encode('utf-8').ljust(64))
                self.sock.send(stringData)
                self.sock.send(stime.encode('utf-8').ljust(64))
                print(u'send images %d' % (cnt))

                length_res = self.recvall(self.sock, 64)
                length1 = length_res.decode('utf-8')
                stringData = self.recvall(self.sock, int(length1))
                res_shape = self.recvall(self.sock, 64)
                res_shape = np.frombuffer(base64.b64decode(res_shape), np.uint)

                if res_shape[0] == 0:
                    continue

                vision_result_per_frame = np.frombuffer(base64.b64decode(stringData), np.float64)
                vision_result_per_frame = vision_result_per_frame.reshape(res_shape)

                (xc, yc, width, height, degrees, obj_id, label_idex, pred_score) = vision_result_per_frame[0]
                points = cv2.boxPoints(([xc, yc], (width, height), degrees))
                pts = np.array(points).reshape((1, -1, 1, 2)).astype(np.int32)
                cv2.polylines(frame, pts, True, (0, 255, 0), 2)
                print('================================================================')

                cv2.imwrite('rescale_frame.jpg', frame)

                ################################
                # Robot code in here
                ################################

                cnt += 1
                # time.sleep(0.095)
        except Exception as e:
            print(e)
            self.sock.close()

    def send_numpy_img(self, np_img: np.ndarray):
        """
        return numpy array with shape (n, 4) 
        each element is (xc, yc, w, h, radian)
        """
        resize_frame = cv2.resize(np_img, dsize=(480, 315), interpolation=cv2.INTER_AREA)

        now = time.localtime()
        stime = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, imgencode = cv2.imencode('.jpg', resize_frame, encode_param)
        data = np.array(imgencode)
        stringData = base64.b64encode(data)
        length = str(len(stringData))
        self.sock.sendall(length.encode('utf-8').ljust(64))
        self.sock.send(stringData)
        self.sock.send(stime.encode('utf-8').ljust(64))
        print(u'send images %d' % (cnt))

        res = self.sock.recvall()
        print(res.encode())

        cnt += 1

    def get_recv(self):
        """
        get result        
        """
        pass

    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf


def main():
    TCP_IP = 'localhost'
    TCP_PORT = 8080
    client = ClientSocket(TCP_IP, TCP_PORT)


if __name__ == "__main__":
    # while True:
    main()
