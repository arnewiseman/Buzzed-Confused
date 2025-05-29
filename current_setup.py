#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import threading
import struct
import sys
import time
from threading import Event

import numpy as np
import cv2
import apriltag

from PyQt6 import QtCore, QtWidgets, QtGui

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.cpx import CPXFunction
from cflib.utils import uri_helper

logging.basicConfig(level=logging.INFO)

URI = uri_helper.uri_from_env(default='tcp://192.168.4.1:5000')
CAM_HEIGHT = 244
CAM_WIDTH = 324


class ImageDownloader(threading.Thread):
    def __init__(self, cpx, cb):
        threading.Thread.__init__(self)
        self.daemon = True
        self._cpx = cpx
        self._cb = cb

    def run(self):
        while True:
            p = self._cpx.receivePacket(CPXFunction.APP)
            [magic, width, height, depth, format, size] = struct.unpack('<BHHBBI', p.data[0:11])
            if magic == 0xBC:
                imgStream = bytearray()
                while len(imgStream) < size:
                    p = self._cpx.receivePacket(CPXFunction.APP)
                    imgStream.extend(p.data)
                bayer_img = np.frombuffer(imgStream, dtype=np.uint8)
                self._cb(bayer_img)


class MainWindow(QtWidgets.QWidget):
    # Move this line OUT of __init__ and define it as a class variable:
    image_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, URI):
        super().__init__()
        self.setWindowTitle('Crazyflie / AI deck FPV + AprilTag Demo')

        self.mainLayout = QtWidgets.QVBoxLayout()
        self.image_frame = QtWidgets.QLabel()
        self.mainLayout.addWidget(self.image_frame)
        self.setLayout(self.mainLayout)

        options = apriltag.DetectorOptions(
            families="tag36h11",
            border=1,
            nthreads=4,
            quad_decimate=1.0,  # try lowering to 1.0 or 0.8 (default is 2.0 sometimes)
            refine_edges=True,
            refine_decode=True,
            refine_pose=False,
        )
        self.detector = apriltag.Detector(options)

        cflib.crtp.init_drivers()
        self.cf = Crazyflie(ro_cache=None, rw_cache='cache')
        self.cf.connected.add_callback(self.connected)
        self.cf.disconnected.add_callback(self.disconnected)
        self.cf.open_link(URI)

        if not hasattr(self.cf.link, 'cpx'):
            print('Not connecting with WiFi')
            self.cf.close_link()
            sys.exit(1)

        # Connect signal to slot
        self.image_signal.connect(self.updateImage)

        self._imgDownload = ImageDownloader(self.cf.link.cpx, self.emit_image_signal)
        self._imgDownload.start()

    def emit_image_signal(self, image):
        # Emit the raw bayer image to Qt thread-safe slot
        self.image_signal.emit(image)

    @QtCore.pyqtSlot(np.ndarray)
    def updateImage(self, image):
        try:
            # Reshape raw Bayer data into 2D image
            bayer_img = image.reshape((CAM_HEIGHT, CAM_WIDTH)).astype(np.uint8)

            # Convert Bayer to RGB
            rgb = cv2.cvtColor(bayer_img, cv2.COLOR_BAYER_BG2RGB)

            # Convert RGB to grayscale
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

            # Detect AprilTags on grayscale
            tags = self.detector.detect(gray)

            # Draw tags on RGB image
            for tag in tags:
                for i in range(4):
                    pt1 = tuple(map(int, tag.corners[i - 1]))
                    pt2 = tuple(map(int, tag.corners[i]))
                    cv2.line(rgb, pt1, pt2, (0, 255, 0), 2)
                center = tuple(map(int, tag.center))
                cv2.circle(rgb, center, 4, (0, 0, 255), -1)
                cv2.putText(rgb, f"ID {tag.tag_id}", center,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
            qimg = qimg.scaled(CAM_WIDTH * 2, CAM_HEIGHT * 2, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.image_frame.setPixmap(QtGui.QPixmap.fromImage(qimg))

            if tags:
                print(f"[AprilTag] Found {len(tags)} tag(s): {[tag.tag_id for tag in tags]}")

        except Exception as e:
            print("Error in updateImage:", e)

    def disconnected(self, URI):
        print('Disconnected')
        sys.exit(1)

    def connected(self, URI):
        print('Connected to {}'.format(URI))


if __name__ == '__main__':
    appQt = QtWidgets.QApplication(sys.argv)
    win = MainWindow(URI)
    win.show()
    appQt.exec()
