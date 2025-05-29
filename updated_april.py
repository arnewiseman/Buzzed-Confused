#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import struct
import sys
import threading
import time

import cv2
import numpy as np
from pupil_apriltags import Detector
from PyQt6 import QtCore, QtGui, QtWidgets

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.cpx import CPXFunction
from cflib.utils import uri_helper

logging.basicConfig(level=logging.INFO)

URI = uri_helper.uri_from_env(default="tcp://192.168.4.1:5000")
CAM_HEIGHT = 244
CAM_WIDTH = 324


class ImageDownloader(threading.Thread):
    """Fetch raw frame packets from the AI‑deck over CPX."""

    def __init__(self, cpx, cb):
        super().__init__(daemon=True)
        self._cpx = cpx
        self._cb = cb

    def run(self):
        while True:
            p = self._cpx.receivePacket(CPXFunction.APP)
            magic, width, height, depth, fmt, size = struct.unpack(
                "<BHHBBI", p.data[0:11])
            if magic == 0xBC:  # start‑of‑frame marker
                img_stream = bytearray()
                while len(img_stream) < size:
                    p = self._cpx.receivePacket(CPXFunction.APP)
                    img_stream.extend(p.data)
                self._cb(np.frombuffer(img_stream, dtype=np.uint8))


class MainWindow(QtWidgets.QWidget):
    image_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, URI: str):
        super().__init__()
        self.setWindowTitle("Crazyflie / AI‑deck FPV + AprilTag Demo")

        # -------- GUI ----------
        self.image_frame = QtWidgets.QLabel()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.image_frame)

        # -------- Detector ----------
        self.detector = Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=2.0,  # down‑sample for speed
            quad_sigma=0.0,
            refine_edges=True,
            refine_decode=True,
            refine_pose=False,
        )

        # -------- Crazyflie link ----------
        cflib.crtp.init_drivers()
        self.cf = Crazyflie(ro_cache=None, rw_cache="cache")
        self.cf.connected.add_callback(self.connected)
        self.cf.disconnected.add_callback(self.disconnected)
        self.cf.open_link(URI)

        if not hasattr(self.cf.link, "cpx"):
            print("Not connecting over Wi‑Fi (AI‑deck missing?)")
            self.cf.close_link()
            sys.exit(1)

        # Thread‑safe hand‑off of frames to the Qt thread
        self.image_signal.connect(self.updateImage)
        self._img_thread = ImageDownloader(
            self.cf.link.cpx, self.image_signal.emit)
        self._img_thread.start()

    # ---------- Callbacks ----------

    @QtCore.pyqtSlot(np.ndarray)
    def updateImage(self, frame: np.ndarray):
        try:
            # Interpret incoming bytes as 8‑bit grayscale
            gray = frame.reshape((CAM_HEIGHT, CAM_WIDTH)).astype(np.uint8)
            gray_eq = cv2.equalizeHist(gray)  # optional contrast boost

            # AprilTag detection (fast C++ backend)
            tags = self.detector.detect(gray_eq)

            # For display only: upscale & recolour
            rgb = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)
            for tag in tags:
                for i in range(4):
                    cv2.line(
                        rgb,
                        tuple(map(int, tag.corners[i - 1])),
                        tuple(map(int, tag.corners[i])),
                        (0, 255, 0),
                        2,
                    )
                center = tuple(map(int, tag.center))
                cv2.circle(rgb, center, 4, (0, 0, 255), -1)
                cv2.putText(
                    rgb,
                    f"ID {tag.tag_id}",
                    center,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

            h, w, ch = rgb.shape
            qt_img = QtGui.QImage(rgb.data, w, h, ch * w,
                                  QtGui.QImage.Format.Format_RGB888)
            qt_img = qt_img.scaled(
                CAM_WIDTH * 2, CAM_HEIGHT * 2, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.image_frame.setPixmap(QtGui.QPixmap.fromImage(qt_img))

            if tags:
                print(
                    f"[AprilTag] Detected {len(tags)} tag(s): {[t.tag_id for t in tags]}")

        except Exception as e:
            print("updateImage error:", e)

    def connected(self, uri):
        print(f"Connected to {uri}")

    def disconnected(self, uri):
        print("Disconnected")
        sys.exit(1)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(URI)
    win.show()
    app.exec()
