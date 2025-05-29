#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import struct
import threading
import numpy as np
import cv2
from pupil_apriltags import Detector
import time
from PyQt6 import QtCore, QtWidgets, QtGui

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.cpx import CPXFunction
from cflib.utils import uri_helper

# Camera settings
CAM_WIDTH = 324
CAM_HEIGHT = 244



class ImageDownloader(threading.Thread):
    def __init__(self, cpx, cb):
        super().__init__()
        self.daemon = True
        self._cpx = cpx
        self._cb = cb

    def run(self):
        while True:
            p = self._cpx.receivePacket(CPXFunction.APP)  # CPX_F_APP == 5
            [magic, width, height, depth, format, size] = struct.unpack('<BHHBBI', p.data[0:11])
            if magic == 0xBC:
                imgStream = bytearray()
                while len(imgStream) < size:
                    p = self._cpx.receivePacket(CPXFunction.APP)
                    imgStream.extend(p.data)
                bayer_img = np.frombuffer(imgStream, dtype=np.uint8)
                self._cb(bayer_img)

class ImageProcessor(QtCore.QThread):
    image_processed = QtCore.pyqtSignal(QtGui.QImage)
    tags_detected = QtCore.pyqtSignal(list) 


    def __init__(self):
        super().__init__()
        self.detector = Detector(
            families='tag25h9', 
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
        self._frame_queue = []
        self._lock = threading.Lock()
        self._running = True
    
    def stop(self):
        self._running = False
        self.wait()

    def run(self):
        while self._running:
            frame = None
            with self._lock:
                if self._frame_queue:
                    frame = self._frame_queue.pop(0)

            if frame is not None:
                try:
                    # Reshape to 2D array
                    raw = frame.reshape((CAM_HEIGHT, CAM_WIDTH))
                    
                    # Convert Bayer pattern to RGB directly
                    # BGGR pattern - start with Blue pixel
                    rgb = cv2.cvtColor(raw, cv2.COLOR_BayerBG2RGB)
                    
                    # Create grayscale copy for tag detection
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    
                    # Mild image enhancement for tag detection
                    gray = cv2.GaussianBlur(gray, (3, 3), 0)
                    
                    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) instead of global
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    gray = clahe.apply(gray)
                    
                    # Run AprilTag detection on enhanced grayscale
                    tags = self.detector.detect(gray)
                    
                    for tag in tags:
                        for i in range(4):
                            pt1 = tuple(map(int, tag.corners[i - 1]))
                            pt2 = tuple(map(int, tag.corners[i]))
                            cv2.line(rgb, pt1, pt2, (0, 255, 0), 2)
                        center = tuple(map(int, tag.center))
                        cv2.circle(rgb, center, 4, (0, 0, 255), -1)
                        cv2.putText(rgb, f"ID {tag.tag_id}", center,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                                        rgb.shape[1] * 3, QtGui.QImage.Format.Format_RGB888)
                    qimg = qimg.scaled(CAM_WIDTH * 2, CAM_HEIGHT * 2,
                                       QtCore.Qt.AspectRatioMode.KeepAspectRatio)
                    self.image_processed.emit(qimg)

                    if tags:
                        print(f"[AprilTag] Found {len(tags)} tag(s): {[tag.tag_id for tag in tags]}")
                        self.tags_detected.emit(tags)  # Emit detected tags

                except Exception as e:
                    print("Processing error:", e)

            time.sleep(0.01)

    def add_frame(self, frame):
        with self._lock:
            # Keep only the most recent frames to prevent lag
            if len(self._frame_queue) < 2:  # Only buffer up to 2 frames
                self._frame_queue.append(frame)
            else:
                # If we're falling behind, skip frames
                self._frame_queue[-1] = frame


class MainWindow(QtWidgets.QWidget):
    def __init__(self, URI):
        super().__init__()
        self.setWindowTitle('Crazyflie AI deck Camera + AprilTag Demo')
        
        # Set up the UI
        self.mainLayout = QtWidgets.QVBoxLayout()
        self.image_frame = QtWidgets.QLabel()
        self.mainLayout.addWidget(self.image_frame)
        self.setLayout(self.mainLayout)

        # Set up image processing
        self.processor = ImageProcessor()
        self.processor.image_processed.connect(self.displayImage)
        self.processor.tags_detected.connect(self.handle_tags)
        self.processor.start()

        # Connect to Crazyflie
        cflib.crtp.init_drivers()
        self.cf = Crazyflie(ro_cache=None, rw_cache='cache')
        self.cf.connected.add_callback(self.connected)
        self.cf.disconnected.add_callback(self.disconnected)
        
        print('Connecting to %s' % URI)
        self.cf.open_link(URI)

        if not hasattr(self.cf.link, 'cpx'):
            print('Not connecting with WiFi')
            self.cf.close_link()
            sys.exit(1)

        # Start camera thread
        self._imgDownload = ImageDownloader(self.cf.link.cpx, self.processor.add_frame)
        self._imgDownload.start()

    def displayImage(self, qimg):
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def handle_tags(self, tags):
        tag_ids = [tag.tag_id for tag in tags]
        print(f"Tags detected in MainWindow: {tag_ids}")

    def disconnected(self, uri):
        print('Disconnected')
        self.processor.stop()
        sys.exit(1)

    def connected(self, uri):
        print('Connected to {}'.format(uri))

    def closeEvent(self, event):
        self.processor.stop()
        event.accept()


# Crazyflie URI and camera settings
URI = uri_helper.uri_from_env(default='tcp://192.168.4.1:5000')
CAM_HEIGHT = 244
CAM_WIDTH = 324


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(URI)
    window.show()
    sys.exit(app.exec())
