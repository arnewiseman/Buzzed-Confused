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
from cflib.positioning.motion_commander import MotionCommander


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
            p = self._cpx.receivePacket(CPXFunction.APP)
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

        self.distance = None
        self.angle = None
        self.x_pos = None
        self.y_pos = None
        self.found_tag = False
    
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
                        # Draw tag as before...
                        for i in range(4):
                            pt1 = tuple(map(int, tag.corners[i - 1]))
                            pt2 = tuple(map(int, tag.corners[i]))
                            cv2.line(rgb, pt1, pt2, (0, 255, 0), 2)
                        center = tuple(map(int, tag.center))
                        cv2.circle(rgb, center, 4, (0, 0, 255), -1)
                        cv2.putText(rgb, f"ID {tag.tag_id}", center,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                        # === 2D DISTANCE ESTIMATION ===
                        TAG_SIZE = 0.128
                        fx = 200.0  

                        # Estimate pixel width from corner 0 to 1
                        pixel_width = np.linalg.norm(tag.corners[0] - tag.corners[1])

                        if pixel_width > 0:
                            self.distance = (TAG_SIZE * fx) / pixel_width
                        else:
                            self.distance = -1  # Invalid

                        # Compute angle from image center
                        cx = CAM_WIDTH / 2
                        cy = CAM_HEIGHT / 2

                        dx = tag.center[0] - cx
                        angle_rad = np.arctan2(dx, fx)

                        # Get relative X,Y position (same plane)
                        self.x_pos = self.distance * np.sin(angle_rad)
                        self.y_pos = self.distance * np.cos(angle_rad)
                        self.angle = np.degrees(angle_rad)

                        # print(f"Distance: {self.distance:.2f}m, Angle: {self.angle:.1f}°, Position: x={self.x_pos:.2f} m, y={self.y_pos:.2f} m")
                        self.found_tag = True

                    qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                                        rgb.shape[1] * 3, QtGui.QImage.Format.Format_RGB888)
                    qimg = qimg.scaled(CAM_WIDTH * 2, CAM_HEIGHT * 2,
                                       QtCore.Qt.AspectRatioMode.KeepAspectRatio)
                    self.image_processed.emit(qimg)

                    if tags:
                        self.tags_detected.emit(tags)

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
        self.cf.disconnected.add_callback(self.disconnected)
        
        print('Connecting to %s' % URI)
        self.cf.open_link(URI)

        while not self.cf.is_connected():
            time.sleep(0.1)
        self.connected(URI)

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
        self.mc = MotionCommander(self.cf, default_height=0.5)
        # self.light_check()
        self.start_search()

    def closeEvent(self, event):
        self.processor.stop()
        event.accept()

    def activate_led_bit_mask(self):
        self.cf.param.set_value('led.bitmask', 255)

    def deactivate_led_bit_mask(self):
        self.cf.param.set_value('led.bitmask', 0)

    def light_check(self):
        self.activate_led_bit_mask()
        time.sleep(2)
        self.deactivate_led_bit_mask()



    def start_search(self):
        def _search():
            with self.mc:
                distance = 0.25
                inc = 0.25
                step = 0.05

                while True:
                    # Loop twice before incrementing the distance
                    for _ in range(2):
                        steps = int(distance / step)
                        # Move forward
                        for _ in range(steps):
                            if self.processor.found_tag:
                                self.mc.stop()
                                self.light_check()
                                return
                            self.mc.left(step)
                        # Turn 90 left
                        time.sleep(0.5)
                        self.mc.turn_left(90)
                    distance += inc



        threading.Thread(target=_search, daemon=True).start()

# Crazyflie URI and camera settings
URI = uri_helper.uri_from_env(default='tcp://192.168.4.1:5000')
CAM_HEIGHT = 244
CAM_WIDTH = 324


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(URI)
    window.show()
    sys.exit(app.exec())
