#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crazyflie AI‑deck FPV + AprilTag viewer + manual pilot
– Xbox controller (pygame) or keyboard (PyQt key events)
"""
import logging
import struct
import sys
import threading
import time

import cv2
import numpy as np
import pygame                               # <-- game‑pad
from pupil_apriltags import Detector
from PyQt6 import QtCore, QtGui, QtWidgets

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.cpx import CPXFunction
from cflib.utils import uri_helper

# ------------- global constants ------------------------------------------------
URI = uri_helper.uri_from_env(default="tcp://192.168.4.1:5000")
CAM_H, CAM_W = 244, 324
RATE_HZ = 50                            # command rate

# set‑point scales (tweak!)
MAX_ROLL_PITCH = 20_000   # ±20 deg  (Crazyflie 1 rad ≈ 57 000)
MAX_YAW_RATE = 250      # deg/s
MAX_THRUST = 50_000   # out of 65 535

logging.basicConfig(level=logging.INFO)


# ===============================================================================
# Camera reception thread
# ===============================================================================
class ImageDownloader(threading.Thread):
    def __init__(self, cpx, cb):
        super().__init__(daemon=True)
        self._cpx = cpx
        self._cb = cb

    def run(self):
        while True:
            p = self._cpx.receivePacket(CPXFunction.APP)
            magic, w, h, depth, fmt, size = struct.unpack(
                "<BHHBBI", p.data[0:11])
            if magic != 0xBC:
                continue
            buf = bytearray()
            while len(buf) < size:
                buf.extend(self._cpx.receivePacket(CPXFunction.APP).data)
            self._cb(np.frombuffer(buf, dtype=np.uint8))


# ===============================================================================
# Manual‑control helper (reads pygame + PyQt state and sends set‑points)
# ===============================================================================
class Pilot(QtCore.QThread):
    """Runs at RATE_HZ and pushes attitude set‑points to the Crazyflie."""

    def __init__(self, cf: Crazyflie, input_state):
        super().__init__(daemon=True)
        self.cf = cf
        self.input = input_state           # dict shared with GUI
        self._running = True

    def run(self):
        period = 1.0 / RATE_HZ
        commander = self.cf.commander
        while self._running and self.cf.is_connected():
            # snapshot the latest stick values
            with self.input["lock"]:
                roll = self.input["roll"]
                pitch = self.input["pitch"]
                yaw = self.input["yaw"]
                thrust = self.input["thrust"]
                armed = self.input["armed"]
            if armed:
                commander.send_setpoint(roll, pitch, yaw, thrust)
            else:
                commander.send_setpoint(0, 0, 0, 0)
            time.sleep(period)

    def stop(self):
        self._running = False
        self.wait()


# ===============================================================================
# Main Qt window
# ===============================================================================
class MainWindow(QtWidgets.QWidget):
    image_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, uri):
        super().__init__()
        self.setWindowTitle("Crazyflie FPV – Xbox/Keyboard Pilot")

        # ---------- UI ----------
        self.label = QtWidgets.QLabel()
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(self.label)

        # ---------- Crazyflie ----------
        cflib.crtp.init_drivers()
        self.cf = Crazyflie(ro_cache=None, rw_cache="cache")
        self.cf.connected.add_callback(self._connected)
        self.cf.disconnected.add_callback(self._disconnected)
        self.cf.open_link(uri)

        # ---------- AprilTag detector ----------
        self.det = Detector(families="tag36h11", nthreads=4,
                            quad_decimate=2.0, refine_edges=True)

        # ---------- shared input state ----------
        self.state = {
            "roll": 0, "pitch": 0, "yaw": 0, "thrust": 0,
            "armed": False,
            "lock": threading.Lock()
        }

        # ---------- pygame (game‑pad) ----------
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count():
            self.joy = pygame.joystick.Joystick(0)
            self.joy.init()
            logging.info("Game‑pad detected: %s", self.joy.get_name())
            self.timer_pygame = QtCore.QTimer(self)
            self.timer_pygame.timeout.connect(self.poll_gamepad)
            self.timer_pygame.start(20)    # 50 Hz
        else:
            self.joy = None
            logging.info("No game‑pad found – keyboard only")

        # ---------- Qt connections ----------
        self.image_signal.connect(self.update_image)

        # ---------- will start pilot thread after cf connects ----------
        self.pilot = None

    # -----------------------------------------------------------------------
    # Crazyflie callbacks
    # -----------------------------------------------------------------------
    def _connected(self, uri):
        print(f"Connected to {uri}")
        if not hasattr(self.cf.link, "cpx"):
            print("AI‑deck CPX link missing – camera will not stream.")
            return

        # camera thread
        self.cam_thread = ImageDownloader(
            self.cf.link.cpx, self.image_signal.emit)
        self.cam_thread.start()

        # pilot thread
        self.pilot = Pilot(self.cf, self.state)
        self.pilot.start()

    def _disconnected(self, uri):
        print("Disconnected")
        if self.pilot:
            self.pilot.stop()
        QtWidgets.QApplication.quit()

    # -----------------------------------------------------------------------
    # Frame processing
    # -----------------------------------------------------------------------
    @QtCore.pyqtSlot(np.ndarray)
    def update_image(self, data):
        gray = data.reshape((CAM_H, CAM_W)).astype(np.uint8)
        tags = self.det.detect(gray)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        for t in tags:
            for i in range(4):
                cv2.line(rgb, tuple(t.corners[i - 1].astype(int)),
                         tuple(t.corners[i].astype(int)), (0, 255, 0), 2)
            center = tuple(t.center.astype(int))
            cv2.circle(rgb, center, 4, (255, 0, 0), -1)
            cv2.putText(rgb, f"ID {t.tag_id}", center,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        img_qt = QtGui.QImage(rgb.data, CAM_W, CAM_H, 3 * CAM_W,
                              QtGui.QImage.Format.Format_RGB888)
        img_qt = img_qt.scaled(CAM_W*2, CAM_H*2,
                               QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.label.setPixmap(QtGui.QPixmap.fromImage(img_qt))

    # -----------------------------------------------------------------------
    # Keyboard input
    # -----------------------------------------------------------------------
    def keyPressEvent(self, e):
        key = e.key()
        with self.state["lock"]:
            if key in (QtCore.Qt.Key_W, QtCore.Qt.Key_Up):
                self.state["pitch"] = -MAX_ROLL_PITCH
            elif key in (QtCore.Qt.Key_S, QtCore.Qt.Key_Down):
                self.state["pitch"] = MAX_ROLL_PITCH
            elif key in (QtCore.Qt.Key_A, QtCore.Qt.Key_Left):
                self.state["roll"] = -MAX_ROLL_PITCH
            elif key in (QtCore.Qt.Key_D, QtCore.Qt.Key_Right):
                self.state["roll"] = MAX_ROLL_PITCH
            elif key == QtCore.Qt.Key_Q:
                self.state["yaw"] = -MAX_YAW_RATE
            elif key == QtCore.Qt.Key_E:
                self.state["yaw"] = MAX_YAW_RATE
            elif key == QtCore.Qt.Key_Space:          # arm & take‑off
                self.arm_takeoff()
            elif key == QtCore.Qt.Key_Escape:         # emergency stop
                self.disarm()

    def keyReleaseEvent(self, e):
        key = e.key()
        with self.state["lock"]:
            if key in (QtCore.Qt.Key_W, QtCore.Qt.Key_S,
                       QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
                self.state["pitch"] = 0
            elif key in (QtCore.Qt.Key_A, QtCore.Qt.Key_D,
                         QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
                self.state["roll"] = 0
            elif key in (QtCore.Qt.Key_Q, QtCore.Qt.Key_E):
                self.state["yaw"] = 0

    # -----------------------------------------------------------------------
    # Game‑pad polling
    # -----------------------------------------------------------------------
    def poll_gamepad(self):
        pygame.event.pump()
        if not self.joy:
            return
        lx = self.joy.get_axis(0)   # left stick X  (roll)
        ly = self.joy.get_axis(1)   # left stick Y  (pitch)
        rx = self.joy.get_axis(3)   # right stick X (yaw)
        rt = (self.joy.get_axis(5) + 1) / 2    # trigger 0‑>1 (thrust)

        b_start = self.joy.get_button(7)       # start
        b_b = self.joy.get_button(1)       # B – kill

        with self.state["lock"]:
            self.state["roll"] = int(lx * MAX_ROLL_PITCH)
            self.state["pitch"] = int(ly * MAX_ROLL_PITCH)
            self.state["yaw"] = int(rx * MAX_YAW_RATE)
            self.state["thrust"] = int(rt * MAX_THRUST)

        if b_start:
            self.arm_takeoff()
        if b_b:
            self.disarm()

    # -----------------------------------------------------------------------
    # Arm / take‑off / land helpers
    # -----------------------------------------------------------------------
    def arm_takeoff(self):
        with self.state["lock"]:
            if self.state["armed"]:
                return
            self.state["armed"] = True
            self.state["thrust"] = 30_000
        self.cf.commander.send_setpoint(0, 0, 0, 30_000)
        time.sleep(1.0)                    # small hop
        with self.state["lock"]:
            self.state["thrust"] = 32_000  # hover

    def disarm(self):
        with self.state["lock"]:
            self.state["armed"] = False
            self.state["thrust"] = 0
        self.cf.commander.send_setpoint(0, 0, 0, 0)


# ===============================================================================
# Main entry‑point
# ===============================================================================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(URI)
    w.show()
    sys.exit(app.exec())
