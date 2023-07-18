from __future__ import annotations
from typing import TYPE_CHECKING
import cv2

if TYPE_CHECKING:
    from .main import Calibrate


def init_cameras(self: Calibrate, device_ids: list[int]):
    self.cameras = []
    for device_id in device_ids:
        self.cameras.append(
            {
                "device_id": device_id,
                "feed": None,
                "frame": None,
                "axis": None,
                "axim": None,
                "height": 0,
                "width": 0,
                "charuco_data": [],
                "charuco_plot": None,
            }
        )

    print("Loading Cameras")
    for camera in self.cameras:
        camera["feed"] = cv2.VideoCapture(camera["device_id"], cv2.CAP_DSHOW)
        camera["feed"].set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
        camera["feed"].set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    booting = True
    while booting:
        booting = False
        for camera in self.cameras:
            if not camera["feed"].isOpened():
                booting = True
                break

    print("Cameras Loaded")
