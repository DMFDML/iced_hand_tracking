from __future__ import annotations
from typing import TYPE_CHECKING
import time
import cv2
import xr
from math import cos, sin, atan2, asin
import numpy as np
import os

if TYPE_CHECKING:
    from .main import Calibrate

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

test_board = {}
for i in range(0, 17):
    test_board[i] = {}
    for j in range(0, 4):
        test_board[i][j] = [1, 0, 0]


def loop(self: Calibrate, frame_rate: int = 10, timeout_secs: float = 5):
    last_handled_frame_timestamp: float = time.time()
    timeout_timestamp = time.time() + timeout_secs

    camerasAreOpen = True
    while camerasAreOpen:
        print("Time Left:", time.time() - timeout_timestamp)
        if time.time() > timeout_timestamp:
            break
        # check the cameras are still accessible
        camerasAreOpen = True
        for camera in self.cameras:
            if not camera["feed"].isOpened():
                camerasAreOpen = False
                break
        if not camerasAreOpen:
            print("Cameras are no longer accesible")
            continue

        # Update the frames
        for camera in self.cameras:
            success, frame = camera["feed"].read()
            if not success:
                continue
            camera["frame"] = frame

        # Check if it is a frame we would like to deal with
        time_elapsed = time.time() - last_handled_frame_timestamp
        if time_elapsed < 1.0 / frame_rate:
            continue

        last_handled_frame_timestamp = time.time()

        for camera in self.cameras:
            camera["frame"] = cv2.cvtColor(camera["frame"], cv2.COLOR_BGR2RGB)
            camera["axim"].set_data(camera["frame"])

        # Now process the frames and update camera information.
        for camera in self.cameras:
            gray_frame = cv2.cvtColor(camera["frame"], cv2.COLOR_BGR2GRAY)
            cells, ids, _ = detector.detectMarkers(gray_frame)

            camera["charuco_data"] = {}
            x = []
            y = []
            for i, cell in enumerate(cells):
                cell_id = ids[i][0]
                corners = cell[0]
                camera["charuco_data"][cell_id] = corners
                for _, corner in enumerate(corners):
                    x.append(float(corner[0]))
                    y.append(float(corner[1]))

            camera["charuco_plot"].set_data(x, y)

        self.fig.canvas.flush_events()

        if self.xr_session:
            self.xr_session.poll_xr_events()
            if self.xr_session.state in (
                xr.SessionState.READY,
                xr.SessionState.SYNCHRONIZED,
                xr.SessionState.VISIBLE,
                xr.SessionState.FOCUSED,
            ):
                self.xr_session.wait_frame()
                self.xr_session.begin_frame()
                _, views = self.xr_session.locate_views()
                pos = views[xr.Eye.LEFT.value].pose
                self.hmd_pos = np.asarray(
                    [pos.position.x, pos.position.y, pos.position.z]
                )

                self.xr_session.end_frame()
            else:
                self.hmd_pos = None
        else:
            self.hmd_pos = None

        # Then create the dataset of matched points.
        cell_id = 0
        corner_id = 0
        input_tensor = []
        any_data = False
        for camera in self.cameras:
            if cell_id in camera["charuco_data"]:
                any_data = True
                input_tensor.append(camera["charuco_data"][cell_id][corner_id, 0])
                input_tensor.append(camera["charuco_data"][cell_id][corner_id, 1])
            else:
                input_tensor.append(np.nan)
                input_tensor.append(np.nan)

        if any_data and self.hmd_pos is not None:
            print(input_tensor, self.hmd_pos)
            self.input_tensors.append(input_tensor)
            if self.output_tensors is not None:
                self.output_tensors = np.vstack((self.output_tensors, self.hmd_pos))
            else:
                self.output_tensors = np.asarray([self.hmd_pos])
                print(self.output_tensors)

            if self.fig3d and self.output_tensors.shape[0] % 20 == 0:
                self.fig3d_scatter.remove()
                self.fig3d_scatter = self.fig3d_ax.scatter(
                    self.output_tensors[:, 0],
                    self.output_tensors[:, 1],
                    self.output_tensors[:, 2],
                    c="b",
                    marker="o",
                )
                self.fig3d.canvas.flush_events()

    # Now store the data.
    print("Storing the data.")

    dir_path = os.path.dirname(os.path.realpath(__file__))

    if self.output_tensors is not None:
        if not os.path.isdir(dir_path + "/tmp"):
            os.mkdir(dir_path + "/tmp")
        np.savetxt(
            dir_path + "/../tmp/output_tensors.csv",
            self.output_tensors,
            delimiter=",",
            newline="\n",
        )
    if self.input_tensors:
        arr = np.asarray(self.input_tensors)
        np.savetxt(
            dir_path + "/../tmp/input_tensors.csv", arr, delimiter=",", newline="\n"
        )

    self.fig.clf()
    return
