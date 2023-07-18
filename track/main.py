from typing import Any
from interfaces import Camera
import os
from matplotlib.figure import Figure
import socket
import numpy as np


class HandTracker:
    cameras: list[Camera] = []
    recognizer: Any = None
    ml_all: Any = None
    ml_12: Any = None
    ml_23: Any = None
    ml_13: Any = None
    fig: None | Figure = None
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_ip: str | None = None
    sock_port: int | None = None

    path_fig3d: None | Figure = None
    path_fig3d_ax: Any = None
    path_fig3d_scatter_left: Any = None
    path_fig3d_scatter_right: Any = None
    path_left: Any = np.array([0.0, 0.0, 0.0], ndmin=2)
    path_right: Any = np.array([0.0, 0.0, 0.0], ndmin=2)

    hands_fig3d: None | Figure = None
    hands_fig3d_ax: Any = None

    x_min: float = 0.0
    x_max: float = 5.0
    y_min: float = 0.0
    y_max: float = 5.0
    z_min: float = 0.0
    z_max: float = 5.0

    def __init__(self) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        tensor_file = dir_path + "/..tmp/input_tensors.csv"

        if os.path.exists(tensor_file):
            m = np.genfromtxt(tensor_file, delimiter=",")
            self.x_min = m[:, 0].min()
            self.x_max = m[:, 0].max()
            self.y_min = m[:, 1].min()
            self.y_max = m[:, 1].max()
            self.z_min = m[:, 2].min()
            self.z_max = m[:, 2].max()
        else:
            print("Warning: Default Plotting Boundaries Used.")

    from _hello import hello
    from _init_cameras import init_cameras
    from _init_fig import init_fig
    from _init_path_fig import init_path_fig
    from _init_hands_fig import init_hands_fig
    from _init_mediapipe import init_mediapipe
    from _init_regression import init_regression
    from _loop import loop


if __name__ == "__main__":
    tracker = HandTracker()
    tracker.init_cameras([1, 2, 3])
    tracker.init_fig()
    tracker.init_path_fig()
    tracker.init_hands_fig()

    dir_path = os.path.dirname(os.path.realpath(__file__))

    task_file = dir_path + "/../tasks/gesture_recognizer.task"
    if not os.path.exists(task_file):
        exit("Task File Not Here")

    tracker.init_mediapipe(task_file)

    tracker.init_regression()

    tracker.loop(10)
