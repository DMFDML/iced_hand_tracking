from __future__ import annotations
from typing import TYPE_CHECKING
import cv2
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from .main import Calibrate


def init_plot(self: Calibrate):
    # Set to interactive.
    plt.ion()

    # Create the subplots.
    self.fig = plt.figure()
    n_plots = len(self.cameras)
    for i, camera in enumerate(self.cameras):
        camera["axis"] = self.fig.add_subplot(1, n_plots, i + 1)
        plt.axis("off")

    ## Init elements and take first frame.
    for camera in self.cameras:
        success, frame = camera["feed"].read()
        if not success:
            exit("Error")
        height, width, _ = frame.shape
        camera["width"] = width
        camera["height"] = height
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera["frame"] = frame

        camera["axim"] = camera["axis"].imshow(frame, vmin=0, vmax=255)

        (charuco,) = camera["axis"].plot([], [], "ob")
        camera["charuco_plot"] = charuco

    self.fig.canvas.flush_events()

    self.fig3d = plt.figure()
    self.fig3d_ax = self.fig3d.add_subplot(projection="3d")
    self.fig3d_ax.view_init(elev=20.0, azim=-35, roll=0)
    self.fig3d_ax.set_xlabel("x")
    self.fig3d_ax.set_ylabel("y")
    self.fig3d_ax.set_zlabel("z")
    self.fig3d_scatter = self.fig3d_ax.scatter3D([], [], [], c="b", marker="o")
    self.fig3d.canvas.flush_events()
