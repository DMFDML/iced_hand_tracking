from __future__ import annotations
from typing import TYPE_CHECKING
import cv2
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from .main import HandTracker


def init_fig(self: HandTracker):
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
        if not camera["feed"]:
            exit("No Camera Feed")
        success, frame = camera["feed"].read()
        if not success:
            exit("Error")
        height, width, _ = frame.shape
        camera["width"] = width
        camera["height"] = height
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera["frame"] = frame

        if camera["axis"]:
            camera["axim"] = camera["axis"].imshow(frame, vmin=0, vmax=255)
            (right,) = camera["axis"].plot([], [], "oy")
            (left,) = camera["axis"].plot([], [], "or")
            camera["right"]["plot"] = right
            camera["left"]["plot"] = left

    self.fig.canvas.flush_events()
