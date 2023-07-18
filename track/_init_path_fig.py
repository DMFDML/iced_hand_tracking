from __future__ import annotations
from typing import TYPE_CHECKING
import cv2
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from .main import HandTracker


def init_path_fig(self: HandTracker):
    # Set to interactive.
    plt.ion()

    self.path_fig3d = plt.figure()
    self.path_fig3d_ax = self.path_fig3d.add_subplot(projection="3d")
    self.path_fig3d_ax.view_init(elev=20.0, azim=-35, roll=0)
    self.path_fig3d_ax.set_xlim(self.x_min, self.x_max)
    self.path_fig3d_ax.set_xlabel("x")
    self.path_fig3d_ax.set_ylim(self.y_min, self.y_max)
    self.path_fig3d_ax.set_ylabel("y (is z)")
    self.path_fig3d_ax.set_zlim(self.z_min, self.z_max)
    self.path_fig3d_ax.set_zlabel("z (is y)")
    self.path_fig3d.canvas.flush_events()
