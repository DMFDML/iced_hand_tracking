from __future__ import annotations
from typing import TYPE_CHECKING
import cv2
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from .main import HandTracker


def init_hands_fig(
    self: HandTracker,
):
    # Set to interactive.
    plt.ion()

    self.hands_fig3d = plt.figure()
    self.hands_fig3d_ax = self.hands_fig3d.add_subplot(projection="3d")
    self.hands_fig3d_ax.view_init(elev=20.0, azim=-35, roll=0)
    self.hands_fig3d_ax.set_xlim(self.x_min, self.x_max)
    self.hands_fig3d_ax.set_xlabel("x")
    self.hands_fig3d_ax.set_ylim(self.y_min, self.y_max)
    self.hands_fig3d_ax.set_ylabel("y (is z)")
    self.hands_fig3d_ax.set_zlim(self.z_min, self.z_max)
    self.hands_fig3d_ax.set_zlabel("z (is y)")
    self.hands_fig3d.canvas.flush_events()
