from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main import HandTracker


def stop(self: HandTracker, signum, frame):
    for cam in self.cameras:
        if cam["feed"].isOpened():
           cam["feed"].release()

    if self.fig:
        self.fig.clf()

    if self.hands_fig3d:
        self.hands_fig3d.clf()

    if self.path_fig3d:
        self.path_fig3d.clf()