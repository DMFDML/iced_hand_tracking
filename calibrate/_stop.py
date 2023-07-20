from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main import HandTracker


def stop(self: HandTracker, signum, frame):
    print("stopping")
    for cam in self.cameras:
        if cam["feed"].isOpened():
           cam["feed"].release()
