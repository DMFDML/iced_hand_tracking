from typing import TypedDict, Any
from enum import Enum


class Camera(TypedDict):
    device_id: int
    feed: Any
    frame: Any
    axis: Any
    axim: Any
    height: int
    width: int
    charuco_data: Any
    charuco_plot: Any
