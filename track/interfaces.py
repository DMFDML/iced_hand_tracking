from typing import TypedDict, Any
from enum import Enum
from matplotlib.image import AxesImage
from matplotlib.pyplot import Axes


class Gesture(str, Enum):
    NONE = "none"
    INDEX_PINCH = "index_pinch"
    MIDDLE_PINCH = "middle_pinch"
    RING_PINCH = "ring_pinch"
    PINKY_PINCH = "pinky_pinch"
    CLOSED_FIST = "closed_fist"
    OPEN_PALM = "open_palm"
    POINTING_UP = "pointing_up"
    THUMB_DOWN = "thumb_down"
    THUMB_UP = "thumb_up"
    VICTORY = "victory"
    I_LOVE_YOU = "i_love_you"


class Hand(TypedDict):
    plot: Any
    landmarks: Any
    gesture: Gesture


class Camera(TypedDict):
    device_id: int
    feed: Any
    frame: Any
    axis: None | Axes
    axim: None | AxesImage
    height: int
    width: int
    right: Hand
    left: Hand
