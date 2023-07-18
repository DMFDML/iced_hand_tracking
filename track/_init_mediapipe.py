from __future__ import annotations
from typing import TYPE_CHECKING
import mediapipe as mp

if TYPE_CHECKING:
    from .main import HandTracker


def init_mediapipe(self: HandTracker, task_path: str):
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=task_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
    )

    self.recognizer = GestureRecognizer.create_from_options(options)
