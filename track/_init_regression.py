from __future__ import annotations
from typing import TYPE_CHECKING
import pickle
import os

if TYPE_CHECKING:
    from .main import HandTracker


def init_regression(self: HandTracker):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    all_cameras = dir_path+"/../tmp/all_cameras_model.p"
    cameras_12 = dir_path+"/../tmp/cameras_12.p"
    cameras_13 = dir_path+"/../tmp/cameras_13.p"
    cameras_23 = dir_path+"/../tmp/cameras_23.p"

    if not os.path.exists(all_cameras):
        exit("No All Camera Model")

    with open(all_cameras, "rb") as f:
        self.ml_all = pickle.load(f)

    if not os.path.exists(cameras_12):
        exit("No Camera 12 Model")
        
    with open(cameras_12, "rb") as f:
        self.ml_12 = pickle.load(f)

    if not os.path.exists(cameras_13):
        exit("No Camera 13 Model")
        
    with open(cameras_13, "rb") as f:
        self.ml_13 = pickle.load(f)

    if not os.path.exists(cameras_23):
        exit("No Camera 23 Model")
        
    with open(cameras_23, "rb") as f:
        self.ml_23 = pickle.load(f)
