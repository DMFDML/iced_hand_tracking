from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main import Calibrate


def hello(self: Calibrate):
    print("world")
