from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main import HandTracker


def hello(self: HandTracker):
    print("world")
