from __future__ import annotations
from typing import TYPE_CHECKING
import xr

if TYPE_CHECKING:
    from .main import Calibrate


def init_xr(self: Calibrate):
    instance = xr.InstanceObject(application_name="track_hmd")
    system = xr.SystemObject(instance)
    window = xr.GlfwWindow(system)
    self.xr_session = xr.SessionObject(system, graphics_binding=window.graphics_binding)
