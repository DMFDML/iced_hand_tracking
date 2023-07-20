from typing import Any
from interfaces import Camera
import sys
import signal

class Calibrate:
    cameras: list[Camera] = []
    fig: Any = None
    xr_session: Any = None
    hmd_pos: Any = None
    input_tensors: Any = []
    output_tensors: Any = None
    fig3d: Any = None
    fig3d_ax: Any = None
    fig3d_scatter: Any = None

    def __init__(self) -> None:
        pass

    from _hello import hello
    from _init_cameras import init_cameras
    from _init_plot import init_plot
    from _init_xr import init_xr
    from _loop import loop
    from _stop import stop


if __name__ == "__main__":
    ids = sys.argv[-1].split(",")
    for i, id in enumerate(ids):
        ids[i] = int(id)
        
    c = Calibrate()
    c.hello()
    c.init_cameras(ids)
    c.init_xr()
    c.init_plot()

    signal.signal(signal.SIGINT, c.stop)

    c.loop(frame_rate=60, timeout_secs=60 * 6)
