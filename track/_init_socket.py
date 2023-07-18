from __future__ import annotations
from typing import TYPE_CHECKING
import socket

if TYPE_CHECKING:
    from .main import HandTracker


def init_socket(self: HandTracker, ip: str = "127.0.0.1", port: int = 5065):
    self.sock_ip = ip
    self.sock_port = port
    print("world")
