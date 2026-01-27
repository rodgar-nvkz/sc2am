import subprocess
import time
from pathlib import Path

import websocket
from portpicker import pick_unused_port

GAME_FOLDER = Path("~/StarCraftII/").expanduser()
GAME_ENTRYPOINT = Path("Versions/Base75689/SC2_x64")
LOCALHOST = "127.0.0.1"


class SC2LocalhostServer:
    def __init__(self) -> None:
        self.host = LOCALHOST
        self.port = pick_unused_port()
        self.server = self.run_instance()
        self.socket = self.connect()

    def run_instance(self):
        path = str(GAME_FOLDER / GAME_ENTRYPOINT)
        args = path, f"-listen={self.host}", f"-port={self.port}", "-displayMode=0"
        return subprocess.Popen(args, stderr=subprocess.DEVNULL, encoding="utf8")

    def connect(self, timeout: int = 45):
        start = time.time()
        path = f"ws://{self.host}:{self.port}/sc2api"
        while start + timeout > time.time():
            try:
                return websocket.create_connection(path, timeout=timeout)
            except (websocket.WebSocketException, ConnectionRefusedError, OSError):
                time.sleep(0.5)
        raise TimeoutError(f"Could not connect to SC2 instance at {path}")

    def close(self):
        self.socket.close()
        self.server.kill()
        self.server.wait()
