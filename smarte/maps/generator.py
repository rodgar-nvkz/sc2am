import random

import mpyq

from smarte.settings import PROJECT_ROOT


class MapGenerator:
    """Maps generation and (future) manipulations"""

    TEMPLATES = ("Flat64c.SC2Map",)

    def __init__(self, folder: str = "maps") -> None:
        self.folder = PROJECT_ROOT / folder

    def __next__(self) -> tuple[str, bytes]:
        """We will stick with one hand crafted arcade.SC2Map for now"""
        template = self.folder / random.choice(self.TEMPLATES)
        mpyq.MPQArchive(template)  # Validate MPQ
        with open(template, "rb") as f:
            return template.name, f.read()

    def unpack(self):
        """Unpack the map archive content into memory"""

    def pack(self):
        """Pack the map archive content back into MPQ format"""
