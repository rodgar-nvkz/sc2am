import random

import mpyq

from scam.settings import PROJECT_ROOT


class MapGenerator:
    """A placeholder for maps generation and manipulation"""

    TEMPLATES = ("Flat32.SC2Map",)

    def __init__(self, folder: str = "maps") -> None:
        self.folder = PROJECT_ROOT / folder

    def __next__(self) -> bytes:
        return self.generate()

    def generate(self) -> bytes:
        """We will stick with one hand crafted arcade.SC2Map for now"""
        template = self.folder / random.choice(self.TEMPLATES)
        mpyq.MPQArchive(template)  # Validate MPQ
        with open(template, "rb") as f:
            return f.read()

    def unpack(self):
        """Unpack the map archive content into memory"""

    def pack(self):
        """Pack the map archive content back into MPQ format"""
