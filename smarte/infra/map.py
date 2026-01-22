"""
SC2 Map manipulation utilities using StormLib.

Provides functions to:
- Extract SC2Map (MPQ archive) to directory
- Repack directory to SC2Map
- Hide UI chat element
"""

import ctypes
import os
from ctypes import POINTER, Structure, byref, c_bool, c_char_p, c_uint32, c_void_p

import mpyq

# StormLib constants
MPQ_FILE_COMPRESS = 0x00000200
MPQ_FILE_REPLACEEXISTING = 0x80000000
MPQ_COMPRESSION_ZLIB = 0x02
BASE_PROVIDER_FILE = 0x00000000
STREAM_PROVIDER_FLAT = 0x00000000


class SFILE_CREATE_MPQ(Structure):
    """StormLib archive creation parameters."""

    _fields_ = [
        ("cbSize", c_uint32),
        ("dwMpqVersion", c_uint32),
        ("pvUserData", c_void_p),
        ("cbUserData", c_uint32),
        ("dwStreamFlags", c_uint32),
        ("dwFileFlags1", c_uint32),  # (listfile) flags
        ("dwFileFlags2", c_uint32),  # (attributes) flags
        ("dwFileFlags3", c_uint32),  # (signature) flags
        ("dwAttrFlags", c_uint32),
        ("dwSectorSize", c_uint32),
        ("dwRawChunkSize", c_uint32),
        ("dwMaxFileCount", c_uint32),
    ]


def get_stormlib():
    """Load StormLib shared library."""
    storm = ctypes.CDLL("/usr/local/lib/libstorm.so.9")

    storm.SFileCreateArchive2.argtypes = [c_char_p, POINTER(SFILE_CREATE_MPQ), POINTER(c_void_p)]
    storm.SFileCreateArchive2.restype = c_bool

    storm.SFileAddFileEx.argtypes = [c_void_p, c_char_p, c_char_p, c_uint32, c_uint32, c_uint32]
    storm.SFileAddFileEx.restype = c_bool

    storm.SFileCloseArchive.argtypes = [c_void_p]
    storm.SFileCloseArchive.restype = c_bool

    return storm


def extract_map(map_path: str, out_dir: str) -> list[str]:
    """Extract SC2Map to directory. Returns list of extracted files."""
    os.makedirs(out_dir, exist_ok=True)
    archive = mpyq.MPQArchive(map_path)
    assert archive.files

    extracted = []
    for f in archive.files:
        fname = f.decode("utf-8") if isinstance(f, bytes) else f
        content = archive.read_file(f)
        if content:
            # Normalize path separators
            fname_norm = fname.replace("\\", os.sep)
            fpath = os.path.join(out_dir, fname_norm)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, "wb") as out:
                out.write(content)
            extracted.append(fname)

    return extracted


def pack_map(src_dir: str, dst_mpq: str) -> bool:
    """Pack directory into SC2Map (MPQ archive)."""
    storm = get_stormlib()

    # Collect all files
    files = []
    for root, _, filenames in os.walk(src_dir):
        for fname in filenames:
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, src_dir)
            # Convert to MPQ path format (backslashes)
            mpq_path = rel_path.replace(os.sep, "\\")
            files.append((full_path, mpq_path))

    print(f"Packing {len(files)} files to {dst_mpq}")

    if os.path.exists(dst_mpq):
        os.remove(dst_mpq)

    # Create archive with unencrypted internal files (listfile, attributes)
    # so mpyq can read the result without encryption support
    create_info = SFILE_CREATE_MPQ()
    create_info.cbSize = ctypes.sizeof(SFILE_CREATE_MPQ)
    create_info.dwMpqVersion = 0  # MPQ v1
    create_info.pvUserData = None
    create_info.cbUserData = 0
    create_info.dwStreamFlags = BASE_PROVIDER_FILE | STREAM_PROVIDER_FLAT
    create_info.dwFileFlags1 = MPQ_FILE_COMPRESS  # (listfile): compressed, NOT encrypted
    create_info.dwFileFlags2 = MPQ_FILE_COMPRESS  # (attributes): compressed, NOT encrypted
    create_info.dwFileFlags3 = 0  # no (signature)
    create_info.dwAttrFlags = 0
    create_info.dwSectorSize = 0x10000  # 64KB sectors
    create_info.dwRawChunkSize = 0
    create_info.dwMaxFileCount = len(files) + 10

    hMpq = c_void_p()
    result = storm.SFileCreateArchive2(dst_mpq.encode("utf-8"), byref(create_info), byref(hMpq))

    if not result or not hMpq.value:
        print("Failed to create archive")
        return False

    success = True
    for full_path, mpq_path in files:
        result = storm.SFileAddFileEx(
            hMpq,
            full_path.encode("utf-8"),
            mpq_path.encode("utf-8"),
            MPQ_FILE_COMPRESS | MPQ_FILE_REPLACEEXISTING,
            MPQ_COMPRESSION_ZLIB,
            MPQ_COMPRESSION_ZLIB,
        )
        if not result:
            print(f"  FAILED: {mpq_path}")
            success = False

    storm.SFileCloseArchive(hMpq)
    return success


def read_galaxy_script(map_path: str) -> str:
    """Read MapScript.galaxy from SC2Map."""
    archive = mpyq.MPQArchive(map_path)
    content = archive.read_file("MapScript.galaxy")
    return content.decode("utf-8") if content else ""


CHAT_HIDE_LAYOUT = """<?xml version="1.0" encoding="utf-8" standalone="yes"?>
<Desc>
    <Frame type="GameUI" name="GameUI" file="GameUI">
        <Frame type="Frame" name="UIContainer">
            <Frame type="Frame" name="FullscreenUpperContainer">
                <Frame type="MessageDisplay" name="ChatDisplay">
                    <Visible val="false"/>
                </Frame>
                <Frame type="MessageDisplay" name="ReplayChatDisplay">
                    <Visible val="false"/>
                </Frame>
            </Frame>
        </Frame>
    </Frame>
</Desc>
"""

DESC_INDEX_LAYOUT = """<?xml version="1.0" encoding="utf-8" standalone="yes"?>
<Desc>
    <Include path="UI/Layout/ChatOverride.SC2Layout"/>
</Desc>
"""


def patch_ui(map_path: str, output_path: str) -> bool:
    """
    Patch an SC2Map to hide the in-game chat display.

    Injects a UI layout override that hides ChatDisplay and ReplayChatDisplay,
    and registers it in ComponentList.SC2Components.
    """
    import tempfile

    assert map_path != output_path

    with tempfile.TemporaryDirectory() as tmp_dir:
        extract_map(map_path, tmp_dir)

        # Create layout directory
        layout_dir = os.path.join(tmp_dir, "Base.SC2Data", "UI", "Layout")
        os.makedirs(layout_dir, exist_ok=True)

        # Write layout files
        with open(os.path.join(layout_dir, "ChatOverride.SC2Layout"), "w", encoding="utf-8") as f:
            f.write(CHAT_HIDE_LAYOUT)

        with open(os.path.join(layout_dir, "DescIndex.SC2Layout"), "w", encoding="utf-8") as f:
            f.write(DESC_INDEX_LAYOUT)

        # Register layout in ComponentList
        comp_path = os.path.join(tmp_dir, "ComponentList.SC2Components")
        if os.path.exists(comp_path):
            with open(comp_path, encoding="utf-8") as f:
                comp = f.read()
            if "uiui" not in comp:
                comp = comp.replace(
                    "</Components>",
                    '    <DataComponent Type="uiui">UI/Layout/DescIndex.SC2Layout</DataComponent>\n</Components>',
                )
                with open(comp_path, "w", encoding="utf-8") as f:
                    f.write(comp)

        return pack_map(tmp_dir, output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python map.py extract <map.SC2Map> <output_dir>")
        print("  python map.py pack <input_dir> <output.SC2Map>")
        print("  python map.py read <map.SC2Map>  # print MapScript.galaxy")
        print("  python map.py inject <map.SC2Map> [output.SC2Map]  # add map command triggers")
        print("  python map.py ui <map.SC2Map> [output.SC2Map]     # hide chat display")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "extract":
        files = extract_map(sys.argv[2], sys.argv[3])
        print(f"Extracted {len(files)} files")

    elif cmd == "pack":
        success = pack_map(sys.argv[2], sys.argv[3])
        print("Success" if success else "Failed")

    elif cmd == "read":
        print(read_galaxy_script(sys.argv[2]))

    elif cmd == "ui":
        success = patch_ui(sys.argv[2], sys.argv[3])
        print("Success" if success else "Failed")

    else:
        print(f"Unknown command: {cmd}")
