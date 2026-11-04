import os
import subprocess
import tempfile
from pathlib import Path


def _cuobjdump(flag, path):
    cuobjdump = Path(os.environ["CUDA_HOME"]) / "bin" / "cuobjdump"
    return subprocess.run([cuobjdump, flag, str(path)], capture_output=True, text=True, check=True).stdout


def cubin_asm(asm):
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(asm["cubin"])
        f.flush()
        return {"ptx": asm["ptx"], "sass": _cuobjdump("-sass", f.name)}


def so_asm(path):
    return {"ptx": _cuobjdump("-ptx", path), "sass": _cuobjdump("-sass", path)}


def dump_asm(asm, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, listing in asm.items():
        (out_dir / f"{name}.ptx").write_text(listing["ptx"])
        (out_dir / f"{name}.sass").write_text(listing["sass"])
