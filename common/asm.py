import os
import re
import subprocess
import tempfile
from functools import cache
from pathlib import Path

_FILE = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]+)"', re.M)
_LOC = re.compile(r"^(\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)).*$", re.M)
_SASS_LOC = re.compile(r'^\s*//## File "([^"]+)", line (\d+)(?: inlined at "([^"]+)", line (\d+))?.*$', re.M)


@cache
def _source_lines(path):
    try:
        return Path(path).read_text().splitlines()
    except OSError:  # e.g. Inductor's temporary kernel files
        return []


def _where(path, line, col=None):
    """file:line[:col] followed by that line's source, when the file is readable."""
    lines = _source_lines(path)
    source = lines[line - 1].strip() if 0 < line <= len(lines) else ""
    position = f"{Path(path).name}:{line}" + (f":{col}" if col is not None else "")
    return f"{position}  {source}".rstrip()


def annotate_ptx(ptx):
    """Comments every .loc directive with file:line:col and that line's source."""
    files = dict(_FILE.findall(ptx))
    return _LOC.sub(lambda m: f"{m[1]}\t// {_where(files.get(m[2], '?'), int(m[3]), m[4])}", ptx)


def _annotate_sass(sass):
    """Rewrites nvdisasm's //## line markers the same way as annotate_ptx."""

    def comment(m):
        inlined = f"  (inlined at {_where(m[3], int(m[4]))})" if m[3] else ""
        return f"\t// {_where(m[1], int(m[2]))}{inlined}"

    return _SASS_LOC.sub(comment, sass)


def _tool(name, *args, cwd=None):
    tool = Path(os.environ["CUDA_HOME"]) / "bin" / name
    return subprocess.run([tool, *map(str, args)], capture_output=True, text=True, check=True, cwd=cwd).stdout


def _sass(cubin_path):
    # nvdisasm, unlike cuobjdump -sass, prints the source lines stored in the cubin.
    return _annotate_sass(_tool("nvdisasm", "--print-code", "--print-line-info-inline", cubin_path))


def cubin_asm(asm):
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(asm["cubin"])
        f.flush()
        return {"ptx": annotate_ptx(asm["ptx"]), "sass": _sass(f.name)}


def so_asm(path):
    with tempfile.TemporaryDirectory() as tmp:
        _tool("cuobjdump", "-xelf", "all", Path(path).resolve(), cwd=tmp)
        sass = "\n".join(_sass(cubin) for cubin in sorted(Path(tmp).glob("*.cubin")))
    return {"ptx": annotate_ptx(_tool("cuobjdump", "-ptx", path)), "sass": sass}


def dump_asm(asm, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, listing in asm.items():
        (out_dir / f"{name}.ptx").write_text(listing["ptx"])
        (out_dir / f"{name}.sass").write_text(listing["sass"])
