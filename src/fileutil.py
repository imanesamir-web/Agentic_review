"""Small file helpers."""
import os
import time


def _can_write(path: str) -> bool:
    try:
        with open(path, "a"):
            return True
    except PermissionError:
        return False


def safe_path(path: str) -> str:
    """`path` itself, or a timestamped sibling when the file is locked (e.g. open in Excel or Word)."""
    if _can_write(path):
        return path
    base, ext = os.path.splitext(path)
    alt = f"{base}_{time.strftime('%H%M%S')}{ext}"
    print(f"[warn] {path} is locked (open in Excel/Word?). Writing {alt} instead.", flush=True)
    return alt
