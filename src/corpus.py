
from __future__ import annotations

from functools import lru_cache
from pathlib import Path, PureWindowsPath

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed_hcmc2026"
NEO = "data"          # thư mục mốc để cắt: mọi đường dẫn đều đi qua <ROOT>/data/...


def _phan_duoi(p: str) -> str | None:
    parts = PureWindowsPath(str(p).replace("/", "\\")).parts
    for i in range(len(parts) - 1, -1, -1):        # lần xuất hiện CUỐI của "data"
        if parts[i].lower() == NEO:
            return str(Path(*parts[i:]))
    return None


def resolve_paths(s: pd.Series, root: Path | None = None) -> pd.Series:
    """Đưa một cột đường dẫn ảnh về gốc của máy hiện tại."""
    root = root or ROOT
    duoi = s.astype(str).map(_phan_duoi)
    hong = int(duoi.isna().sum())
    if hong:
        raise ValueError(
            f"{hong}/{len(s)} đường dẫn không chứa thư mục '{NEO}' nên không "
            f"ghép lại được — ví dụ: {s[duoi.isna()].iloc[0]!r}")
    return duoi.map(lambda d: str(root / d))


@lru_cache(maxsize=1)
def load_metadata() -> pd.DataFrame:
    d = pd.read_parquet(PROC / "metadata.parquet")
    d["image_path"] = resolve_paths(d["image_path"])
    return d


def kiem_tra(n: int = 200) -> tuple[int, int]:
    d = load_metadata()
    mau = d.image_path.iloc[:: max(1, len(d) // n)][:n]
    co = sum(Path(p).exists() for p in mau)
    return co, len(mau)


if __name__ == "__main__":
    d = load_metadata()
    co, tong = kiem_tra()
    print(f"{len(d):,} khung · {d.video_id.nunique()} video")
    print(f"gốc dự án : {ROOT}")
    print(f"ảnh mẫu   : {d.image_path.iloc[0]}")
    print(f"tồn tại   : {co}/{tong} ảnh lấy mẫu"
          + ("  ✅" if co == tong else "  ⚠️ thiếu keyframe trên máy này"))
