
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


def khung_that(pts_time: float, fps: float) -> int:
    """Khung THẬT trong video, để soi bằng trình phát.

    Cột frame_idx của ban tổ chức là làm SÀN của pts*fps, nên 12,9% khung
    (22.922/177.321) thiếu đúng 1 so với khung thật. Đo trên 40 khung bất đồng:
    round(pts*fps) khớp ảnh 39/40, frame_idx khớp 0/40.

    KHÔNG dùng số này để nộp bài — bài nộp phải giữ nguyên frame_idx của ban
    tổ chức. Số này chỉ để người thi mở video ra kiểm bằng mắt.
    """
    return int(round(float(pts_time) * float(fps)))


def moc_gio(pts_time: float) -> str:
    """pts_time thành mm:ss.mmm để dán thẳng vào ô tua của trình phát."""
    t = max(0.0, float(pts_time))
    return f"{int(t // 60):02d}:{t % 60:06.3f}"


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
