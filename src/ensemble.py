"""Cộng điểm của một encoder thứ hai trên TOÀN corpus.

Khác `rerank.encoder_scores` ở chỗ nó chấm cả 177.321 khung chứ không chỉ rổ
top-100, nên đổi được cả R@100. Cơ sở đo và cách chọn trọng số: xem
docs/bao_cao_he_thong.tex, mục "Ghép hai encoder trên toàn corpus".
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
KERNEL_OUT = ROOT / "data" / "kernel_out"

# Phải KHÁC HỌ với encoder chính, không chỉ khác tên: ghép hai bản cùng kiến trúc
# thì chúng sai giống nhau và không thêm được gì (đã đo với L-16-384).
PHU = ("ViT-B-16-SigLIP2-384", "webli")
PHU_FEATURES = KERNEL_OUT / "features_ViT-B-16-SigLIP2-384__webli.npy"

W_MAC_DINH = 0.5


def san_sang() -> bool:
    """Có đủ tệp để chạy ensemble không."""
    return PHU_FEATURES.exists()


@lru_cache(maxsize=1)
def _features() -> np.ndarray:
    """Vector ảnh của encoder phụ, đã đổi sang float32 và giữ trong RAM.

    ⚠️ Đừng đổi lại thành mmap float16 rồi ép kiểu ở mỗi lần tìm — 45,8 giây
    một truy vấn. Xem sổ bẫy trong báo cáo.
    """
    if not san_sang():
        raise FileNotFoundError(
            f"thiếu {PHU_FEATURES}.\nTải cùng chỗ với bộ vector chính "
            f"(xem README, mục 'Cài trên một máy mới'), hoặc tắt ensemble.")
    return np.load(PHU_FEATURES).astype(np.float32, copy=False)


@lru_cache(maxsize=1)
def _text_encoder():
    import open_clip
    import torch
    m, _, _ = open_clip.create_model_and_transforms(PHU[0], pretrained=PHU[1])
    m.eval()
    return m, open_clip.get_tokenizer(PHU[0]), torch


def diem_phu(query: str) -> np.ndarray:
    """Điểm của encoder phụ cho toàn corpus, cùng thứ tự hàng với metadata."""
    m, tk, torch = _text_encoder()
    with torch.no_grad():
        v = m.encode_text(tk([query])).float()
        v /= v.norm(dim=-1, keepdim=True)
    return _features() @ v.numpy()[0].astype(np.float32)


def zscore(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-9)


def ghep(diem_chinh: np.ndarray, query: str, w: float = W_MAC_DINH) -> np.ndarray:
    """z(encoder chính) + w·z(encoder phụ).

    `diem_chinh` phải là điểm của CẢ corpus — z-score trên rổ con cho thang khác
    hẳn, khi đó w không còn đúng nghĩa như lúc hiệu chỉnh.
    """
    p = diem_phu(query)
    if len(p) != len(diem_chinh):
        raise ValueError(
            f"encoder phụ có {len(p):,} khung nhưng điểm chính có "
            f"{len(diem_chinh):,} — hai bộ vector không cùng corpus")
    return zscore(diem_chinh) + w * zscore(p)
