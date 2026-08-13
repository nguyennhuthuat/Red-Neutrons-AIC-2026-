"""
shrink_keyframes.py — thu nhỏ 177k keyframe rồi đóng gói để upload lên Kaggle.

VÌ SAO PHẢI THU NHỎ
    Bộ keyframe gốc nặng 28,66 GB (177.321 ảnh, trung bình 170 KB). Upload chừng
    đó lên Kaggle là tự chuốc lấy đứt kết nối — lần upload 8,39 GB audio đã dính
    SSLEOFError giữa chừng. Mà CLIP dù sao cũng resize ảnh về 224px trước khi
    nhìn, nên phần lớn 28 GB kia bị vứt đi ngay ở bước tiền xử lý.

ĐÃ ĐO CHỨ KHÔNG ĐOÁN (script scratchpad/shrink_metric.py, 81 query, rổ top-100)
    B-16 từ ảnh GỐC                FINAL 0.5111
    B-16 từ ảnh THU NHỎ 384px q92  FINAL 0.5111   ← giống hệt
    cosine giữa hai bộ vector: 0.9957

    Bài học: cosine 0.9957 nghe như mất mát đáng kể nhưng ĐIỂM KHÔNG ĐỔI. Cosine
    là chỉ số sai để quyết định việc này — phải đo thẳng cái mình cần.

VÌ SAO 384px CHỨ KHÔNG PHẢI 224px
    Thử cả hai. Làm sẵn đúng phép tiền xử lý của CLIP (224px + cắt giữa) lại cho
    cosine TỆ HƠN (0.9885) so với giữ 384px (0.9964). Lý do: nén JPEG ở đúng độ
    phân giải model nhìn thì vết nén khối 8x8 nằm nguyên trong ảnh; còn giữ 384px
    thì bước thu nhỏ về 224 tự bình quân hoá vết nén đi. **Nén rồi mới thu nhỏ
    luôn tốt hơn thu nhỏ rồi mới nén.**
    Ngoài ra 384px giữ cửa mở cho SigLIP-384 và ViT-L-14-336.

VÌ SAO BICUBIC
    Đó là bộ lọc CLIP dùng (Resize(size=224, interpolation=bicubic)). Dùng LANCZOS
    thì thành hai bộ lọc khác nhau chồng lên nhau.

Chạy:
    python src/shrink_keyframes.py            # xem trước, không ghi gì
    python src/shrink_keyframes.py --apply    # thu nhỏ + đóng gói
"""

import argparse
import io
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "hcmc2026"
OUT = ROOT / "data" / "keyframes_kaggle_upload"

SHORT = 384        # cạnh ngắn, px
QUALITY = 92       # JPEG
N_ZIP = 12         # số gói; ~1 GB/gói, cỡ đã upload trót lọt ở đợt audio


def shrink_bytes(path: Path) -> bytes:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = SHORT / min(w, h)
    if s < 1:
        im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
    buf = io.BytesIO()
    # subsampling=0 = không lấy mẫu con kênh màu. Giữ màu sắc nguyên vẹn, mà
    # màu là chi tiết query hay nhắc tới nhất ("áo đỏ", "bảng xanh").
    im.save(buf, format="JPEG", quality=QUALITY, optimize=True, subsampling=0)
    return buf.getvalue()


def find_videos() -> dict[str, Path]:
    """video_id -> thư mục ảnh. L26 bị BTC chia thành Keyframes_L26_a..e nên
    không thể duyệt theo tên gói; khớp theo '_V' trong tên thư mục con."""
    out = {}
    for pack in sorted(RAW.glob("Keyframes_*")):
        for d in pack.rglob("*"):
            if d.is_dir() and "_V" in d.name:
                out[d.name] = d
    return out


def do_video(args):
    vid, src, dst_root = args
    dst = Path(dst_root) / vid
    dst.mkdir(parents=True, exist_ok=True)
    n = tot = 0
    for p in sorted(src.glob("*.jpg")):
        target = dst / p.name
        if target.exists():                    # chạy lại thì bỏ qua ảnh đã xong
            tot += target.stat().st_size
            n += 1
            continue
        b = shrink_bytes(p)
        target.write_bytes(b)
        tot += len(b)
        n += 1
    return vid, n, tot


def make_zips(src_root: Path, n_zip: int):
    """Đóng gói CÂN BẰNG theo dung lượng, dùng ZIP_STORED (không nén lại).

    Hai điều đã học ở đợt upload audio:
      - `kaggle datasets create --dir-mode zip` CHỈ nén thư mục con, file lẻ vẫn
        upload từng cái một -> 177k lần gọi API, chắc chắn đứt. Phải tự nén.
      - JPEG đã nén rồi, nén lại bằng deflate chỉ tốn CPU mà không giảm được gì.
    """
    vids = sorted(d for d in src_root.iterdir() if d.is_dir())
    sizes = [(d, sum(f.stat().st_size for f in d.glob("*.jpg"))) for d in vids]
    sizes.sort(key=lambda x: -x[1])

    buckets: list[list[Path]] = [[] for _ in range(n_zip)]
    loads = [0] * n_zip
    for d, s in sizes:                          # tham lam: bỏ vào gói nhẹ nhất
        i = loads.index(min(loads))
        buckets[i].append(d)
        loads[i] += s

    for i, (bucket, load) in enumerate(zip(buckets, loads), 1):
        zp = src_root / f"keyframes_{i:02d}.zip"
        # Ghi ra .part rồi mới đổi tên. Nếu bỏ bước này thì lần chạy bị ngắt
        # giữa chừng để lại một file .zip cụt, và vòng lặp "đã có thì bỏ qua"
        # bên dưới sẽ coi nó là xong -> upload archive hỏng lên Kaggle mà không
        # có lấy một dòng cảnh báo. ĐÃ DÍNH THẬT một lần.
        if zp.exists():
            print(f"  {zp.name} đã có, bỏ qua")
            continue
        tmp = zp.with_suffix(".part")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
            for d in bucket:
                for f in sorted(d.glob("*.jpg")):
                    z.write(f, arcname=f"{d.name}/{f.name}")
        tmp.replace(zp)
        print(f"  {zp.name}: {len(bucket)} video · {load/1024**3:.2f} GB", flush=True)

    # Kiểm tra cuối: mở lại từng gói và đếm. Upload xong mới phát hiện hỏng thì
    # mất cả tiếng đồng hồ đường truyền.
    print("\nkiểm tra lại các gói ...")
    total = 0
    for zp in sorted(src_root.glob("*.zip")):
        try:
            with zipfile.ZipFile(zp) as z:
                n = len(z.namelist())
            total += n
            print(f"  {zp.name}: {n:,} file · {zp.stat().st_size/1024**3:.2f} GB · ok")
        except Exception as e:
            raise SystemExit(f"❌ {zp.name} HỎNG ({type(e).__name__}) — xoá file này "
                             f"rồi chạy lại lệnh")
    return total


def make_side_files(dst: Path, n_probe: int = 500):
    """Gói phụ vài MB: metadata + query + MẪU ĐỐI CHỨNG để kiểm tra căn hàng.

    Dùng khi lấy keyframe từ dataset công khai của người khác (vd nguynnc/aic2025)
    thay vì tự upload. Lúc đó câu hỏi sống còn là: ảnh của họ có ĐÚNG là ảnh của
    mình không? Nếu họ trích keyframe theo cách khác thì số thứ tự `n` lệch, và
    hệ thống sẽ trả về ảnh sai mà mọi thứ trông vẫn bình thường.

    Cách chứng minh: lấy `n_probe` khung ngẫu nhiên kèm vector B-32 do BTC cấp.
    Trên Kaggle, mã hoá đúng những khung đó bằng ViT-B-32-quickgelu rồi so cosine.
    Trùng ảnh thì cosine ~0.999 (đã đo mốc này ở [data-hcmc2026]); lệch ảnh thì
    rơi xuống ~0.2-0.5 ngay lập tức.
    """
    import numpy as np
    meta = pd.read_parquet(ROOT / "data" / "processed_hcmc2026" / "metadata.parquet")
    feats = np.load(ROOT / "data" / "processed_hcmc2026" / "features.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(meta), size=min(n_probe, len(meta)), replace=False))

    dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst / "probe_b32.npz",
        video_id=meta.video_id.to_numpy()[idx].astype("U16"),
        n=meta.n.to_numpy()[idx].astype(np.int32),
        feat=np.asarray(feats[idx], dtype=np.float32),
    )
    import shutil
    for src in (ROOT / "data" / "processed_hcmc2026" / "metadata.parquet",
                ROOT / "eval" / "queries_hcmc2026.csv"):
        shutil.copy2(src, dst / src.name)

    tot = sum(p.stat().st_size for p in dst.iterdir() if p.is_file())
    print(f"gói phụ -> {dst} · {tot/1024**2:.1f} MB · {len(idx)} khung đối chứng")
    print("  upload: kaggle datasets create -p . -u   (vài giây)")


if __name__ == "__main__":
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--side-only", action="store_true",
                    help="chỉ tạo gói phụ (metadata + query + mẫu đối chứng), "
                         "dùng khi lấy keyframe từ dataset công khai")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--zips", type=int, default=N_ZIP)
    a = ap.parse_args()

    if a.side_only:
        make_side_files(ROOT / "data" / "kaggle_side_files")
        raise SystemExit

    vids = find_videos()
    n_img = sum(len(list(d.glob("*.jpg"))) for d in vids.values())
    raw_gb = sum(f.stat().st_size for d in vids.values() for f in d.glob("*.jpg")) / 1024**3
    print(f"{len(vids)} video · {n_img:,} ảnh · {raw_gb:.2f} GB gốc")
    if len(vids) != 873:
        print(f"⚠ mong đợi 873 video, thấy {len(vids)} — bộ Keyframes có thể tải dở")

    if not a.apply:
        # Ước lượng bằng cách thu nhỏ thật 60 ảnh rải đều, không dùng con số phỏng đoán.
        import random
        random.seed(0)
        samp = random.sample([p for d in list(vids.values())[::13]
                              for p in list(d.glob("*.jpg"))[:3]], 60)
        kb = sum(len(shrink_bytes(p)) for p in samp) / len(samp) / 1024
        print(f"\nthu nhỏ cạnh ngắn {SHORT}px q{QUALITY} -> ~{kb:.0f} KB/ảnh "
              f"= ~{kb*n_img/1024**2:.2f} GB ({raw_gb/(kb*n_img/1024**2):.1f}x nhỏ hơn)")
        print(f"chia {a.zips} gói -> ~{kb*n_img/1024**2/a.zips:.2f} GB/gói")
        print("\n(xem trước — thêm --apply để chạy thật)")
        raise SystemExit

    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(v, d, str(OUT)) for v, d in sorted(vids.items())]
    t0 = time.time()
    done = tot = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (vid, n, sz) in enumerate(ex.map(do_video, jobs, chunksize=4), 1):
            done += n
            tot += sz
            if i % 50 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(jobs)} video · {done:,} ảnh · {tot/1024**3:.2f} GB · "
                      f"{done/el:.0f} ảnh/s · còn ~{(n_img-done)/max(done/el,1e-9)/60:.0f} phút",
                      flush=True)
    print(f"\nthu nhỏ xong {done:,} ảnh · {tot/1024**3:.2f} GB · {(time.time()-t0)/60:.1f} phút")
    assert done == n_img, f"thiếu ảnh: {done:,} vs {n_img:,}"

    # Kèm ba file nhỏ để notebook trên Kaggle tự lo được mọi thứ:
    #   metadata.parquet  -> thứ tự hàng chuẩn (sorted video_id, rồi n tăng dần)
    #   queries_*.csv     -> 81 query, chấm Recall ngay tại chỗ khỏi tải về mới biết
    #   probe_b32.npz     -> 500 khung đối chứng để kiểm tra căn hàng
    make_side_files(OUT)

    print(f"\nđóng {a.zips} gói ...")
    packed = make_zips(OUT, a.zips)
    assert packed == n_img, f"gói chứa {packed:,} ảnh nhưng corpus có {n_img:,}"
    print(f"✅ {packed:,} ảnh trong {a.zips} gói, khớp corpus")

    print(f"\nxong. Upload bằng:\n"
          f"  kaggle datasets create -p {OUT} -u\n"
          f"(nhớ tạo dataset-metadata.json trong thư mục đó trước)\n"
          f"\n⚠ ĐỂ DATASET Ở CHẾ ĐỘ PRIVATE. Gói này chứa keyframe của BTC và\n"
          f"  queries_hcmc2026.csv — bộ eval tự annotate của đội. KHÔNG thêm cờ\n"
          f"  --public (mặc định của Kaggle là private, cứ để nguyên).")
