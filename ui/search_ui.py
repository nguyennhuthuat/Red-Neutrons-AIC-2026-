
from pathlib import Path
import csv
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("VLM_CHO_429", "0")

import corpus  # noqa: E402  (nạp metadata + sửa đường dẫn ảnh cho đúng máy)
import ensemble  # noqa: E402  (encoder phụ, cộng điểm trên toàn corpus)
import rerank  # noqa: E402  (tầng re-rank, xem src/rerank.py)
import nopbai  # noqa: E402  (đóng gói .zip đúng cấu trúc thể lệ)

DATASET = "hcmc2026"
PROCESSED = ROOT / "data" / f"processed_{DATASET}"
with open(PROCESSED / "manifest.json", encoding = "utf-8") as fp: 
    MANIFEST = json.load(fp)
CLIP_MODEL = MANIFEST["clip_model"]
CLIP_PRETRAINED = MANIFEST["clip_pretrained"]
# Điểm đã đo của từng bộ vector, hiện thẳng lên màn hình để biết đang chạy bộ nào.
BASELINES = {
    "ViT-B-32-quickgelu": 0.4765,          # bộ BTC cấp sẵn
    "ViT-L-14": 0.5580,
    "ViT-gopt-16-SigLIP2-384": 0.7457,
    "ViT-B-16-SigLIP2-384": 0.7556,
    "ViT-SO400M-16-SigLIP2-384": 0.7704,
    "ViT-L-16-SigLIP2-384": 0.7802,
    "ViT-L-16-SigLIP2-512": 0.7901,
}
BASELINE = BASELINES.get(CLIP_MODEL, float("nan"))

st.set_page_config(page_title="RED-NEUTRONS · AIC 2026", layout="wide")


@st.cache_data(show_spinner="Loading metadata ...")
def load_metadata() -> pd.DataFrame:
    # Qua corpus.load_metadata chứ không đọc thẳng parquet — nó sửa đường dẫn ảnh.
    return corpus.load_metadata()


@st.cache_resource(show_spinner="Loading FAISS index ...")
def load_index():
    import faiss
    return faiss.read_index(str(PROCESSED / "faiss.index"))


@st.cache_resource(show_spinner="Loading CLIP text encoder ...")
def load_clip():
    import torch
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.eval()
    return model, tokenizer, torch


def preprocess_query(query: str, mode: str = "vi") -> tuple[str, str | None]:
    query = (query or "").strip()
    if mode != "google" or not query:
        return query, None
    import translate as tr
    en = tr.to_english(query)
    if en is None:
        return query, "⚠ dịch hỏng — đang dùng câu tiếng Việt gốc (vẫn chạy được)"
    return en, f"→ dịch: *{en}*"


def encode_text(query: str) -> np.ndarray:
    model, tokenizer, torch = load_clip()
    with torch.no_grad():
        tokens = tokenizer([query])
        vec = model.encode_text(tokens).float().numpy()
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return vec.astype(np.float32)


def search(query: str, top_k: int, ens_w: float = 0.0,
           mark_rows: list[int] | None = None, beta: float = 0.4,
           thuong_video: float = 3.0) -> pd.DataFrame:
    meta = load_metadata()
    qvec = encode_text(query)

    dim = load_features_ram().shape[1]
    if qvec.shape[1] != dim:
        st.error(f"Encoder dim {qvec.shape[1]} != vector dim {dim}. "
                 f"Sai biến thể CLIP — kiểm lại manifest.json.")
        st.stop()

    loai = (set(st.session_state.get("loai_video", []))
            if st.session_state.get("dung_tich", False) else set())
    mark_rows = [int(x) for x in (mark_rows or [])]
    if mark_rows:
        ens_w = ens_w or ensemble.W_MAC_DINH   # phản hồi cần cả hai encoder

    if ens_w > 0:
        FR = load_features_ram()
        chinh = FR @ qvec[0]
        phu = ensemble.diem_phu(query)
        if mark_rows:
            mF = FR[mark_rows].mean(0)
            mF /= np.linalg.norm(mF) + 1e-9
            G = ensemble.features_phu()
            mG = G[mark_rows].mean(0)
            mG /= np.linalg.norm(mG) + 1e-9
            chinh = chinh + beta * (FR @ mF)
            phu = phu + beta * (G @ mG)
        tong = ensemble.zscore(chinh) + ens_w * ensemble.zscore(phu)
        if mark_rows and thuong_video:
            vids = set(meta["video_id"].to_numpy()[mark_rows])
            tong = tong + thuong_video * np.isin(meta["video_id"].to_numpy(),
                                                 list(vids))
        if loai:
            tong = np.where(np.isin(meta["video_id"].to_numpy(), list(loai)),
                            -np.inf, tong)
        top = np.argsort(-tong, kind="stable")[:top_k]
        scores, ids = tong[None, top], top[None, :]
    else:
        scores, ids = load_index().search(qvec, top_k)   # chỉ nạp khi thật cần
    hits = meta.iloc[ids[0]].copy()
    hits["score"] = scores[0]
    hits["row_id"] = ids[0]
    return hits.reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=512)
def _anh_mo(path: str) -> bytes:
    import io
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im = Image.blend(im, Image.new("RGB", im.size, (255, 255, 255)), 0.72)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


def bo_gan_trung(hits: pd.DataFrame, nguong: float = 0.95) -> pd.DataFrame:
    if hits.empty or "row_id" not in hits:
        return hits
    feats = load_features()
    giu, vecs = [], []
    for j, rid in enumerate(hits["row_id"].to_numpy()):
        v = np.asarray(feats[int(rid)], dtype=np.float32)
        if any(float(v @ u) >= nguong for u in vecs):
            continue
        giu.append(j)
        vecs.append(v)
    return hits.iloc[giu].reset_index(drop=True)


def diversify_by_video(hits: pd.DataFrame, per_video: int) -> pd.DataFrame:
    return (hits.groupby("video_id", sort=False)
                .head(per_video)
                .reset_index(drop=True))


@st.cache_resource(show_spinner="Loading features ...")
def load_features():
    """Vector ảnh, dùng cho tìm-bằng-ảnh và tìm-trong-video."""
    return np.load(PROCESSED / "features.npy", mmap_mode="r")


@st.cache_resource(show_spinner="Nạp vector vào RAM cho ensemble ...")
def load_features_ram():
    return np.load(PROCESSED / "features.npy")


def kho() -> dict:
    """Kho bài nộp: {tên file .csv -> danh sách dòng}. Sống suốt phiên."""
    return st.session_state.setdefault("kho", {})


def luu_kho(ten: str, dong: list) -> str:
    ten = ten.strip()
    ten = ten if ten.endswith(".csv") else f"{ten}.csv"
    kho()[ten] = dong
    return ten


def o_luu(ten_mac_dinh: str, khoa: str, dung_dong):
    """Ô nhập tên file + nút lưu, dùng chung cho cả ba tab."""
    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        ten = c1.text_input(
            "Tên file nộp", ten_mac_dinh, key=f"ten_{khoa}",
            help="Lấy ĐÚNG tên file truy vấn BTC phát, chỉ đổi .txt thành .csv. "
                 "Quy ước: query-<số>-<kis|qa|trake>.csv")
        c2.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
        if c2.button("💾 Lưu vào kho", key=f"luu_{khoa}", type="primary",
                     width="stretch"):
            try:
                dong = dung_dong()
            except ValueError as exc:
                st.error(str(exc))
                return
            t = luu_kho(ten, dong)
            st.toast(f"{t}: đã lưu {len(dong)} dòng")
            st.rerun()


def search_by_image(row_id: int, top_k: int, same_video_only: bool) -> pd.DataFrame:
    meta = load_metadata()
    feats = load_features()
    v = np.asarray(feats[row_id], dtype=np.float32)

    if same_video_only:
        scope = np.where(meta["video_id"].values == meta["video_id"].iloc[row_id])[0]
    else:
        scope = np.arange(len(meta))

    sims = np.asarray(feats[scope], dtype=np.float32) @ v
    keep = np.argsort(-sims)[:top_k]
    order = scope[keep]
    hits = meta.iloc[order].copy()
    hits["score"] = sims[keep]
    hits["row_id"] = order
    return hits.reset_index(drop=True)


def group_by_video(hits: pd.DataFrame) -> pd.DataFrame:
    g = (hits.groupby("video_id", sort=False)
             .agg(n_hit=("score", "size"), score=("score", "max"),
                  row=("score", "idxmax"))
             .sort_values("score", ascending=False)
             .reset_index())
    cot = ["n", "pts_time", "frame_idx", "image_path"]
    if "row_id" in hits.columns:
        cot.append("row_id")
    return g.join(hits.loc[g["row"], cot].reset_index(drop=True))


def all_frames_of(video_id: str) -> pd.DataFrame:
    meta = load_metadata()
    return meta[meta["video_id"] == video_id].copy()


def apply_rerank(hits: pd.DataFrame, query_vi: str, query_en: str,
                 depth: int, use_encoder: bool,
                 deep: int = 0) -> tuple[pd.DataFrame, np.ndarray, bool]:
    base = hits["score"].to_numpy()
    went_deep = False

    if deep and deep > depth:
        paths = hits["image_path"].tolist()[:deep]
        with st.spinner(f"VLM đang chấm {depth} ảnh ..."):
            vlm, went_deep = rerank.vlm_scores_adaptive(
                query_vi, paths, base, shallow=depth, deep=deep)
        if went_deep:
            st.info(f"Rổ {depth} đầu trông không chắc chắn (điểm quá sát nhau) — "
                    f"đã tự chấm tiếp tới {deep} ảnh.")
    else:
        paths = hits["image_path"].tolist()[:depth]
        with st.spinner(f"VLM đang chấm {len(paths)} ảnh ..."):
            vlm = rerank.vlm_scores(query_vi, paths)

    depth = len(vlm)                      # số ảnh THẬT SỰ đã chấm
    head = hits.iloc[:depth]

    enc = None
    if use_encoder:
        with st.spinner(f"Encoder đang mã hoá lại {depth} ảnh (chậm trên CPU) ..."):
            enc = rerank.encoder_scores(query_en, head["image_path"].tolist())

    rr = rerank.combine(head["score"].to_numpy(), vlm=vlm, enc=enc)
    order = np.argsort(-rr, kind="stable")

    head = head.iloc[order].copy()
    head["vlm"] = vlm[order]
    head["rr"] = rr[order]

    tail = hits.iloc[depth:].copy()
    tail["vlm"] = np.nan
    tail["rr"] = np.nan
    return pd.concat([head, tail], ignore_index=True), vlm, went_deep


def frame_card(hit, show_similar_button=True):
    """Một ô ảnh + thông tin nộp bài + nút tìm khung tương tự trong cùng video."""
    vd = str(hit["video_id"])
    bi_loai = vd in st.session_state.get("loai_video", [])
    img = hit.get("image_path", "")
    if img and Path(img).exists():
        st.image(_anh_mo(img) if bi_loai else img, width="stretch")
    else:
        st.markdown(":grey_background[no image]")

    st.code(f"{hit['video_id']}, {int(hit['frame_idx'])}", language=None)

    v = hit.get("vlm", np.nan)
    phu = [f"{hit['pts_time']:.0f}s"]
    if not pd.isna(v):
        phu.append(f":orange[VLM {v:.0f}]")
    st.caption(" · ".join(phu))

    if "row_id" in hit:
        rid = int(hit["row_id"])
        da = rid in st.session_state.get("marked", [])
        c1, c2, c3 = st.columns(3) if show_similar_button else (st, None, st)
        if c1.checkbox("OK", value=da, key=f"mk_{rid}",
                       help="Đúng hướng — đưa vào phản hồi") != da:
            ds = list(st.session_state.get("marked", []))
            ds.remove(rid) if da else ds.append(rid)
            st.session_state["marked"] = ds
            st.rerun()
        if c3.checkbox("Loại", value=bi_loai, key=f"lk_{rid}",
                       help=f"Làm mờ mọi thẻ của {vd} — đã xem, không phải cái này"
                       ) != bi_loai:
            ds = set(st.session_state.get("loai_video", []))
            ds.discard(vd) if bi_loai else ds.add(vd)
            st.session_state["loai_video"] = list(ds)
            st.rerun()
        if c2 is not None and c2.button(
                "≈", key=f"sim_{rid}",
                help="Tìm khung giống khung này TRONG CÙNG VIDEO. Đo được: giới "
                     "hạn trong video cho FINAL 0.800, toàn corpus chỉ 0.678."):
            st.session_state["seed_row"] = rid
            st.rerun()


def show_frames(df, ncol, similar=True):
    for start in range(0, len(df), ncol):
        cols = st.columns(ncol)
        for col, (_, hit) in zip(cols, df.iloc[start:start + ncol].iterrows()):
            with col:
                frame_card(hit, similar)


def show_videos(hits, ncol):
    """Chế độ video: mỗi video một ô, bấm vào để bung toàn bộ khung của nó."""
    vids = group_by_video(hits)
    st.caption(f"{len(vids)} video — bấm **Mở** để xem mọi khung của video đó")

    for start in range(0, len(vids), ncol):
        cols = st.columns(ncol)
        for col, (_, v) in zip(cols, vids.iloc[start:start + ncol].iterrows()):
            with col:
                bi_loai = v["video_id"] in st.session_state.get("loai_video", [])
                img = v.get("image_path", "")
                if img and Path(img).exists():
                    st.image(_anh_mo(img) if bi_loai else img, width="stretch")
                st.code(f"{v['video_id']}, {int(v['frame_idx'])}", language=None)
                st.caption(f"{int(v['n_hit'])} khung khớp")

                rid_v = int(v["row_id"]) if "row_id" in v.index else None
                d1, d2 = st.columns(2)
                if rid_v is not None:
                    da_v = rid_v in st.session_state.get("marked", [])
                    # Khoá theo row_id chứ KHÔNG theo video_id — xem phụ lục B.
                    if d1.checkbox("OK", value=da_v, key=f"mkv_{rid_v}",
                                   help="Đúng hướng — đưa vào phản hồi") != da_v:
                        ds = list(st.session_state.get("marked", []))
                        ds.remove(rid_v) if da_v else ds.append(rid_v)
                        st.session_state["marked"] = ds
                        st.rerun()
                if d2.checkbox("Loại", value=bi_loai, key=f"loai_{v['video_id']}",
                               help="Làm mờ video này — đã xem, không phải cái này"
                               ) != bi_loai:
                    ds = set(st.session_state.get("loai_video", []))
                    ds.discard(v["video_id"]) if bi_loai else ds.add(v["video_id"])
                    st.session_state["loai_video"] = list(ds)
                    st.rerun()
                if st.button("Mở toàn bộ khung", key=f"open_{v['video_id']}",
                             width="stretch"):
                    st.session_state["open_video"] = v["video_id"]

    vid = st.session_state.get("open_video")
    if vid:
        st.divider()
        head, close = st.columns([6, 1])
        head.subheader(f"{vid} — toàn bộ khung")
        if close.button("Đóng"):
            st.session_state.pop("open_video", None)
            st.rerun()

        frames = all_frames_of(vid)
        frames["row_id"] = frames.index
        frames["score"] = frames["row_id"].map(hits.set_index("row_id")["score"])
        st.caption(f"{len(frames)} khung, xếp theo thời gian. "
                   f"Khung có điểm là khung đã lọt vào kết quả tìm kiếm.")
        show_frames(frames.sort_values("n"), ncol, similar=False)


st.title("RED-NEUTRONS — AIC 2026")

with st.sidebar:
    top_k = st.slider("Số ô hiển thị", 20, 200, 50, step=10,
                      help="Quét hết 50 ô rồi mới tìm lại — đo được là hơn hẳn "
                           "dừng ở 20 (phút 2: 0,610 so với 0,565).")
    cols_per_row = st.slider("Số cột", 3, 8, 5)
    view = st.radio(
        "Xem theo", ["Video (nên dùng)", "Khung"], index=0, horizontal=True,
        help="Đúng VIDEO 0,988 còn đúng KHUNG chỉ 0,654 — máy gần như luôn tìm "
             "đúng video, chỉ không biết đúng giây, mà giây thì người liếc là thấy.")
    video_mode = view.startswith("Video")

    with st.expander("Nâng cao — đã hiệu chỉnh, đừng đổi khi thi"):
        st.caption(f"Encoder: **{CLIP_MODEL}** · nền {BASELINE:.4f} trên 81 truy vấn")

        bo_trung = st.checkbox(
            "Gộp khung gần trùng (chỉ khi xem)", value=True,
            help="23/50 ô đầu là bản gần trùng của một ô đứng trên. Gộp lại thì "
                 "cùng 50 ô cho ~27 khoảnh khắc khác nhau. KHÔNG áp dụng cho bảng "
                 "nộp — ở đó nó làm mất khung đáp án ở 10/81 truy vấn.")
        nguong_trung = st.slider("Ngưỡng gần trùng", 0.90, 0.99, 0.95, 0.01,
                                 disabled=not bo_trung)

        # 20 = TẮT. Mọi mức thấp hơn đều làm tụt điểm; chỉ hạ khi đang DÒ dữ liệu.
        per_video = st.slider("Tối đa số khung mỗi video", 1, 20, 20,
                              help="20 = tắt. Hạ xuống làm TỤT điểm KIS "
                                   "(5→0.4667, 3→0.4346 so với 0.4765).")

        use_vlm = st.checkbox(
            "VLM re-rank (Gemini) — chậm", value=False,
            help="Đáng +0,010 ở gợi ý ngắn, +0,005 ở mô tả đầy đủ, nhưng mỗi lần "
                 "tìm chậm thêm 9-20 giây (và 30 giây nữa nếu khoá bị chặn). "
                 "Chỉ bật khi đã tìm xong và còn dư thời gian.")
        rerank_depth = st.select_slider("Chấm lại bao nhiêu ô đầu",
                                        options=[10, 20, 30, 50], value=20,
                                        disabled=not use_vlm)
        auto_deep = st.checkbox(
            "Đào sâu khi rổ có vẻ hỏng", value=False, disabled=not use_vlm,
            help="Lợi ở nhóm khó (0.215→0.262) nhưng hại ở nhóm dễ "
                 "(0.906→0.882), mà nhóm dễ chiếm 84%. Với NGƯỜI ngồi tìm thì "
                 "đào sâu không bao giờ hại, nên vẫn giữ nút.")
        deep_to = st.select_slider("Đào sâu tới", options=[100, 200, 300],
                                   value=200, disabled=not (use_vlm and auto_deep))
        # Encoder thứ hai chấm lại rổ: đã đo là KHÔNG cộng dồn với VLM, chỉ chậm.
        use_encoder = False

        if not ensemble.san_sang():
            ens_w = 0.0
            st.caption(f"⚠️ Chưa có `{ensemble.PHU_FEATURES.name}` trong "
                       f"`data/kernel_out/` — xem README mục cài đặt.")
        else:
            # Cả dải 0,2-1,5 đều vượt nền; 0,5 là điểm tốt nhất đã đo.
            ens_w = st.select_slider(
                "Trọng số encoder phụ", options=[0.0, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5],
                value=ensemble.W_MAC_DINH,
                help="FINAL: tắt 0.7901 · 0.2→0.8049 · 0.5→0.8148 · 0.7→0.8074 "
                     "· 1.0→0.8025.")

    st.divider()
    with st.expander("🍳 Tra bảng nguyên liệu (nấu ăn)"):
        import ocr_thecuoi as tcmod
        _tc = tcmod.bang()
        if _tc is None:
            st.caption("Chưa lập chỉ mục. Chạy `python src/ocr_thecuoi.py` "
                       "(~23 phút) để quét thẻ tóm tắt cuối mỗi video L26.")
        else:
            st.caption(f"{len(_tc)} thẻ. Gõ tên nguyên liệu — không cần dấu, "
                       f"máy OCR cũng không có dấu.")
            _q = st.text_input("nguyên liệu", key="tc_q",
                               placeholder="vd: thit bo bam, giam gao, ot sung")
            if _q.strip():
                _kq = tcmod.tim(_q, limit=8)
                if not len(_kq):
                    st.info("Không thẻ nào chứa từ này.")
                for _r in _kq.itertuples():
                    st.write(f"**{_r.video_id}** · frame {_r.frame_idx} "
                             f"· điểm {_r.diem}")

    with st.expander("🔤 Tra chữ trên khung"):
        import ocr_chu as ocmod
        _oc = ocmod.bang()
        if _oc is None:
            st.caption("Chưa lập chỉ mục. Chạy `python src/ocr_chu.py --nhom "
                       "L22 L21 L25 L23` — bốn nhóm đặc chữ nhất, ~7 giờ.")
        else:
            _nh = sorted(_oc.video_id.str[:3].unique())
            st.caption(f"{len(_oc):,} khung · nhóm {' '.join(_nh)}. Gõ chữ bạn "
                       f"đoán CÓ TRÊN MÀN HÌNH, không phải mô tả cảnh — gõ mô "
                       f"tả thì kênh này thua hẳn CLIP.")
            _qc = st.text_input("chữ trên khung", key="oc_q",
                                placeholder="vd: chang 4, 1.1 km, bo giao duc")
            if _qc.strip():
                _kc = ocmod.tim(_qc, limit=6)
                if not len(_kc):
                    st.info("Không khung nào chứa từ này.")
                _m = load_metadata()
                _hang = {(v, int(n)): i for i, (v, n)
                         in enumerate(zip(_m.video_id, _m.n))}
                for _r in _kc.itertuples():
                    _i = _hang.get((_r.video_id, int(_r.n)))
                    st.write(f"**{_r.video_id}** · frame {_r.frame_idx} "
                             f"· điểm {_r.diem}")
                    if _i is not None:
                        st.image(_m.image_path.iat[_i], width="stretch")
                        # Nối vào đúng cơ chế tìm-ảnh-giống của tab KIS: một
                        # khung đọc được chữ mới là đầu mối, không phải đáp án.
                        if st.button("🔍 Tìm ảnh giống", key=f"oc_seed_{_i}",
                                     width="stretch"):
                            st.session_state["seed_row"] = int(_i)
                            st.rerun()
                    st.caption(_r.text.replace(chr(10), " · ")[:150])

    st.divider()
    st.subheader(f"📦 Kho bài nộp ({len(kho())})")
    if not kho():
        st.caption("Trống. Tìm xong ở mỗi tab thì bấm **Lưu vào kho** — mỗi câu "
                   "truy vấn một file .csv, tối đa 100 dòng.")
    else:
        hong = []
        for ten in sorted(kho()):
            dong = kho()[ten]
            loi = nopbai.kiem_tep(ten, dong)
            if loi:
                hong.append(ten)
            st.markdown(f"{'🔴' if loi else '✅'} `{ten}` · **{len(dong)}** dòng")
        if hong:
            st.error(f"{len(hong)} file còn lỗi — mở tab **📦 Nộp bài** để soi và "
                     f"sửa. Nộp sai định dạng VẪN trừ một lượt trong ba lượt.")
        else:
            st.download_button(
                "📦 Tải submission.zip", nopbai.dong_goi(dict(kho())),
                file_name="submission.zip", mime="application/zip",
                type="primary", width="stretch",
                help="Đã có sẵn thư mục submission/ bên trong đúng như thể lệ. "
                     "Nộp thẳng file này lên hệ thống BTC.")

tab_kis, tab_qa, tab_trake, tab_nop = st.tabs(
    ["KIS", "Q&A", "TRAKE", "📦 Nộp bài"])


with tab_kis:
    query_vi = st.text_input(
        "Gợi ý từ ban tổ chức (dán nguyên, tiếng Việt)",
        placeholder="vd: người phụ nữ đội nón lá đang hái dứa ngoài ruộng",
        help="Dán NGUYÊN văn gợi ý. Đo được: nhờ LLM viết lại cho 'đầy đủ hơn' "
             "làm TỤT điểm ở mọi mức (phút 2: 0,486 → 0,403).")

    with st.expander("Cách xử lý câu"):
        tr_mode = st.selectbox(
            "Xử lý câu", ["dịch sang tiếng Anh", "để nguyên tiếng Việt"], index=0,
            help="Đo lại 17/08 trên nền hiện tại: dịch 0.7827 · để nguyên 0.7457 "
                 "(+0,037). Quan trọng hơn: dịch kéo R@100 từ 0,901 lên 0,975 — "
                 "6 câu vốn KHÔNG lọt top-100 ở đâu cả nay đã hiện ra. ĐỪNG đổi "
                 "sang 'để nguyên' trừ khi mất mạng; hỏng thì nó tự lùi về câu gốc.")
        query_en_override = st.text_input(
            "Hoặc tự viết câu tiếng Anh", placeholder="women in ao dai in a lotus field",
            help="Chỉ dùng khi bạn thật sự tả sát hình. Đo được: máy dịch đã "
                 "THẮNG người viết tay ở R@1 (0,556 so với 0,543) vì nó bám sát "
                 "chữ trong đề, còn người viết trôi chảy thì xa mô tả pixel hơn.")

    _cu = st.session_state.get("truy_van_truoc", "")
    _moi = (query_vi or "").strip()
    if _moi and _cu and not (_moi.startswith(_cu) or _cu.startswith(_moi)):
        if st.session_state.get("marked") or st.session_state.get("loai_video"):
            st.session_state["marked"] = []
            st.session_state["loai_video"] = []
            st.toast("Câu truy vấn mới — đã xoá đánh dấu và video đã loại")
    if _moi:
        st.session_state["truy_van_truoc"] = _moi

    # ── Một dòng trạng thái cho cả hai cơ chế phản hồi ───────────────────────
    marked = st.session_state.get("marked", [])
    da_loai = st.session_state.get("loai_video", [])
    beta = 0.4
    if marked or da_loai:
        s1, s2, s3 = st.columns([3, 1, 1])
        phan = []
        if marked:
            phan.append(f"**{len(marked)}** khung đánh dấu")
        if da_loai:
            phan.append(f"**{len(da_loai)}** video đã loại")
        s1.success(" · ".join(phan))
        uu_tien = s2.toggle(
            "Dùng ô tích để tìm", value=st.session_state.get("dung_tich", False),
            key="dung_tich",
            help="TẮT (mặc định): ô tích chỉ để lọc bằng mắt. BẬT: ô OK kéo kết "
                 "quả về phía khung đã tích (+0,141 ở phút 2) và ô Loại gạt hẳn "
                 "video khỏi tìm kiếm (+0,121) — nhưng danh sách sẽ xáo lại.")
        if s3.button("Xoá hết", help="Bỏ mọi đánh dấu và mọi video đã loại"):
            st.session_state["marked"] = []
            st.session_state["loai_video"] = []
            st.rerun()
    else:
        uu_tien = st.session_state.get("dung_tich", False)
        st.caption("Tích **OK** ở thẻ đúng cảnh, tích **Loại** ở thẻ đã xem và "
                   "thấy sai — thẻ bị loại sẽ mờ đi để bạn khỏi nhìn lại.")

    seed = st.session_state.get("seed_row")
    if seed is not None:
        _meta = load_metadata()
        _s = _meta.iloc[seed]
        st.subheader(f"Khung giống {_s['video_id']} n={int(_s['n'])}")
        c1, c2 = st.columns([3, 1])
        same_only = c1.checkbox(
            "Chỉ tìm trong cùng video", value=True,
            help="ĐO ĐƯỢC: giới hạn trong video cho FINAL 0.800 (R@5 = 0.780); mở "
                 "toàn corpus chỉ 0.678; còn tìm-bằng-ảnh từ một khung bấm đại thì "
                 "chỉ 0.212 — tệ hơn cả tìm bằng chữ. Láng giềng của ảnh sai thì cũng sai.")
        if c2.button("Đóng khung tương tự"):
            st.session_state.pop("seed_row", None)
            st.rerun()
        show_frames(search_by_image(seed, top_k, same_only), cols_per_row, similar=False)
        st.divider()

    if query_vi.strip() or query_en_override.strip():
        if query_en_override.strip():
            query, note = query_en_override.strip(), "→ dùng câu tiếng Anh bạn tự viết"
        elif tr_mode.startswith("dịch"):
            with st.spinner("Đang dịch ..."):
                query, note = preprocess_query(query_vi, "google")
        else:
            query, note = preprocess_query(query_vi, "vi")
        if note:
            st.caption(note)

        depth = rerank_depth if use_vlm else 0
        deep = deep_to if (use_vlm and auto_deep) else 0
        hits = search(query, max(top_k, depth, deep, 100), ens_w=ens_w,
                      # Ô tích chỉ đụng tới kết quả khi công tắc được bật.
                      mark_rows=st.session_state.get("marked") if uu_tien else None,
                      beta=beta,
                      thuong_video=3.0 if uu_tien else 0.3)

        vlm_scores = None
        if use_vlm and depth:
            try:
                hits, vlm_scores, _ = apply_rerank(hits, query_vi.strip() or query,
                                                   query, depth, use_encoder, deep)
            except Exception as exc:                       # thiếu API key, hết hạn mức...
                st.warning(f"Re-rank hỏng, hiển thị kết quả CLIP thuần — {exc}")

        nop = hits.iloc[:100]
        hits = diversify_by_video(hits, per_video).iloc[:top_k]
        if bo_trung:
            truoc = len(hits)
            hits = bo_gan_trung(hits, nguong_trung)
            if truoc != len(hits):
                st.caption(f"đã gộp {truoc - len(hits)} khung gần trùng — còn "
                           f"{len(hits)} khoảnh khắc khác nhau (chỉ ảnh hưởng "
                           f"danh sách xem, danh sách nộp giữ nguyên 100 dòng)")

        if vlm_scores is not None:
            if rerank.all_failed(vlm_scores):
                st.warning("VLM không chấm được ô nào (hết hạn mức hoặc mất mạng) — "
                           "đang hiển thị thứ tự CLIP thuần.")
            elif rerank.basket_uncertain(hits["score"].to_numpy(), shallow=rerank_depth):
                st.error("⚠️ Điểm của các ứng viên đầu bảng quá sát nhau — dấu hiệu "
                         "**đáp án có thể không nằm trong rổ này**. Nên đổi cách diễn "
                         "đạt câu truy vấn, tăng độ sâu, hoặc tìm bằng ảnh.")

        st.caption(f"{len(hits)} keyframes from {hits['video_id'].nunique()} videos")

        st.caption(f"{len(nop)} ứng viên · mã nộp nằm ngay dưới mỗi ảnh, "
                   f"bấm vào là chép được")

        # Nộp cả 100 dòng, không phải một. R@1 chỉ 0,481 còn R@100 là 0,951, mà
        # thể lệ không phạt dòng sai — dòng 2-100 là bảo hiểm miễn phí.
        o_luu("query-1-kis", "kis",
              lambda: [nopbai.dong_kis(r.video_id, r.frame_idx)
                       for r in nop.itertuples()])

        if video_mode:
            show_videos(hits, cols_per_row)
        else:
            show_frames(hits, cols_per_row)
    else:
        st.info("Enter a query to search.")


# ══════════════════════════════ Q&A ══════════════════════════════
with tab_qa:
    st.caption(
        "Đề Q&A gồm **mô tả sự kiện + một câu hỏi**. Nộp `video_id, frame_id, answer`; "
        "**sai câu trả lời là 0 điểm dù tìm đúng khung** — đây là dạng duy nhất mà "
        "tìm kiếm giỏi vẫn có thể ăn 0.")
    qa_desc = st.text_input(
        "Mô tả sự kiện (tiếng Việt)", key="qa_desc",
        placeholder="vd: một nhân viên vườn thú đang cho đàn chim ăn bên hồ nước",
        help="CHỈ mô tả mới được dùng để TÌM. Đo được: nhét thêm câu hỏi vào vector "
             "tìm kiếm không cải thiện gì (35,7%), dùng riêng câu hỏi thì tệ hẳn (7,1%).")
    qa_ques = st.text_input(
        "Câu hỏi (tiếng Việt)", key="qa_ques",
        placeholder="vd: đàn chim trong ảnh là loài gì?")
    qa_top = st.select_slider("Số ảnh đưa cho VLM xem", options=[10, 20, 30, 50],
                              value=20, key="qa_top")
    qa_kieu = st.radio(
        "Kiểu câu hỏi", ["nhận dạng", "đếm"], horizontal=True, key="qa_kieu",
        help="Chế độ đếm hỏi CẢ toàn khung LẪN từng ô rời, rồi chọn theo độ "
             "lớn: ít vật thì tin toàn khung (4/4), nhiều ký hiệu nhỏ thì cộng "
             "ô. Hai con số lệch nhau thì nó báo, đừng bỏ qua cảnh báo đó — "
             "vài chục ký hiệu nhỏ là chỗ VLM đếm mỗi lần một khác.")
    # Đo được: cùng một khung hỏi 5 lần ra 2, 3, 1, 1, 11. Chạy lại vài lần rồi
    # nhìn CẢ PHỔ là tín hiệu tin cậy duy nhất có thật cho câu đếm — khác hẳn
    # confidence do mô hình tự khai, thứ đã đo được là vô dụng.
    qa_lan = 1
    if qa_kieu == "đếm":
        qa_lan = st.select_slider(
            "Đếm lại mấy lần (bước 3)", options=[1, 3, 5], value=3, key="qa_lan",
            help="Mỗi lần đếm tốn 7 lời gọi API. Ba lần ra ba số khác nhau "
                 "nghĩa là đừng tin số nào cả — hãy tự đếm bằng mắt.")

    qa_marked = st.session_state.get("marked", [])
    if qa_marked:
        st.success(f"Đang dùng {len(qa_marked)} khung đánh dấu (đánh dấu ở tab KIS "
                   f"hoặc ngay trong kết quả bên dưới)")
    qa_uu_tien = st.checkbox(
        "Ưu tiên video đã đánh dấu (chỉ bật khi CHẮC CHẮN)", value=False,
        key="qa_uu_tien",
        help="Điểm khung 0,8160 → 0,8440 NẾU nhận đúng video. Nhận nhầm thì mất "
             "nhiều hơn được: hoà vốn ở 84%, xem báo cáo phụ lục B.")

    # HAI BƯỚC, không gộp. Đo được: 30/51 câu trả lời sai là vì khung đáp án nằm
    # NGOÀI rổ ảnh đưa cho VLM (đúng 79% khi trong rổ, 30% khi ngoài). Gộp một nút
    # thì người thi tiêu lời gọi API trên rổ CHƯA đánh dấu — hỏng đúng chỗ đắt nhất.
    b1, b2 = st.columns(2)
    if b1.button("1 · Tìm khung", key="qa_tim"):
        st.session_state["qa_da_tim"] = True
    tra_loi = b2.button("2 · Trả lời từ rổ hiện tại", key="qa_go",
                        type="primary", disabled=not st.session_state.get("qa_da_tim"))

    # Mô tả đổi thì rổ cũ vô nghĩa, buộc tìm lại.
    if st.session_state.get("qa_desc_truoc") != qa_desc:
        st.session_state["qa_desc_truoc"] = qa_desc
        st.session_state["qa_da_tim"] = False
        st.session_state.pop("qa_got", None)      # đáp án cũ thuộc rổ cũ
        st.session_state.pop("qa_ky", None)
        st.session_state.pop("qa_ocr", None)
        st.session_state.pop("qa_asr_dap", None)

    if st.session_state.get("qa_da_tim"):
        if not qa_desc.strip():
            st.warning("Cần mô tả sự kiện.")
        elif tra_loi and not qa_ques.strip():
            st.warning("Cần câu hỏi để trả lời.")
        else:
            import qa as qamod
            hits = search(qa_desc.strip(), max(qa_top, 100), ens_w=ens_w,
                          mark_rows=qa_marked,
                          thuong_video=3.0 if qa_uu_tien else 0.3)
            if tra_loi:
                with st.spinner("Đang hỏi VLM ..."):
                    st.session_state["qa_got"] = qamod.answer_over_hits(
                        qa_ques.strip(), hits, desc=qa_desc.strip(), top=qa_top)
            # Giữ đáp án qua các lần chạy lại: mỗi lần hỏi lại là một lời gọi API.
            got = st.session_state.get("qa_got")
            if got is None and not tra_loi:
                st.info(f"Rổ đã dựng, chưa tốn lời gọi API nào. **Tích OK ở khung "
                        f"đúng cảnh** rồi mới bấm bước 2 — khung đáp án phải nằm "
                        f"trong {qa_top} ảnh đầu, nếu không thì VLM không có gì "
                        f"để đọc (đo được: trong rổ đúng 79%, ngoài rổ 30%).")
            elif got is None:
                st.error("Không gọi được VLM (hết hạn mức hoặc mất mạng). Dưới đây "
                         "là kết quả tìm kiếm thuần — tự đọc ảnh và tự trả lời.")
            else:
                c1, c2 = st.columns([1, 2])
                with c1:
                    if Path(got["image_path"]).exists():
                        st.image(got["image_path"], width="stretch")
                with c2:
                    if got["answer"]:
                        st.subheader(got["answer"])
                    else:
                        # Im lặng KHÔNG phải lỗi API. Rổ đã kèm lời nói của
                        # từng ảnh, nên im lặng nghĩa là đáp án không có trong
                        # ảnh LẪN lời nói của cả 20 khung — hầu như luôn vì rổ
                        # chưa chứa khung đúng (đo được: 7/20 câu tên riêng).
                        st.subheader("Chưa đủ bằng chứng để trả lời")
                        st.info("Rổ này đã gồm cả **ảnh** lẫn **lời thuyết minh** "
                                "của 20 khung, nên im lặng nghĩa là khung đáp án "
                                "chưa nằm trong rổ. Sửa **mô tả sự kiện** rồi tìm "
                                "lại, hoặc tích OK vài khung đúng cảnh để đẩy rổ "
                                "— đừng hỏi lại nguyên văn, kết quả sẽ y hệt.")
                    # confidence do VLM TỰ KHAI: đo được 10/10 ở một câu
                    # bịa hẳn đáp án, 0/10 ở câu mà đáp án nằm ngay hạng 1.
                    # Bày kèm cảnh báo, tuyệt đối không dùng để lọc.
                    st.caption(f"kênh **{got.get('kenh', 'chỉ ảnh')}** · "
                               f"{got['reason']}")
                    st.caption(f":gray[mô hình tự khai {got['confidence']:.0f}/10 "
                               f"— con số này KHÔNG đáng tin, đã đo: 10/10 ở một "
                               f"câu bịa hẳn đáp án. Đừng dùng nó để quyết định.]")
                    st.code(f"{got['video_id']}, {got['frame_idx']}, {got['answer']}",
                            language=None)
                    if got["vlm_frame_idx"] != got["frame_idx"]:
                        st.warning(
                            f"VLM đọc đáp án từ **{got['vlm_video_id']}** frame "
                            f"{got['vlm_frame_idx']}, khác khung đang nộp "
                            f"({got['video_id']} frame {got['frame_idx']}). "
                            "Hai chỗ lệch nhau là dấu hiệu nên soi lại bằng mắt.")

                # Bước 3 — đọc lại ĐÚNG một khung ở độ phân giải gốc. Khung
                # 1280x720 gửi ở 512px chỉ còn 512x288, số trên biển báo mất hẳn.
                v_dung = got["vlm_video_id"] or got["video_id"]
                f_dung = got["vlm_frame_idx"] or got["frame_idx"]
                kh = hits[(hits.video_id == v_dung) & (hits.frame_idx == f_dung)]
                p_anh = (str(kh.iloc[0]["image_path"]) if len(kh)
                         else got["image_path"])

                loi_noi = qamod.asr_khung(v_dung, f_dung)
                if loi_noi:
                    with st.expander("🎙️ Lời nói quanh khung này (miễn phí)"):
                        st.write(loi_noi)
                        # Câu hỏi TÊN thì kênh này một mình đã 0,950, mà chỉ tốn
                        # một lời gọi — không việc gì bắt chờ đủ 9 góc nhìn.
                        if st.button("Hỏi thẳng lời nói (1 lời gọi, ~5 giây)",
                                     key="qa_hoi_asr"):
                            with st.spinner("Đang đọc lời nói ..."):
                                st.session_state["qa_asr_dap"] = (qamod._hoi_mot(
                                    qamod.PROMPT_ASR.format(
                                        asr=loi_noi[:2000],
                                        question=qa_ques.strip()), None) or {})
                        da = st.session_state.get("qa_asr_dap")
                        if da is not None:
                            st.success(f"Lời nói trả lời: "
                                       f"**{da.get('answer') or '(không nói tới)'}**")
                            st.caption("Hỏi TÊN (đèo, cầu, trường, giải đấu) thì "
                                       "tin kênh này: 0,950 so với 0,100 của kênh "
                                       "ảnh trên bộ đo 20 câu tên riêng.")
                # OCR mất ~3 giây nên chỉ quét khi người thi mở ra xem.
                with st.expander("🔤 Chữ OCR đọc được trong khung (~3 giây)"):
                    if st.button("Quét chữ", key="qa_ocr_nut"):
                        with st.spinner("Đang quét chữ ..."):
                            st.session_state["qa_ocr"] = (
                                qamod.ocr_khung(p_anh) or "(không thấy chữ nào)")
                    if st.session_state.get("qa_ocr"):
                        st.text(st.session_state["qa_ocr"])
                        st.caption("Máy OCR không có dấu tiếng Việt và hay lẫn "
                                   "8/B, 0/O — bước 3 tự khôi phục lại dấu.")

                nhan3 = ("3 · Đếm bằng cách chia ô rồi CỘNG" if qa_kieu == "đếm"
                         else "3 · Đọc kỹ khung này (ô phóng to + lời nói + OCR)")
                if st.button(nhan3, key="qa_docky"):
                    with st.spinner("Đang soi từng ô ..."):
                        if qa_kieu == "đếm":
                            # Ba lần chứ không một lần: đo được cùng một khung
                            # hỏi 5 lần ra 4 đáp án khác nhau.
                            st.session_state["qa_ky"] = qamod.dem_lap(
                                qa_ques.strip(), p_anh, desc=qa_desc.strip(),
                                lan=qa_lan)
                        else:
                            st.session_state["qa_ky"] = qamod.doc_ky(
                                qa_ques.strip(), p_anh, desc=qa_desc.strip(),
                                asr=loi_noi)
                ky = st.session_state.get("qa_ky")
                if ky:
                    st.success(f"Đọc kỹ: **{ky['answer']}** — {ky['reason']}")
                    if "count_toan" in ky:
                        st.caption(f"toàn khung đếm {ky['count_toan']} · "
                                   f"cộng ô đếm {ky['count_cong']}")
                    if ky.get("cac_lan") and not ky.get("on_dinh", True):
                        st.warning(
                            f"⚠ {len(ky['cac_lan'])} lần chạy ra {ky['cac_lan']}"
                            f" — **con số này là rút thăm**. Tự đếm bằng mắt "
                            f"trước khi nộp.")
                    elif ky.get("cac_lan"):
                        st.caption(f"{len(ky['cac_lan'])} lần chạy đều ra "
                                   f"{ky['count']} — đồng thuận.")
                    st.code(f"{v_dung}, {f_dung}, {ky['answer']}", language=None)
                    st.caption("Dòng trên là bản của bước 3. Muốn nộp nó thì sửa "
                               "trong tab 📦 Nộp bài — kho vẫn đang giữ bản bước 2.")
                    with st.expander("từng góc nhìn"):
                        st.table(ky.get("views", []))

                # Cả 100 dòng dùng CHUNG câu trả lời: quy chế chấm đáp án của
                # dòng trúng, nên đáp án sai là 0 dù khung đúng.
                o_luu("query-2-qa", "qa",
                      lambda: [nopbai.dong_qa(*r) for r in
                               qamod.submission_rows(hits, got, limit=100)])
                st.divider()
            st.caption(f"{qa_top} ứng viên đầu — ảnh đầu tiên là khung sẽ nộp")
            show_frames(hits.iloc[:qa_top], cols_per_row, similar=False)
    else:
        st.info("Nhập mô tả rồi bấm **1 · Tìm khung**. Đánh dấu khung đúng cảnh "
                "trước, rồi mới bấm **2 · Trả lời** — thứ tự này đáng nhiều điểm.")


# ═════════════════════════════ TRAKE ═════════════════════════════
with tab_trake:
    st.caption(
        "Đề TRAKE là một **chuỗi khoảnh khắc theo thời gian** trong CÙNG một video. "
        "Nộp `video_id, frame_id₁ … frame_idₙ`; **sai video là 0 ngay**, đúng video "
        "thì ăn theo tỉ lệ số mốc trúng. Cửa sổ mỗi mốc thường **dưới 10 frame** nên "
        "bắt buộc căn trên video gốc — keyframe cách nhau ~90 frame, không đủ mịn.")

    tk_moments = st.text_area(
        "Mô tả từng mốc — MỖI DÒNG MỘT MỐC, đúng thứ tự đề bài", height=150,
        key="tk_moments",
        placeholder="hai bàn tay trộn nhân trong tô thuỷ tinh\n"
                    "phết trứng lên tấm bánh tráng\n"
                    "cuốn bánh tráng thành cuốn chả giò\n"
                    "vớt chả giò vàng bằng vợt lưới",
        help="Thứ tự dòng CHÍNH LÀ ràng buộc thời gian — hệ thống ép mốc sau phải "
             "nằm sau mốc trước. Đo được: ép thứ tự giảm lệch 828 → 444 frame và "
             "sửa 3/3 chuỗi khỏi bị đảo ngược thời gian.")
    goc = [s.strip() for s in tk_moments.split("\n") if s.strip()]

    tk_dich = st.checkbox(
        "Dịch từng mốc sang tiếng Anh", value=True, key="tk_dich",
        help="Đo đầu–cuối 18/08 trên 8 chuỗi: tiếng Việt thô 0,3875 · dịch máy "
             "0,4000 · tiếng Anh VIẾT TAY 0,4417. Tiếng Việt thô còn mất một video "
             "ở bước 1 (7/8) — sai video là 0 điểm. Nếu bạn tự gõ được tiếng Anh "
             "sát hình thì gõ thẳng vào ô trên và TẮT ô này: hơn dịch máy 0,042.")

    if not goc:
        st.info("Nhập các mốc, mỗi dòng một mốc.")
        texts = []
    else:
        if tk_dich:
            with st.spinner("Đang dịch từng mốc ..."):
                texts = [preprocess_query(t, "google")[0] for t in goc]
            if texts != goc:
                st.caption("→ dịch: " + " · ".join(f"*{t}*" for t in texts))
        else:
            texts = list(goc)

    if texts:
        import trake as tkmod
        # Đổi mô tả mốc là mọi kết quả căn cũ thành rác — dọn, đừng để lẫn vào
        # 100 dòng nộp của câu mới.
        if st.session_state.get("tk_kho_texts") != texts:
            st.session_state["tk_kho_texts"] = list(texts)
            st.session_state["tk_kho"] = {}
            st.session_state.pop("tk_kq", None)
            st.session_state.pop("tk_kq_video", None)
        with st.spinner("Đang xếp hạng video theo cả chuỗi ..."):
            F = load_features_ram()
            S = np.vstack([F @ encode_text(t)[0] for t in texts])
            xh = tkmod.rank_videos_scores(S, top_k=10, kem_neo=True)
        meta_tk = load_metadata()

        st.subheader("1. Chọn video")
        st.caption(f"Mỗi hàng là MỘT video, bày cả **{len(texts)} mốc** theo đúng "
                   "thứ tự thời gian — đây chính là bộ mỏ neo hệ thống đã chọn cho "
                   "video đó. Nhìn cả hàng để biết video có chứa CẢ chuỗi hay chỉ "
                   "khớp một mốc. Xếp theo tổng điểm của chuỗi.")
        for vid, diem, hang in xh:
            st.markdown(f"**{vid}** · điểm chuỗi {diem:.3f}")
            cols = st.columns(len(texts))
            for j, (col, row) in enumerate(zip(cols, hang)):
                r = meta_tk.iloc[row]
                with col:
                    p = r["image_path"]
                    if p and Path(p).exists():
                        st.image(p, width="stretch")
                    st.caption(f"mốc {j + 1} · f{int(r['frame_idx'])}"
                               f"\n\n`{float(S[j, row]):.3f}`")
            st.divider()
        tk_video = st.selectbox("Video sẽ nộp", [v for v, _, _ in xh],
                                key="tk_video")

        st.subheader("2. Căn từng mốc trên video gốc")
        st.caption(f"{len(texts)} mốc × ~13 giây mã hoá ≈ **{len(texts) * 13} giây** "
                   f"(38 lần mã hoá mỗi mốc, bán kính dò ±60 frame).")
        if st.button("Căn chuỗi", key="tk_go", type="primary"):
            try:
                with st.spinner(f"Đang căn {len(texts)} mốc trên {tk_video} ..."):
                    # Giữ lại qua các lần chạy lại: căn một chuỗi tốn hàng chục giây.
                    kq = tkmod.locate_sequence(tk_video, texts)
                    st.session_state["tk_kq"] = kq
                    st.session_state["tk_kq_video"] = tk_video
                    # Căn thêm video khác thì CỘNG vào, không đè: mỗi video góp
                    # thêm dòng dự phòng cho đủ 100.
                    st.session_state.setdefault("tk_kho", {})[tk_video] = kq
            except FileNotFoundError as exc:
                st.error(str(exc))
        moments = st.session_state.get("tk_kq")
        if moments and st.session_state.get("tk_kq_video") != tk_video:
            st.info(f"Kết quả dưới đây căn trên **{st.session_state['tk_kq_video']}**, "
                    f"không phải video đang chọn. Bấm **Căn chuỗi** để căn lại.")
        if moments:
            tk_video = st.session_state["tk_kq_video"]
            ids = [m["frame_idx"] for m in moments]
            got_ids, imgs = tkmod.read_frames(tk_video, ids)
            pic = dict(zip(got_ids, imgs))
            dong_tk = tkmod.submission_row(tk_video, moments)
            st.code(", ".join(str(x) for x in dong_tk), language=None)
            if ids != sorted(ids):
                st.warning("⚠️ Kết quả KHÔNG tăng dần theo thời gian — hai mốc sát "
                           "nhau có thể đã trùng vùng. Soi lại bằng mắt trước khi nộp.")
            for start in range(0, len(moments), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, (col, m) in enumerate(
                        zip(cols, moments[start:start + cols_per_row]), start):
                    with col:
                        im = pic.get(m["frame_idx"])
                        if im is not None:
                            st.image(im, width="stretch")
                        st.caption(f"frame **{m['frame_idx']}** · điểm {m['score']:.3f}"
                                   f"\n\nneo {m['anchor']} "
                                   f"({m['frame_idx'] - m['anchor']:+d})"
                                   f"\n\n_{goc[j]}_")
            st.caption("Điểm thấp = có thể căn sai; soi kỹ mốc đó trước khi nộp.")

            # Thể lệ cho TRAKE tối đa 100 dòng y như KIS, và ví dụ trong thể lệ
            # có sẵn nhiều dòng cùng một video lệch nhau vài frame.
            xep = [v for v, _, _ in xh]
            tk_kho = st.session_state.get("tk_kho", {})
            da_can = sorted(tk_kho, key=lambda v: xep.index(v) if v in xep else 99)
            # Video đang xem đứng đầu, các video đã căn khác nối sau.
            da_can = [tk_video] + [v for v in da_can if v != tk_video]
            cap = [(v, tk_kho[v]) for v in da_can if v in tk_kho]
            dong_nop = [nopbai.dong_trake(v, f)
                        for v, *f in tkmod.submission_rows(cap, limit=100)]

            st.markdown(f"**Sẽ nộp {len(dong_nop)} dòng** "
                        f"từ {len(cap)} video đã căn.")
            st.caption(
                "Dòng 1 là chuỗi trên. Các dòng sau lấy từ **chính bảng điểm** "
                "hệ thống đã tính khi căn (~37 khung mỗi mốc) nên KHÔNG tốn thêm "
                "một lần mã hoá nào, và thể lệ không phạt dòng sai. Muốn thêm "
                "video dự phòng: đổi ô *Video sẽ nộp* rồi bấm **Căn chuỗi** lần "
                "nữa — kết quả cũ vẫn được giữ.")
            with st.expander(f"Soi {len(dong_nop)} dòng"):
                st.code("\n".join(dong_nop), language=None)
            o_luu("query-3-trake", "trake", lambda: dong_nop)


# ═════════════════════════════ NỘP BÀI ═════════════════════════════
with tab_nop:
    st.caption(
        "Soi lại **đúng thứ sẽ nộp** trước khi tải về. Mỗi gói chỉ được nộp "
        "**3 lần** và **lần cuối cùng** mới được tính điểm — nộp sai định dạng "
        "vẫn trừ một lượt, nên chỗ này đáng nhìn kỹ.")

    if not kho():
        st.info("Kho đang trống. Tìm ở tab KIS / Q&A / TRAKE rồi bấm "
                "**💾 Lưu vào kho**.")
    else:
        chon = st.selectbox("File đang soi", sorted(kho()), key="nop_chon")
        dong = kho()[chon]
        loi = nopbai.kiem_tep(chon, dong)

        if loi:
            st.error("**Còn lỗi, chưa nộp được:**\n"
                     + "\n".join(f"- {e}" for e in loi))
        else:
            st.success(f"`{chon}` · {len(dong)}/100 dòng · đúng định dạng thể lệ")

        h1, h2, h3 = st.columns([2, 1, 1])
        ten_moi = h1.text_input(
            "Đổi tên file", chon, key=f"ten_moi_{chon}",
            help="Phải trùng tên file truy vấn BTC phát, chỉ đổi .txt thành .csv.")
        h2.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
        if h2.button("Đổi tên", width="stretch") and ten_moi.strip() != chon:
            kho().pop(chon)
            luu_kho(ten_moi, dong)
            st.rerun()
        h3.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
        if h3.button("🗑 Xoá file", width="stretch"):
            kho().pop(chon)
            st.rerun()

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("##### ✏️ Sửa trực tiếp")
            st.caption("Mỗi dòng một bản ghi. Đây là **văn bản thật** sẽ nằm "
                       "trong .csv — sửa xong nhớ bấm Áp dụng.")
            moi = st.text_area(
                "nội dung", "\n".join(dong), height=430,
                key=f"sua_{chon}", label_visibility="collapsed")
            a1, a2 = st.columns(2)
            if a1.button("✅ Áp dụng sửa", type="primary", width="stretch"):
                kho()[chon] = [x for x in moi.split("\n") if x.strip()]
                st.rerun()
            a2.download_button("⬇ Tải .csv này", nopbai.mot_tep(dong),
                               file_name=chon, mime="text/csv", width="stretch")

        with c2:
            st.markdown("##### 👁 Máy chấm sẽ đọc ra thế này")
            st.caption("Tách trường bằng chính `csv.reader`, không phải bằng mắt. "
                       "Cột lệch hoặc ô trống là dấu hiệu ngoặc kép sai.")
            bang = [next(csv.reader(io.StringIO(x)), []) for x in dong]
            rong = max((len(r) for r in bang), default=0)
            bang = [r + [""] * (rong - len(r)) for r in bang]
            dang = nopbai.dang_cua(chon)
            if dang == "kis" and rong == 2:
                cot = ["video_id", "frame_idx"]
            elif dang == "qa" and rong == 3:
                cot = ["video_id", "frame_idx", "answer"]
            elif dang == "trake" and rong >= 2:
                cot = ["video_id"] + [f"mốc {i}" for i in range(1, rong)]
            else:
                cot = [f"trường {i + 1}" for i in range(rong)]
            xem = pd.DataFrame(bang, columns=cot)
            xem.index = range(1, len(xem) + 1)
            st.dataframe(xem, height=430, width="stretch")
