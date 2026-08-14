"""Giao diện thi: ba tab cho ba dạng đề KIS / Q&A / TRAKE.

    streamlit run ui/search_ui.py

Cần artifact do src/prepare_data.py dựng trong data/processed_hcmc2026/.
Cơ sở đo của các giá trị mặc định: docs/bao_cao_he_thong.tex, mục "Giao diện thi".
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import corpus  # noqa: E402  (nạp metadata + sửa đường dẫn ảnh cho đúng máy)
import ensemble  # noqa: E402  (encoder phụ, cộng điểm trên toàn corpus)
import rerank  # noqa: E402  (tầng re-rank, xem src/rerank.py)

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
    """Chuẩn bị câu để đưa vào encoder. Trả (câu dùng, ghi chú hiển thị).

    Ô nhập chính là TIẾNG VIỆT vì SigLIP2 đọc thẳng được. Dịch hỏng thì lùi về
    câu gốc — đường lùi êm, không về 0 như thời ViT-B-32.
    """
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


def search(query: str, top_k: int, ens_w: float = 0.0) -> pd.DataFrame:
    """`query` đã qua preprocess_query.

    `ens_w > 0` thì bỏ FAISS và tự nhân ma trận: FAISS chỉ trả top-k của encoder
    chính, mà thứ ta cần chính là những khung encoder chính bỏ sót.
    """
    meta = load_metadata()
    index = load_index()
    qvec = encode_text(query)

    if qvec.shape[1] != index.d:
        st.error(f"Encoder dim {qvec.shape[1]} != index dim {index.d}. "
                 f"Wrong CLIP variant — check which model BTC used.")
        st.stop()

    if ens_w > 0:
        chinh = load_features_ram() @ qvec[0]
        tong = ensemble.ghep(chinh, query, w=ens_w)
        top = np.argsort(-tong, kind="stable")[:top_k]
        scores, ids = tong[None, top], top[None, :]
    else:
        scores, ids = index.search(qvec, top_k)
    hits = meta.iloc[ids[0]].copy()
    hits["score"] = scores[0]
    # Giữ số hàng GỐC: vừa là chỉ số vector, vừa là thứ nút "tìm ảnh tương tự"
    # cần. reset_index(drop=True) làm mất nó.
    hits["row_id"] = ids[0]
    return hits.reset_index(drop=True)


def diversify_by_video(hits: pd.DataFrame, per_video: int) -> pd.DataFrame:
    """Giữ tối đa `per_video` khung mỗi video.

    ⚠️ Làm TỤT điểm chấm tự động — đừng bật cho danh sách nộp. Hữu ích cho người
    ngồi tìm thì ngược lại.
    """
    return (hits.groupby("video_id", sort=False)
                .head(per_video)
                .reset_index(drop=True))


@st.cache_resource(show_spinner="Loading features ...")
def load_features():
    """Vector ảnh, dùng cho tìm-bằng-ảnh và tìm-trong-video."""
    return np.load(PROCESSED / "features.npy", mmap_mode="r")


@st.cache_resource(show_spinner="Nạp vector vào RAM cho ensemble ...")
def load_features_ram():
    """Bản nằm hẳn trong RAM (726 MB), dùng cho ensemble.

    Ensemble nhân với CẢ 177k khung mỗi truy vấn nên mmap là chậm không chấp nhận
    được. Tìm-bằng-ảnh vẫn dùng mmap vì nó chỉ đụng vài nghìn hàng.
    """
    return np.load(PROCESSED / "features.npy")


def search_by_image(row_id: int, top_k: int, same_video_only: bool) -> pd.DataFrame:
    """Tìm khung giống khung `row_id`. Vector có sẵn nên gần như miễn phí.

    Bấm đại một ảnh rồi tìm ảnh giống là VÔ ÍCH (láng giềng của ảnh sai thì cũng
    sai): 0.2123 so với 0.8000 khi đã khoanh đúng video. Nên `same_video_only`
    mặc định BẬT.
    """
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
    """Gộp kết quả về mức VIDEO, giữ khung điểm cao nhất làm đại diện.

    CLIP tìm đúng VIDEO 0.988 nhưng đúng KHUNG chỉ 0.654 — nó gần như luôn đúng
    video, chỉ không biết đúng giây, mà giây thì người liếc qua là thấy.
    """
    g = (hits.groupby("video_id", sort=False)
             .agg(n_hit=("score", "size"), score=("score", "max"),
                  row=("score", "idxmax"))
             .sort_values("score", ascending=False)
             .reset_index())
    return g.join(hits.loc[g["row"], ["n", "pts_time", "frame_idx", "image_path"]]
                      .reset_index(drop=True))


def all_frames_of(video_id: str) -> pd.DataFrame:
    meta = load_metadata()
    return meta[meta["video_id"] == video_id].copy()



def apply_rerank(hits: pd.DataFrame, query_vi: str, query_en: str,
                 depth: int, use_encoder: bool,
                 deep: int = 0) -> tuple[pd.DataFrame, np.ndarray, bool]:
    """Chấm lại `depth` kết quả đầu bằng VLM (+ encoder thứ hai nếu bật).

    `deep > 0` bật đào sâu thích ứng: chỉ chấm tiếp tới `deep` khi rổ đáng ngờ.
    Trả (hits đã xếp lại, điểm VLM, có đào sâu hay không); hits thêm cột `vlm`,
    `rr`, phần đuôi chưa chấm mang vlm = NaN.
    """
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
    # stable: rất nhiều ảnh hoà điểm VLM (trung bình 9,8 ảnh cùng 10đ trong rổ
    # 100) — khi hoà thì phải giữ nguyên thứ tự CLIP, đừng để sort xáo lung tung.
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
    img = hit.get("image_path", "")
    if img and Path(img).exists():
        st.image(img, width="stretch")
    else:
        st.markdown(":grey_background[no image]")

    # ⚠️ frame_id BTC chấm là `frame_idx` (= pts_time×fps), KHÔNG phải `n`
    # (số thứ tự keyframe). Hai số khác nhau ở mọi hàng.
    st.code(f"{hit['video_id']}, {int(hit['frame_idx'])}", language=None)

    v = hit.get("vlm", np.nan)
    badge = "" if pd.isna(v) else f" · :orange[VLM {v:.0f}/10]"
    sc = hit.get("score", np.nan)
    sc_txt = "" if pd.isna(sc) else f" · score={sc:.3f}"
    st.caption(f"t={hit['pts_time']:.1f}s · n={int(hit['n'])}{sc_txt}{badge}")

    if show_similar_button and "row_id" in hit:
        # key phải duy nhất trong cả trang, dùng row_id vì nó là số hàng toàn cục
        if st.button("Khung tương tự", key=f"sim_{int(hit['row_id'])}",
                     help="Tìm khung giống khung này TRONG CÙNG VIDEO. "
                          "Đo được: giới hạn trong video cho FINAL 0.800, "
                          "còn tìm toàn corpus chỉ 0.678."):
            st.session_state["seed_row"] = int(hit["row_id"])
            # Bắt buộc rerun: panel kết quả ở ĐẦU trang, đã vẽ xong trước khi nút ở
            # cuối trang được bấm. Không rerun thì phải bấm hai lần mới thấy.
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
                img = v.get("image_path", "")
                if img and Path(img).exists():
                    st.image(img, width="stretch")
                st.caption(f"**{v['video_id']}** · {int(v['n_hit'])} khung khớp\n\n"
                           f"tốt nhất n={int(v['n'])} · t={v['pts_time']:.1f}s "
                           f"· score={v['score']:.3f}")
                if st.button("Mở", key=f"open_{v['video_id']}"):
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
        # Khung không lọt rổ để TRỐNG chứ không điền 0: trống = "chưa chấm",
        # 0 = "chấm rồi, không liên quan" — hai chuyện khác nhau.
        frames["score"] = frames["row_id"].map(hits.set_index("row_id")["score"])
        st.caption(f"{len(frames)} khung, xếp theo thời gian. "
                   f"Khung có điểm là khung đã lọt vào kết quả tìm kiếm.")
        show_frames(frames.sort_values("n"), ncol, similar=False)


st.title("RED-NEUTRONS — AIC 2026")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K results", 10, 200, 50, step=10)
    # Mặc định 20 = tắt hẳn. Xem docstring diversify_by_video: mọi mức < 20 đều
    # làm tụt điểm KIS. Chỉ hạ xuống khi đang DÒ dữ liệu chứ không phải khi thi.
    per_video = st.slider("Max keyframes per video", 1, 20, 20,
                          help="20 = tắt. Hạ xuống chỉ để dò dữ liệu — đo được là "
                               "làm tụt điểm KIS (5→0.4667, 3→0.4346 so với 0.4765)")
    cols_per_row = st.slider("Grid columns", 3, 8, 5)

    st.divider()
    st.subheader("Cách hiển thị")
    view = st.radio(
        "Chế độ", ["Theo video (nên dùng)", "Theo khung"], index=0,
        help="Đo trên 81 query: đúng KHUNG trong top-100 chỉ 0.654, nhưng đúng "
             "VIDEO tới 0.988 (80/81). L23: khung 0/8 → video 8/8. CLIP gần như "
             "luôn tìm đúng video, chỉ không biết đúng giây — mà giây thì người "
             "liếc qua là thấy.")
    video_mode = view.startswith("Theo video")

    st.divider()
    st.subheader("Re-rank")
    st.caption(f"Đang tìm bằng **{CLIP_MODEL}** — nền {BASELINE:.4f} trên 81 query "
               f"(bộ ViT-B-32 của BTC chỉ 0.4765)")
    use_vlm = st.checkbox("Bật VLM re-rank (Gemini)", value=True)
    rerank_depth = st.select_slider("Chấm lại bao nhiêu kết quả đầu",
                                    options=[20, 50, 100], value=20,
                                    help="20 ảnh ≈ 7 giây · 100 ảnh ≈ 35 giây")
    # Mặc định TẮT vì lỗ ở danh sách nộp xếp tự động (nhóm dễ chiếm 84% và bị
    # phá). Vẫn giữ vì với NGƯỜI thì đào sâu không bao giờ hại.
    auto_deep = st.checkbox(
        "Tự đào sâu khi rổ đáng ngờ", value=False,
        help="Chấm rổ nông trước; nếu điểm 20 ứng viên đầu quá sát nhau thì chấm "
             "tiếp tới rổ sâu. Đo được là KHÔNG cải thiện điểm tự động (0.7901 so "
             "với 0.7951 khi chỉ chấm rổ 20) — nhưng hữu ích khi bạn đang tự tìm.")
    deep_to = st.select_slider("Đào sâu tới", options=[100, 200, 300], value=200,
                               disabled=not auto_deep)
    # Chấm lại bằng encoder thứ hai TRÊN RỔ đã bỏ: ~50 giây CPU mà chỉ xáo
    # trong rổ. "Ghép encoder" dưới đây dùng vector sẵn, chấm cả 177k khung.
    use_encoder = False

    st.divider()
    st.subheader("Ghép encoder thứ hai")
    if not ensemble.san_sang():
        ens_w = 0.0
        st.caption(f"⚠️ Chưa có `{ensemble.PHU_FEATURES.name}` trong "
                   f"`data/kernel_out/` — xem README mục cài đặt.")
    else:
        # Kéo được khung MỚI vào rổ nên đổi được cả R@100 (0.951 → 0.975).
        # Cả dải 0.2-1.5 đều vượt nền; 0.5 chỉ tình cờ là điểm may nhất.
        ens_w = st.select_slider(
            "Trọng số encoder phụ", options=[0.0, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5],
            value=ensemble.W_MAC_DINH,
            help="0 = tắt. Cộng z(SigLIP2-L-512) + w·z(SigLIP2-B16) trên cả 177k khung. "
                 "FINAL: tắt 0.7901 · 0.2→0.8049 · 0.5→0.8148 · 0.7→0.8074 · 1.0→0.8025. "
                 "Chậm hơn FAISS vài trăm ms vì phải nhân ma trận đầy đủ.")

tab_kis, tab_qa, tab_trake = st.tabs(
    ["KIS — tìm khoảnh khắc", "Q&A — trả lời câu hỏi", "TRAKE — chuỗi khoảnh khắc"])

with tab_kis:
    query_vi = st.text_input(
        "Câu truy vấn (tiếng Việt)",
        # Placeholder phải tả cảnh CÓ THẬT trong corpus, nếu không hệ thống trả về
        # hình hiệu đầu chương trình và người dùng tưởng nó hỏng.
        placeholder="vd: người phụ nữ đội nón lá đang hái dứa ngoài ruộng",
        help="Gõ thẳng tiếng Việt như đề thi phát ra. SigLIP2 đọc được tiếng Việt "
             "(0.7037 trên bộ eval); encoder ViT-B-32 cũ thì cho 0.0000 tuyệt đối.")

    col_a, col_b = st.columns([1, 2])
    tr_mode = col_a.selectbox(
        "Xử lý câu", ["dịch sang tiếng Anh", "để nguyên tiếng Việt"], index=0,
        help="Đo được: dịch 0.7210 · để nguyên 0.7037. Dịch hơn +0.017 nhưng tốn "
             "~0,25 s và phụ thuộc mạng; hỏng thì tự lùi về câu gốc.")
    query_en_override = col_b.text_input(
        "Hoặc tự viết câu tiếng Anh (để trống nếu không cần)",
        placeholder="women in ao dai posing in a lotus field",
        help="Người viết tay tả sát hình hơn máy dịch (0.7802 so với 0.7210). Dùng "
             "khi muốn diễn đạt lại theo cách khác.")

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
        # Encoder ăn bản dịch, còn VLM luôn đọc TIẾNG VIỆT GỐC — nó hiểu tiếng Việt
        # tốt và bản dịch chỉ làm mất chi tiết.
        if query_en_override.strip():
            query, note = query_en_override.strip(), "→ dùng câu tiếng Anh bạn tự viết"
        elif tr_mode.startswith("dịch"):
            with st.spinner("Đang dịch ..."):
                query, note = preprocess_query(query_vi, "google")
        else:
            query, note = preprocess_query(query_vi, "vi")
        if note:
            st.caption(note)

        # Lấy rổ đủ sâu để tầng re-rank còn chỗ kéo kết quả từ dưới lên. Khi bật đào
        # sâu thích ứng thì phải lấy sẵn tới `deep_to`, dù phần lớn query sẽ không dùng.
        depth = rerank_depth if use_vlm else 0
        deep = deep_to if (use_vlm and auto_deep) else 0
        # Lấy ÍT NHẤT 100 ứng viên: danh sách nộp được phép 100 dòng và R@k lấy
        # MAX trên k dòng đầu, nên mỗi dòng bỏ trống là một cơ hội bị vứt.
        hits = search(query, max(top_k, depth, deep, 100), ens_w=ens_w)

        vlm_scores = None
        if use_vlm and depth:
            try:
                hits, vlm_scores, _ = apply_rerank(hits, query_vi.strip() or query,
                                                   query, depth, use_encoder, deep)
            except Exception as exc:                       # thiếu API key, hết hạn mức...
                st.warning(f"Re-rank hỏng, hiển thị kết quả CLIP thuần — {exc}")

        # HAI danh sách, tối ưu NGƯỢC nhau — đừng gộp lại:
        #   nop  = thứ tự thô, đủ 100 dòng   -> cho MÁY CHẤM
        #   hits = đã đa dạng hoá, cắt top_k -> cho NGƯỜI XEM
        # Gộp lại từng làm bảng nộp còn 24 dòng, vứt trắng 76 chỗ.
        nop = hits.iloc[:100]
        hits = diversify_by_video(hits, per_video).iloc[:top_k]

        if vlm_scores is not None:
            if rerank.all_failed(vlm_scores):
                # Phân biệt rõ với trường hợp dưới: đây là lỗi ĐƯỜNG MẠNG, không
                # phải kết luận gì về dữ liệu. Gộp chung là đổ oan cho corpus.
                st.warning("VLM không chấm được ô nào (hết hạn mức hoặc mất mạng) — "
                           "đang hiển thị thứ tự CLIP thuần.")
            elif rerank.basket_uncertain(hits["score"].to_numpy(), shallow=rerank_depth):
                # Cò dựa trên BIÊN ĐỘ điểm truy xuất (AUC 0.907). Đừng khôi phục luật cũ
                # "đỉnh VLM < 10" — trên nền SigLIP2 nó chỉ còn đúng 31%.
                st.error("⚠️ Điểm của các ứng viên đầu bảng quá sát nhau — dấu hiệu "
                         "**đáp án có thể không nằm trong rổ này**. Nên đổi cách diễn "
                         "đạt câu truy vấn, tăng độ sâu, hoặc tìm bằng ảnh.")

        st.caption(f"{len(hits)} keyframes from {hits['video_id'].nunique()} videos")

        # ── bảng nộp ────────────────────────────────────────────────────────
        # R@k lấy MAX trên k dòng đầu ⇒ thêm dòng không bao giờ làm mất điểm đã có.
        # Bỏ trống chỗ trong 100 dòng là tự vứt cơ hội.
        with st.expander(f"📋 Bảng nộp — {len(nop)} dòng", expanded=False):
            nop = nop[["video_id", "frame_idx"]].copy()
            nop.insert(0, "hạng", range(1, len(nop) + 1))
            txt = "\n".join(f"{r.video_id}, {int(r.frame_idx)}"
                            for r in nop.itertuples())
            c1, c2 = st.columns([2, 3])
            with c1:
                st.download_button("Tải .csv", txt, file_name="submission.csv",
                                   mime="text/csv")
                st.caption(
                    "Nộp đủ 100 dòng: R@k lấy MAX trên k dòng đầu nên dòng thêm "
                    "chỉ có thể được điểm, không bao giờ mất.\n\n"
                    "⚠️ Cột thứ hai là **frame_idx** (khung thật trong video), "
                    "KHÔNG phải `n` (số thứ tự keyframe).")
            with c2:
                st.code(txt, language=None)

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

    if st.button("Tìm & trả lời", key="qa_go", type="primary"):
        if not qa_desc.strip() or not qa_ques.strip():
            st.warning("Cần cả mô tả lẫn câu hỏi.")
        else:
            import qa as qamod
            hits = search(qa_desc.strip(), max(qa_top, 100), ens_w=ens_w)
            with st.spinner("Đang hỏi VLM ..."):
                got = qamod.answer_over_hits(qa_ques.strip(), hits,
                                             desc=qa_desc.strip(), top=qa_top)
            if got is None:
                st.error("Không gọi được VLM (hết hạn mức hoặc mất mạng). Dưới đây "
                         "là kết quả tìm kiếm thuần — tự đọc ảnh và tự trả lời.")
            else:
                c1, c2 = st.columns([1, 2])
                with c1:
                    if Path(got["image_path"]).exists():
                        st.image(got["image_path"], width="stretch")
                with c2:
                    st.subheader(got["answer"] or "(VLM không trả lời được)")
                    st.caption(f"mức chắc chắn {got['confidence']:.0f}/10 · {got['reason']}")
                    st.code(f"{got['video_id']}, {got['frame_idx']}, {got['answer']}",
                            language=None)
                    # Khung NỘP theo CLIP top-1, không phải khung VLM chọn: VLM chọn ảnh nào
                    # nó TRẢ LỜI ĐƯỢC, kể cả ảnh ở video khác.
                    if got["vlm_frame_idx"] != got["frame_idx"]:
                        st.warning(
                            f"VLM đọc đáp án từ **{got['vlm_video_id']}** frame "
                            f"{got['vlm_frame_idx']}, khác khung đang nộp "
                            f"({got['video_id']} frame {got['frame_idx']}). "
                            "Hai chỗ lệch nhau là dấu hiệu nên soi lại bằng mắt.")
                # Nộp đủ 100 dòng vì R@k lấy MAX. Câu trả lời dùng chung cho mọi dòng —
                # đáp án không phụ thuộc chọn khung nào trong cùng một cảnh.
                with st.expander("📋 Bảng nộp Q&A — 100 dòng", expanded=False):
                    rows = qamod.submission_rows(hits, got, limit=100)
                    txt = "\n".join(f"{v}, {f}, {a}" for v, f, a in rows)
                    c1, c2 = st.columns([2, 3])
                    c1.download_button("Tải .csv", txt, file_name="submission_qa.csv",
                                       mime="text/csv")
                    c1.caption("Sai câu trả lời là 0 điểm dù đúng khung — kiểm "
                               "câu trả lời bằng mắt trước khi nộp.")
                    c2.code(txt[:1500] + ("\n..." if len(txt) > 1500 else ""),
                            language=None)
                st.divider()
            st.caption(f"{qa_top} ứng viên đầu — ảnh đầu tiên là khung sẽ nộp")
            show_frames(hits.iloc[:qa_top], cols_per_row, similar=False)
    else:
        st.info("Nhập mô tả và câu hỏi rồi bấm **Tìm & trả lời**.")


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
    texts = [s.strip() for s in tk_moments.split("\n") if s.strip()]

    if not texts:
        st.info("Nhập các mốc, mỗi dòng một mốc.")
    else:
        # ── giai đoạn 1a: khoanh VIDEO ───────────────────────────────────
        # Chấm bằng TỔNG điểm của đường đi ĐÃ ÉP THỨ TỰ thời gian, không phải
        # điểm cao nhất của một mốc bất kỳ: luật cũ cho một video thắng chỉ nhờ
        # tình cờ chứa MỘT khung giống MỘT mốc. Đo được: 4/5 -> 5/5 chuỗi đúng
        # video, điểm TRAKE đầu-cuối 0.3467 -> 0.3967. Sai video là 0 điểm cho
        # CẢ chuỗi nên đây là chỗ đáng đầu tư nhất của TRAKE.
        import trake as tkmod
        with st.spinner("Đang xếp hạng video theo cả chuỗi ..."):
            # Dựng ma trận điểm bằng encoder ĐÃ NẠP SẴN của giao diện. Gọi
            # tkmod.rank_videos(texts) sẽ nạp bản SigLIP2-L thứ hai (~1,7 GB) và
            # làm cạn bộ nhớ ảo — đã dính thật.
            F = load_features_ram()
            S = np.vstack([F @ encode_text(t)[0] for t in texts])
            xh = tkmod.rank_videos_scores(S, top_k=10)
        best = pd.Series(dict(xh))
        # Ảnh đại diện: khung khớp nhất trong video đó, lấy từ rổ tìm kiếm.
        pool = pd.concat([search(t, 100, ens_w=ens_w) for t in texts],
                         ignore_index=True)
        rep = (pool.sort_values("score", ascending=False)
                   .drop_duplicates("video_id").set_index("video_id"))

        st.subheader("1. Chọn video")
        st.caption("Xếp theo tổng điểm của chuỗi khi bị ép đúng thứ tự thời gian — "
                   "video phải chứa được CẢ chuỗi, không chỉ một mốc.")
        for start in range(0, len(best), cols_per_row):
            cols = st.columns(cols_per_row)
            for col, vid in zip(cols, best.index[start:start + cols_per_row]):
                with col:
                    # Video do luật mới đưa lên có thể không nằm trong rổ top-100
                    # của bất kỳ mốc nào, nên rep có thể thiếu nó.
                    p = rep.loc[vid, "image_path"] if vid in rep.index else ""
                    if p and Path(p).exists():
                        st.image(p, width="stretch")
                    st.caption(f"**{vid}** · {best[vid]:.3f}")
        tk_video = st.selectbox("Video sẽ nộp", list(best.index), key="tk_video")

        st.subheader("2. Căn từng mốc trên video gốc")
        st.caption(f"{len(texts)} mốc × ~13 giây mã hoá ≈ **{len(texts) * 13} giây** "
                   f"(38 lần mã hoá mỗi mốc, bán kính dò ±60 frame).")
        if st.button("Căn chuỗi", key="tk_go", type="primary"):
            moments = None
            try:
                with st.spinner(f"Đang căn {len(texts)} mốc trên {tk_video} ..."):
                    moments = tkmod.locate_sequence(tk_video, texts)
            except FileNotFoundError as exc:
                st.error(str(exc))
            if moments:
                ids = [m["frame_idx"] for m in moments]
                # Ghép theo ID THẬT đọc được, không theo sorted(set(ids)): read_frames bỏ
                # frame hỏng nên zip sẽ gán nhầm ảnh sang mốc khác.
                got_ids, imgs = tkmod.read_frames(tk_video, ids)
                pic = dict(zip(got_ids, imgs))
                st.code(", ".join(str(x) for x in tkmod.submission_row(tk_video, moments)),
                        language=None)
                if ids != sorted(ids):
                    # Không nên xảy ra vì mỏ neo đã bị ép thứ tự, nhưng căn tinh
                    # còn dịch ±120 frame nên hai mốc sát nhau vẫn có thể chồng lên.
                    st.warning("⚠️ Kết quả KHÔNG tăng dần theo thời gian — hai mốc sát "
                               "nhau có thể đã trùng vùng. Soi lại bằng mắt trước khi nộp.")
                for start in range(0, len(moments), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for col, m in zip(cols, moments[start:start + cols_per_row]):
                        with col:
                            im = pic.get(m["frame_idx"])
                            if im is not None:
                                st.image(im, width="stretch")
                            st.caption(f"frame **{m['frame_idx']}** · điểm {m['score']:.3f}"
                                       f"\n\nneo {m['anchor']} "
                                       f"({m['frame_idx'] - m['anchor']:+d})"
                                       f"\n\n_{m['text']}_")
                st.caption("Điểm thấp = có thể căn sai; soi kỹ mốc đó trước khi nộp.")
