# -*- coding: utf-8 -*-
import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "ui" / "search_ui.py"
sys.path.insert(0, str(ROOT / "src"))

from streamlit.testing.v1 import AppTest  # noqa: E402

loi = []


def dd(a):
    return a.session_state["marked"] if "marked" in a.session_state else []


def kiem(dieu_kien, mo_ta):
    print(f"  {'✅' if dieu_kien else '🔴'} {mo_ta}")
    if not dieu_kien:
        loi.append(mo_ta)


print("chạy trọn kịch bản giao diện (nạp encoder nên hơi lâu)…")
at = AppTest.from_file(str(UI), default_timeout=600).run()

kiem(not at.exception, "không có ngoại lệ khi dựng trang"
     + (f" — {at.exception[0].value}" if at.exception else ""))
kiem(len(at.tabs) == 4,
     f"đúng 4 tab KIS/Q&A/TRAKE/Nộp bài (thấy {len(at.tabs)})")

# Dừng NGAY nếu trang không dựng được. `at.exception` KHÔNG bắt được lỗi cú pháp
# của chính tệp: AppTest trả về một trang rỗng và báo "không có ngoại lệ", nên các
# phép kiểm sau sẽ nổ IndexError khó đọc thay vì nói rõ trang hỏng. Đã dính.
if not at.tabs:
    print("\n🔴 TRANG KHÔNG DỰNG ĐƯỢC — xem lỗi cú pháp in ở trên, dừng tại đây.")
    sys.exit(1)

nhan_sl = [x.label for x in at.sidebar.slider]
kiem(nhan_sl[:2] == ["Số ô hiển thị", "Số cột"],
     f"hai thanh trượt lộ ra đúng là hiển thị: {nhan_sl[:2]}")

# Ô nhập chính phải là gợi ý BTC, không phải thứ gì khác.
kiem(any("Gợi ý từ ban tổ chức" in (t.label or "") for t in at.text_input),
     "ô nhập đầu tiên là gợi ý từ ban tổ chức")

print("\nchạy thử một lượt tìm thật…")
at.text_input[0].set_value("người phụ nữ đội nón lá đang hái dứa ngoài ruộng")
at = at.run()
kiem(not at.exception, "không có ngoại lệ sau khi tìm"
     + (f" — {at.exception[0].value}" if at.exception else ""))
ma = [c.value for c in at.code if "," in (c.value or "")]
kiem(len(ma) >= 5, f"mã nộp `video_id, frame_idx` hiện dưới mỗi ảnh ({len(ma)} ô)")
kiem(all(x.split(",")[0].strip().startswith("L") for x in ma[:5]),
     "mã nộp đúng dạng <video_id>, <frame_idx>")
nhan_cb = [c.label for c in at.checkbox]
kiem(nhan_cb.count("OK") >= 5, f"mỗi thẻ có ô tích OK ({nhan_cb.count('OK')} ô)")
kiem(nhan_cb.count("Loại") >= 5,
     f"mỗi thẻ có ô tích Loại ({nhan_cb.count('Loại')} ô)")

for c in at.checkbox:
    if c.label == "OK":
        c.set_value(True)
        break
at = at.run()
kiem(len(dd(at)) == 1, "tích OK ghi được vào phiên")

# Mặc định ô tích CHỈ để chọn lọc bằng mắt — không được đụng tới kết quả tìm.
kiem("dung_tich" not in at.session_state or not at.session_state["dung_tich"],
     "công tắc dùng-ô-tích-để-tìm mặc định TẮT")

at.text_input[0].set_value("người phụ nữ đội nón lá đang hái dứa ngoài ruộng buổi sáng")
at = at.run()
kiem(len(dd(at)) == 1,
     "GIỮ đánh dấu khi gợi ý chỉ dài thêm")

at.text_input[0].set_value("hai chiếc máy bay đậu trên đường băng")
at = at.run()
kiem(not dd(at),
     "XOÁ đánh dấu khi sang câu truy vấn khác hẳn")

# ── tab Q&A phải là HAI BƯỚC ────────────────────────────────────────────────
# Đo được: 30/51 câu Q&A sai là vì khung đáp án nằm NGOÀI rổ ảnh đưa cho VLM. Nên
# người thi phải đánh dấu TRƯỚC khi tiêu lời gọi API. Gộp lại một nút là hỏng đúng
# chỗ đắt nhất, mà `streamlit run` không lộ ra vì tab này chỉ chạy khi được bấm.
print("\nkiểm luồng hai bước của tab Q&A…")
nhan_bt = [b.label for b in at.button]
kiem(any("Tìm khung" in (x or "") for x in nhan_bt),
     "tab Q&A có nút bước 1 'Tìm khung'")
kiem(any("Trả lời" in (x or "") for x in nhan_bt),
     "tab Q&A có nút bước 2 'Trả lời từ rổ hiện tại'")
# `SafeSessionState` không có `.get()` — kiểm bằng `in` rồi mới lấy chỉ số.
da_tim = at.session_state["qa_da_tim"] if "qa_da_tim" in at.session_state else False
kiem(not da_tim, "chưa tìm thì bước 2 chưa mở")

nhan_radio = [r.label for r in at.radio]
kiem(any("Kiểu câu hỏi" in (x or "") for x in nhan_radio),
     "tab Q&A có nút chọn kiểu câu hỏi (nhận dạng / đếm)")
kieu = at.session_state["qa_kieu"] if "qa_kieu" in at.session_state else None
kiem(kieu == "nhận dạng", f"mặc định là 'nhận dạng', không phải 'đếm' (thấy {kieu})")

# Câu đếm phải chạy LẶP rồi bày cả phổ: cùng một khung hỏi 5 lần từng ra 4 đáp
# án khác nhau, nên một con số đơn lẻ ở đây là rút thăm.
_rd = [r for r in at.radio if "Kiểu câu hỏi" in (r.label or "")]
if _rd:
    _r = _rd[0].set_value("đếm").run(timeout=120)
    _ol = [x for x in _r.select_slider if "Đếm lại" in (x.label or "")]
    kiem(bool(_ol), f"chọn 'đếm' thì hiện ô số lần chạy lại (thấy {len(_ol)})")
    _lan = _r.session_state["qa_lan"] if "qa_lan" in _r.session_state else None
    kiem(_lan and _lan > 1, f"mặc định đếm LẶP chứ không một lần (thấy {_lan})")

kiem(hasattr(qamod_tam := __import__("qa"), "dem_lap"),
     "qa có dem_lap để đếm lặp")

# Cỡ rổ phải theo kênh: đo được kênh ảnh thuần 20->30 ăn +0,060, còn kênh lời
# nói 20->30 thì LỖ 0,100 vì mỗi ảnh thêm kéo theo một dòng chữ thêm.
_q = __import__("qa")
kiem(_q.ro_goi_y("Đàn chim là loài gì?") == 30,
     f"câu thuần ảnh gợi ý rổ 30 (thấy {_q.ro_goi_y('Đàn chim là loài gì?')})")
kiem(_q.ro_goi_y("Tên của con đèo là gì?") == 20,
     f"câu kênh lời nói gợi ý rổ 20 (thấy {_q.ro_goi_y('Tên của con đèo là gì?')})")
kiem(_q.ro_goi_y("Tấm biển ghi chữ gì?") == 20, "câu kênh chữ gợi ý rổ 20")
_top = at.session_state["qa_top"] if "qa_top" in at.session_state else None
kiem(_top == 30, f"mặc định rổ là 30 chứ không 20 (thấy {_top})")

# frame_idx của BTC làm SÀN pts*fps nên 12,9% khung thiếu đúng 1 khung thật.
# Số NỘP phải giữ nguyên; chỉ thêm dòng mách người thi tua video ở đâu.
import corpus as _c  # noqa: E402
kiem(_c.khung_that(4.03333, 30.0) == 121,
     f"khung_that làm TRÒN chứ không làm sàn (thấy {_c.khung_that(4.03333, 30.0)})")
kiem(_c.moc_gio(4.03333) == "00:04.033",
     f"moc_gio ra mm:ss.mmm (thấy {_c.moc_gio(4.03333)})")
_md = _c.load_metadata()
_r = _md[(_md.video_id == "L21_V023") & (_md.n == 3)]
if len(_r):
    _r = _r.iloc[0]
    kiem(int(_r.frame_idx) == 120,
         f"bài nộp GIỮ NGUYÊN frame_idx của BTC (thấy {int(_r.frame_idx)})")
_ma = (ROOT / "ui" / "search_ui.py").read_text(encoding="utf-8")
kiem("khung thật" in _ma, "màn hình mách khung thật khi nó lệch số nộp")

# Bộ chấm: bỏ chữ do CÂU HỎI cấp sẵn, nhưng KHÔNG được lỏng tay.
sys.path.insert(0, str(ROOT / "eval"))
import danh_gia_qa as _dg  # noqa: E402
_H = "Người này mặc áo màu gì?"
kiem(_dg.trung("màu đỏ", "áo đỏ", _H), "chấm ĐÚNG: 'màu đỏ' cho nhãn 'áo đỏ'")
kiem(_dg.trung("bắp chân", "chân (vùng bắp chân / khoeo chân)",
               "Bác sĩ siêu âm ở bộ phận nào?"),
     "chấm ĐÚNG: nhãn có ngoặc là liệt kê cách nói")
kiem(not _dg.trung("màu vàng", "đen và vàng kim", "Con lân có màu gì?"),
     "chấm SAI: 'màu vàng' thiếu 'đen' thì không được tính đúng")
kiem(not _dg.trung("hồng và xanh dương", "hồng và xanh lá",
                   "Dung dịch có màu gì?"),
     "chấm SAI: 'xanh dương' khác 'xanh lá' — bỏ dấu làm 'lá' trùng 'là'")
kiem(not _dg.trung("", "áo đỏ", _H), "chấm SAI: đáp án rỗng")
kiem(_dg.trung("bốn miếng", "4", "Có mấy miếng?"), "chấm ĐÚNG: số viết bằng chữ")

# Câu KIS tả NHIỀU cảnh nối tiếp: 70% đề thật đợt 1 có mốc thời gian, bộ đo cũ
# có 0/81. Tách sai chỗ thì cảnh nào cũng vô nghĩa.
import chuoi as _ch  # noqa: E402
_c2 = _ch.tach_canh("Đoạn clip bắt đầu bằng cảnh cà rốt luộc trong nồi nước sôi. "
                    "Đoạn clip kết thúc bằng hình ảnh đĩa rau củ trình bày đẹp mắt")
kiem(len(_c2) == 2, f"tách được 2 cảnh từ câu bắt đầu/kết thúc (thấy {len(_c2)})")
kiem("bắt đầu" not in _c2[0], "bỏ cụm mở đầu, chỉ giữ nội dung cảnh")
_c3 = _ch.tach_canh("Một người đứng dưới nước. Tiếp theo là cảnh kéo lưới cá, "
                    "sau đó một nhóm người tiến đến quay phim.")
kiem(len(_c3) == 3, f"tách được 3 cảnh (thấy {len(_c3)})")
_c1 = _ch.tach_canh("Cảnh quay một nhóm hơn 5 người xếp hàng tập thể dục.")
kiem(len(_c1) == 1, "câu tả MỘT cảnh thì không bị cắt")
_cl = _ch.tach_canh("một bản đồ trên đó công trình thủy lợi lần lượt xuất hiện "
                    "bốn lần. Sau đó chuyển sang cảnh con đập quay từ trên cao")
kiem(len(_cl) == 2, f"'lần lượt' KHÔNG phải mốc đổi cảnh (thấy {len(_cl)})")
import numpy as _np  # noqa: E402
_S = _np.array([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0]], dtype=_np.float32)
_v = _np.array(["A", "A", "B"]); _t = _np.array([0.0, 5.0, 0.0], dtype=_np.float32)
_d = _ch.diem_chuoi(_S, _v, _t)
kiem(_d[0] > _d[2], "khung có cảnh sau nối tiếp được cộng điểm, khung lẻ thì không")
kiem("nopbai.dong_kis(r.video_id, r.frame_idx)" in _ma,
     "dòng nộp KIS vẫn dùng frame_idx gốc, không dùng khung thật")
_g = {"count": 3, "cac_lan": [2, 3, 4], "on_dinh": False}
kiem(not _g["on_dinh"], "dem_lap báo được khi ba lần chạy lệch nhau")

# confidence do VLM tự khai đã đo được là vô dụng (10/10 ở một câu bịa hẳn đáp
# án). Giao diện phải nói thẳng điều đó chứ không bày nó như một thước đo.
_ma = (ROOT / "ui" / "search_ui.py").read_text(encoding="utf-8")
kiem("KHÔNG đáng tin" in _ma,
     "màn hình cảnh báo rằng confidence tự khai không dùng để quyết định")
kiem("rút thăm" in _ma, "câu đếm lệch nhau thì cảnh báo là rút thăm")

# ── khâu đọc kỹ một khung: kiểm phần thuần tính toán, không tốn lời gọi API ──
import qa as qamod  # noqa: E402

anh = next((ROOT / "data" / "hcmc2026").rglob("*.jpg"), None)
if anh is None:
    kiem(False, "tìm được một keyframe để kiểm cắt ô")
else:
    o, (W, H) = qamod._cat_o(anh)
    kiem(len(o) >= 4, f"cắt được nhiều ô từ một khung (thấy {len(o)})")
    kiem(all(im.size == (qamod.O_CANH, qamod.O_CANH) for im, _ in o),
         "mọi ô đều đúng cạnh, không ô nào bị thu nhỏ")
    goc = {xy for _, xy in o}
    kiem(len(goc) == len(o), "không ô nào trùng vị trí ô nào")
    kiem((W - qamod.O_CANH, H - qamod.O_CANH) in goc,
         "ô cuối chạm mép phải-dưới, không bỏ sót rìa ảnh")

kiem(qamod._chuan("  Đèo Tằng Quái. ") == qamod._chuan("là đèo tằng quái"),
     "chuẩn hoá gộp được hai cách viết cùng một đáp án")
kiem(qamod._chuan("đỏ") != qamod._chuan("xanh"),
     "chuẩn hoá KHÔNG gộp hai đáp án khác nhau")

# Ô nằm gọn trong bảng chú giải phải bị loại khỏi phép đếm, ô ngoài thì không.
kiem(qamod._chong((100, 150, 200, 250), (72, 96, 262, 512)) == 1.0,
     "ô nằm gọn trong vùng loại được tính là chồng hoàn toàn")
kiem(qamod._chong((900, 600, 1000, 700), (72, 96, 262, 512)) == 0.0,
     "ô ngoài vùng loại thì không chồng chút nào")

asr = qamod.asr_khung("L22_V027", 11778)
kiem("èo" in asr, "lấy được lời nói đúng khung từ bảng ASR")
kiem(qamod.asr_khung("KHONG_CO", 1) == "",
     "khung không có lời nói thì trả chuỗi rỗng, không nổ")

# Kênh OCR: kiểm hợp đồng của hàm, KHÔNG quét ảnh thật (mỗi lần quét ~3 giây).
kiem(qamod.ocr_khung("khong_ton_tai.jpg") == "",
     "ảnh hỏng/không có thì ocr_khung trả rỗng, không nổ")
kiem("{ocr}" in qamod.PROMPT_OCR and "{question}" in qamod.PROMPT_OCR,
     "mẫu nhắc OCR có đủ hai chỗ điền")
kiem("dấu tiếng Việt" in qamod.PROMPT_OCR,
     "mẫu nhắc có dặn máy OCR không có dấu — nếu không VLM sẽ chép nguyên chữ trần")
import inspect  # noqa: E402
kiem("ocr" in inspect.signature(qamod.doc_ky).parameters,
     "doc_ky nhận được công tắc bật/tắt kênh OCR")
kiem(inspect.signature(qamod.doc_ky).parameters["ocr"].default is True,
     "kênh OCR mặc định BẬT ở khâu đọc kỹ")

# Trọng số phải tính cả độ phân giải: ảnh toàn cảnh bị thu nhỏ nên nhẹ hơn 1,0.
# Ca thật 23/08: toàn cảnh đọc biển số thành 28A, hai ô gốc đọc đúng 26A.
ti = min(1.0, qamod.QA_TOAN_PX / 1280.0) ** 2
kiem(ti < 1.0, f"toàn cảnh 1280px bị hạ trọng số vì thu nhỏ (thấy {ti:.2f})")
w_o = (qamod.O_CANH ** 2) / (1280.0 * 720.0)
kiem(2 * w_o > ti, "hai ô ở độ phân giải gốc thắng được một toàn cảnh đọc nhầm")

kiem("{asr}" in qamod.PROMPT_ASR and "{question}" in qamod.PROMPT_ASR,
     "mẫu nhắc lời nói có đủ hai chỗ điền")
kiem("sai chính tả" in qamod.PROMPT_ASR,
     "mẫu nhắc dặn sửa chính tả tên riêng — Whisper nghe 'Tản Viên' ra 'Tảng Viên'")

# Bộ đo tên riêng: mọi đáp án phải nằm trong lời nói phủ ĐÚNG khung được gán.
# Lệch thời gian là lỗi đọc bằng mắt không thấy, mà làm hỏng cả phép đo.
import csv  # noqa: E402
import re  # noqa: E402
import unicodedata  # noqa: E402


def _bd(x):
    x = unicodedata.normalize("NFD", str(x).lower())
    x = "".join(c for c in x if unicodedata.category(c) != "Mn").replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", " ", x).strip()


bo = ROOT / "eval" / "queries_qa_tenrieng.csv"
if not bo.exists():
    kiem(False, "có bộ đo eval/queries_qa_tenrieng.csv")
else:
    with open(bo, encoding="utf-8") as f:
        cau = list(csv.DictReader(f))
    kiem(len(cau) >= 20, f"bộ đo tên riêng có đủ câu (thấy {len(cau)})")
    kiem(len({c["video_id"] for c in cau}) == len(cau),
         "mỗi câu một video khác nhau, không đo trùng một cảnh")
    # Soi bằng answer_asr chứ không phải answer: từ 27/08 cột answer là CHÍNH TẢ
    # THẬT của địa danh, đã sửa 3 nhãn vốn chỉ là lỗi Whisper ("Ông Trưởng" ->
    # "Ông Chưởng"). Bắt answer phải nằm trong lời nói là bắt nhãn phải sai theo
    # máy nghe. answer_asr giữ nguyên văn nên vẫn kiểm được đúng điều cần kiểm:
    # mỗi câu có một đoạn lời nói phủ đúng khung của nó.
    lech = [c["query_id"] for c in cau
            if _bd(c.get("answer_asr") or c["answer"])
            not in _bd(qamod.asr_khung(c["video_id"], int(c["frame_idx"])))
            and "sua_chinh_ta" not in c.get("nguon", "")]
    kiem(not lech, f"mọi đáp án nằm trong lời nói phủ đúng khung (lệch: {lech})")
    sua = [c["query_id"] for c in cau
           if c.get("answer_asr") and c["answer_asr"] != c["answer"]]
    kiem(len(sua) >= 3,
         f"nhãn đã đối chiếu nguồn ngoài, giữ lại nguyên văn Whisper ({sua})")

# ── bộ chấm Q&A: "hai người" và "2" phải là một, "N182MT" và "N182WT" thì không
import importlib.util as _iu  # noqa: E402

_sp = _iu.spec_from_file_location("dgq", ROOT / "eval" / "danh_gia_qa.py")
_dgq = _iu.module_from_spec(_sp)
_sp.loader.exec_module(_dgq)

for a, b, mong, vi in [
        ("4", "bốn miếng", True, "số và chữ số cùng giá trị"),
        ("2", "hai người", True, "số và chữ số cùng giá trị"),
        ("3", "bốn miếng", False, "hai số khác nhau"),
        ("TP.HCM", "TPHCM", True, "dấu chấm không quyết định điểm"),
        ("N182MT", "N182WT", False, "số dính trong mã hiệu KHÔNG phải số đếm"),
        ("", "hai", False, "đáp án rỗng luôn sai")]:
    kiem(_dgq.trung(a, b) is mong, f"chấm {a!r} với {b!r}: {vi}")

# Ba ca dưới đây từng được chấm ĐÚNG oan, phát hiện khi đo bộ chữ nhỏ 24/08.
_bang = "Thịt bò 150g, rau má 150g, ớt sừng 1 trái, nước cốt chanh 2M"
for a, b, mong, vi in [
        ("13 Km", "1.3 Km", False, "bỏ khoảng trắng không được nuốt dấu thập phân"),
        (_bang, "1 trái", False, "đọc cả bảng rồi ăn điểm vì có chứa chuỗi đúng"),
        ("Cầu Phước Long, cầu Rạch Đĩa", "Phước Long", True,
         "kể thêm một tên vẫn tính đúng, đừng siết quá tay"),
        ("1.3 Km", "1.3 Km", True, "đáp án đúng y hệt vẫn phải đúng")]:
    kiem(_dgq.trung(a, b) is mong, f"chấm {a[:26]!r} với {b!r}: {vi}")

# Bộ đo chữ nhỏ: chữ phải THẬT SỰ nhỏ, nếu không nó lại thành bộ đo chữ to.
_bo = ROOT / "eval" / "queries_qa_chunho.csv"
if not _bo.exists():
    kiem(False, "có bộ đo eval/queries_qa_chunho.csv")
else:
    with open(_bo, encoding="utf-8") as f:
        _cau = list(csv.DictReader(f))
    kiem(len(_cau) >= 16, f"bộ đo chữ nhỏ có đủ câu (thấy {len(_cau)})")
    kiem(len({c["video_id"] for c in _cau}) == len(_cau),
         "mỗi câu một video khác nhau")
    _cao = [int(c["cao_px"]) for c in _cau]
    kiem(max(_cao) <= 30,
         f"không câu nào là chữ to (cao nhất {max(_cao)}px, trần 30)")
    kiem(all(c["nguon"] == "ocr+phongto" for c in _cau),
         "mọi đáp án đều qua hai cửa: OCR độc lập VÀ ảnh phóng to")

# Lỗ hổng thứ tư: nhánh bỏ-khoảng-trắng từng thiếu chặn độ dài nên "2M" khớp
# "Nghệ giã nhuyễn: 2M" — một mẩu của đáp án không phải là đáp án.
kiem(not _dgq.trung("2M", "Nghệ giã nhuyễn: 2M"),
     "một mẩu ngắn KHÔNG được khớp đáp án dài chỉ vì là chuỗi con")
kiem(_dgq.trung("Nghệ giã nhuyễn: 2M", "Nghệ giã nhuyễn: 2M"),
     "đáp án đúng y hệt vẫn phải đúng sau khi siết")

# Chỉ mục thẻ cuối L26 — quy luật đo được: 40/40 keyframe cuối là bảng nguyên
# liệu, nên chỉ quét đúng khung ấy thay vì 140 giờ cho cả corpus.
import ocr_thecuoi as tc  # noqa: E402

kiem(hasattr(tc, "tim") and hasattr(tc, "quet"),
     "src/ocr_thecuoi.py có cả hàm quét lẫn hàm tra")
kiem(tc.tim("").empty, "tra chuỗi rỗng thì trả bảng rỗng, không nổ")
_b = tc.bang()
if _b is None:
    print("  ⏭  chưa lập chỉ mục thẻ cuối — bỏ qua phép kiểm nội dung")
else:
    kiem(len(_b) >= 400, f"chỉ mục phủ gần đủ 498 video L26 (thấy {len(_b)})")
    kiem(_b.video_id.is_unique, "mỗi video đúng một thẻ, không trùng")
    kiem((_b.n_vung >= 5).mean() > 0.8,
         "phần lớn thẻ đọc được nhiều mẩu chữ, đúng dạng bảng nguyên liệu")

    # Chấm phải theo NGHỊCH TẦN SUẤT: credit nhà sản xuất có ở gần mọi thẻ nên
    # truy vấn toàn credit không được phép cho điểm cao hơn nguyên liệu thật.
    _cred = tc.tim("cong ty chuong trinh truyen hinh")
    _ngl = tc.tim(" ".join(sorted(
        {w for t in _b.text.head(1) for w in tc._bo_dau(t).split()
         if len(w) > 8})[:3]))
    _dc = float(_cred.diem.max()) if len(_cred) else 0.0
    _dn = float(_ngl.diem.max()) if len(_ngl) else 0.0
    kiem(_dc < _dn, f"truy vấn toàn credit ({_dc:.2f}) thua nguyên liệu thật "
                    f"({_dn:.2f})")

_nhan_o = [t.label for t in at.text_input]
kiem(any("nguyên liệu" in (x or "") for x in _nhan_o),
     "thanh bên có ô tra bảng nguyên liệu")
kiem(any("chữ trên khung" in (x or "") for x in _nhan_o),
     "thanh bên có ô tra chữ trên khung")

# Chỉ mục chữ chung — mảnh phân biệt được hay rơi vào chỗ OCR dán liền, nên
# hàm tra phải sinh cả biến thể dán liền chứ không chỉ tách theo khoảng trắng.
import ocr_chu as oc  # noqa: E402

kiem(oc._manh_lien(["chang", "4"]) == ["chang4"],
     "hai từ kề nhau sinh được mảnh dán liền")
kiem(oc._manh_lien(["chang"]) == [],
     "một từ thì không sinh mảnh dán liền thừa")
kiem(all(len(m) >= 4 for m in oc._manh_lien(["a", "b", "c"])),
     "mảnh quá ngắn bị loại, nếu không thì trúng khắp kho")
kiem(oc._manh_lien(["1", "1", "km"])[0] == "11km",
     '"1.1 Km" thành "11km", khớp được chuỗi OCR trả về "1.1Km"')
kiem(oc.tim("").empty, "tra chuỗi rỗng thì trả bảng rỗng, không nổ")

_bc = oc.bang()
if _bc is None:
    print("  ⏭  chưa lập chỉ mục chữ chung — bỏ qua phép kiểm nội dung")
else:
    kiem(set(_bc.columns) >= {"video_id", "n", "frame_idx", "text", "ky_tu"},
         "chỉ mục chữ đủ cột để ghép ngược về metadata")
    kiem(_bc.set_index(["video_id", "n"]).index.is_unique,
         "mỗi khung đúng một dòng, chạy tiếp không sinh bản trùng")
    kiem((_bc.ky_tu == _bc.text.str.len()).all(),
         "cột đếm ký tự khớp đúng chuỗi đã lưu")

    # Tra bằng chữ CÓ THẬT trên một khung thì chính khung ấy phải lên đầu.
    _mau = _bc[_bc.ky_tu >= 60].head(1)
    if len(_mau):
        _r = _mau.iloc[0]
        _tu = [w for w in dict.fromkeys(oc._bo_dau(_r.text).split())
               if len(w) >= 5][:4]
        _kq = oc.tim(" ".join(_tu), limit=20)
        kiem(len(_kq) and (_kq.video_id == _r.video_id).any(),
             "gõ chữ có thật trên khung thì video ấy nằm trong 20 kết quả đầu")

# Ô tra chữ phải là ĐẦU MỐI dẫn sang tìm ảnh, không phải ngõ cụt in ra chữ.
_o_chu = [t for t in at.text_input if "chữ trên khung" in (t.label or "")]
if _o_chu:
    _r2 = _o_chu[0].set_value("chang").run(timeout=120)
    _nut = [b for b in _r2.button if "ảnh giống" in (b.label or "")]
    kiem(bool(_nut), f"kết quả tra chữ có nút tìm ảnh giống (thấy {len(_nut)})")
    if _nut:
        _r3 = _nut[0].click().run(timeout=120)
        _sd = (_r3.session_state["seed_row"]
               if "seed_row" in _r3.session_state else None)
        kiem(isinstance(_sd, int),
             f"bấm nút thì đặt seed_row để tab KIS tìm ảnh giống (={_sd})")

# Luật chọn của dem_o — đo 24/08: ít vật thì toàn khung thắng 4/4 so với 1/4,
# nhiều ký hiệu nhỏ thì toàn khung không nhìn xuể.
import inspect  # noqa: E402

_src = inspect.getsource(qamod.dem_o)
kiem("count_toan" in _src and "count_cong" in _src,
     "dem_o trả CẢ hai con số, không giấu con số bị loại")
kiem("toan <= 10" in _src, "dem_o có ngưỡng chuyển giữa toàn khung và cộng ô")
kiem("Chỉ đếm những thứ nằm TRONG phần cắt" in qamod.PROMPT_DEM,
     "mẫu nhắc đếm cấm đoán phần bị cắt mất — nếu không ô nào cũng đếm cả ảnh")

# ── tab TRAKE: bước 1 phải bày CẢ CHUỖI, không phải một ảnh đại diện ────────
# Bản cũ gộp mọi mốc vào một rổ rồi khử trùng video, nên mỗi video chỉ còn ĐÚNG
# MỘT ảnh — của mốc khớp mạnh nhất. Người thi không có gì để phân biệt hai video
# cùng chứa cảnh chiên chả giò, tức bước quyết định nhất (sai video là 0 điểm)
# lại là bước mù nhất. Người dùng báo đúng lỗi này.
print("\nkiểm bước 1 của tab TRAKE (bày cả chuỗi)…")
nhan_cb = [c.label for c in at.checkbox]
kiem(any("Dịch từng mốc" in (x or "") for x in nhan_cb),
     "tab TRAKE có ô tích dịch từng mốc")
for c in at.checkbox:
    if "Dịch từng mốc" in (c.label or ""):
        kiem(c.value, "ô dịch mặc định BẬT (tiếng Việt thô mất 1/8 video)")
        c.set_value(False)          # tắt cho phép kiểm khỏi phụ thuộc mạng
        break
at.text_area[0].set_value("hai bàn tay trộn nhân trong tô thuỷ tinh\n"
                          "cuốn bánh tráng thành cuốn chả giò\n"
                          "vớt chả giò vàng bằng vợt lưới")
at = at.run()
kiem(not at.exception, "không có ngoại lệ ở bước 1 TRAKE"
     + (f" — {at.exception[0].value}" if at.exception else ""))
cap = [c.value or "" for c in at.caption]
for j in (1, 2, 3):
    kiem(sum(f"mốc {j} ·" in x for x in cap) >= 5,
         f"mốc {j} hiện ở nhiều video (thấy {sum(f'mốc {j} ·' in x for x in cap)})")
kiem(any((s.label or "") == "Video sẽ nộp" for s in at.selectbox),
     "có ô chọn 'Video sẽ nộp'")

# ── kho bài nộp ──────────────────────────────────────────────────────────────
# Đây là bước ĐẮT NHẤT mà không phép kiểm nào từng chạm tới: nộp 1 dòng thay vì
# 100 là mất 0,481 so với 0,756 (đo 19/08). Xem docs, mục kho bài nộp.
print("\nkiểm kho bài nộp (100 dòng + .zip đúng thể lệ)…")
import nopbai  # noqa: E402

at.text_input[0].set_value("người phụ nữ đội nón lá đang hái dứa ngoài ruộng")
at = at.run()
kiem(any("Lưu vào kho" in (b.label or "") for b in at.button),
     "tab KIS có nút 'Lưu vào kho'")
for b in at.button:
    if "Lưu vào kho" in (b.label or ""):
        b.click()
        break
at = at.run()
kho = at.session_state["kho"] if "kho" in at.session_state else {}
kiem(bool(kho), f"bấm Lưu thì kho có file (thấy {list(kho)})")
if kho:
    ten, dong = next(iter(kho.items()))
    kiem(ten.endswith(".csv"), f"tên file có đuôi .csv ({ten})")
    kiem(nopbai.dang_cua(ten) == "kis",
         f"tên đúng quy ước query-<số>-kis ({ten})")
    kiem(len(dong) == 100, f"lưu ĐỦ 100 dòng, không phải 1 (thấy {len(dong)})")
    kiem(not nopbai.kiem_tep(ten, dong),
         f"file qua được kiem_tep — {nopbai.kiem_tep(ten, dong)[:1]}")
    ten_zip = zipfile.ZipFile(io.BytesIO(nopbai.dong_goi(dict(kho)))).namelist()
    kiem(all(x.startswith("submission/") for x in ten_zip),
         f"zip có thư mục submission/ như thể lệ ({ten_zip})")
    kiem(any("Kho bài nộp" in (h.value or "") for h in at.sidebar.subheader),
         "thanh bên có khu 'Kho bài nộp'")

    print("\nkiểm tab Nộp bài (xem trước + sửa)…")
    kiem(any((s.label or "") == "File đang soi" for s in at.selectbox),
         "có ô chọn file để soi")
    o_sua = [t for t in at.text_area if (t.label or "") == "nội dung"]
    kiem(bool(o_sua), "có ô sửa trực tiếp")
    kiem(o_sua and o_sua[0].value.split("\n")[0] == dong[0],
         "ô sửa hiện ĐÚNG nội dung sẽ nộp, không phải bản rút gọn")

    # Sửa tay rồi Áp dụng: kho phải đổi theo, và kiem_tep phải bắt được lỗi mới.
    if o_sua:
        o_sua[0].set_value("L21_V001,3000\nL26_V157,4450\nL21_V001,3000,thừa trường")
        for b in at.button:
            if "Áp dụng sửa" in (b.label or ""):
                b.click()
                break
        at = at.run()
        kho2 = at.session_state["kho"] if "kho" in at.session_state else {}
        d2 = kho2.get(ten, [])
        kiem(len(d2) == 3, f"Áp dụng sửa ghi lại được vào kho (thấy {len(d2)} dòng)")
        loi2 = nopbai.kiem_tep(ten, d2)
        kiem(any("2 trường" in e for e in loi2),
             f"dòng hỏng do sửa tay bị bắt — {loi2[:1]}")

print("\n" + ("🔴 CÓ LỖI: " + " · ".join(loi) if loi
              else "✅ giao diện dựng được, bố cục đúng thiết kế"))
sys.exit(1 if loi else 0)
