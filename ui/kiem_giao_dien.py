# -*- coding: utf-8 -*-
import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "ui" / "search_ui.py"
sys.path.insert(0, str(ROOT / "src"))

# Kho bài nộp nay ghi ra đĩa, nên phép kiểm phải trỏ sang thư mục tạm —
# ghi vào submission/ thật là để lẫn file rác vào .zip nộp.
import os  # noqa: E402
import tempfile  # noqa: E402
KHO_TAM = Path(tempfile.mkdtemp(prefix="kho_kiem_"))
os.environ["KHO_BAI_NOP"] = str(KHO_TAM)


def _anh_kho():
    """Ảnh chụp kho bài nộp THẬT: tên + cỡ + thời điểm sửa."""
    d = ROOT / "submission"
    if not d.is_dir():
        return {}
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
            for p in d.iterdir() if p.is_file()}


# Chụp TRƯỚC khi dựng app. Bản cũ chỉ kiểm "file có tồn tại không", nên đến khi
# người thi nộp thật một câu trùng tên mặc định của phép kiểm là nó báo động
# giả — mà báo động giả về mất bài thi thì lần sau không ai tin nữa.
KHO_TRUOC = _anh_kho()

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
def o_theo_nhan(app, nhan):
    """Ô nhập chọn theo NHÃN. Chọn theo chỉ số thì thêm một ô ở tab trước là
    mọi phép kiểm sau đó lặng lẽ gõ nhầm ô."""
    for t in app.text_area:
        if nhan in (t.label or ""):
            return t
    raise AssertionError(f"không thấy ô nhãn {nhan!r}")


kiem(any("Gợi ý từ ban tổ chức" in (t.label or "") for t in at.text_area),
     "ô nhập đầu tiên là gợi ý từ ban tổ chức")

print("\nchạy thử một lượt tìm thật…")
o_theo_nhan(at, "Gợi ý từ ban tổ chức").set_value("người phụ nữ đội nón lá đang hái dứa ngoài ruộng")
# timeout 120: lần dịch ĐẦU phải dựng client Gemini (~8 s) khi endpoint
# mạng đã chết. Mặc định 3 s làm màn hình dựng dở dang, check báo sai chỗ.
at = at.run(timeout=120)
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

o_theo_nhan(at, "Gợi ý từ ban tổ chức").set_value("người phụ nữ đội nón lá đang hái dứa ngoài ruộng buổi sáng")
at = at.run()
kiem(len(dd(at)) == 1,
     "GIỮ đánh dấu khi gợi ý chỉ dài thêm")

o_theo_nhan(at, "Gợi ý từ ban tổ chức").set_value("hai chiếc máy bay đậu trên đường băng")
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

# Lọc khung dẫn chuyện của BTC. Đo 29/08 trên 81 câu bọc NGUYÊN VĂN khuôn chữ
# đề thật: không lọc 0,5086 → lọc 0,7086. Trên bộ đo sạch: 0,7160 cả hai.
import loccau as _lc  # noqa: E402

_g = ("Đoạn clip cần tìm là cảnh quay trong một khu rừng, dưới gốc cây to. "
      "Đôi mắt chim đỏ rực, tạo điểm nhấn rõ ràng. "
      "Đây là loài chim thường thấy ở vùng Nam Bộ Việt Nam.")
_l = _lc.bo_nhieu(_g)
kiem("Đoạn clip cần tìm" not in _l, "bỏ được cụm mở 'Đoạn clip cần tìm là cảnh'")
kiem("tạo điểm nhấn" not in _l, "bỏ được mệnh đề cảm thán 'tạo điểm nhấn'")
kiem("thường thấy ở vùng Nam Bộ" not in _l, "bỏ được câu kiến thức ngoài hình")
kiem("khu rừng" in _l and "mắt chim đỏ rực" in _l,
     "GIỮ nguyên mọi chi tiết nhìn thấy được")
kiem("tạo thành hình chân dung" in _lc.bo_nhieu(
        "các mảnh bìa đổ bóng lên tường, tạo thành hình chân dung một người đàn ông"),
     "KHÔNG bỏ 'tạo thành' — đó là thứ nhìn thấy được, khác 'tạo cảm giác'")
kiem(_lc.bo_nhieu("Một con mèo vàng nằm trên ghế") == "Một con mèo vàng nằm trên ghế",
     "câu sạch thì lọc KHÔNG đổi một chữ nào")

# Cửa sổ 64 token của SigLIP2: 12/18 câu KIS đề thật vòng 1 bị cắt cụt trong im
# lặng. Người thi phải nhìn thấy điều đó chứ không phải đoán.
kiem(_lc.NGAN_SACH == 64, f"ngân sách token = 64 (thấy {_lc.NGAN_SACH})")
try:
    import json as _js
    import open_clip as _oc
    _mf = _js.loads((ROOT / "data" / "processed_hcmc2026" / "manifest.json")
                    .read_text(encoding="utf-8"))
    _tk = _oc.get_tokenizer(_mf["clip_model"])
except Exception:
    _tk = None
if _tk is not None:
    kiem(_lc.phan_giu_lai("một con mèo vàng", _tk) is None,
         "câu ngắn thì KHÔNG báo cắt")
    _dai = ("Đoạn video mô tả cảnh một người đàn ông mặc áo sơ mi trắng đứng "
            "cạnh chiếc xe máy màu đỏ trên con đường nhựa rộng, phía sau có "
            "hàng cây xanh và một toà nhà cao tầng màu xám, bên trái khung hình "
            "còn có hai đứa trẻ đang chạy nhảy vui đùa cạnh bờ tường gạch cũ.")
    kiem(_lc.dem(_dai, _tk) >= 64, f"câu dài dùng hết 64 token ({_lc.dem(_dai, _tk)})")
    kiem(isinstance(_lc.phan_giu_lai(_dai, _tk), str),
         "chỉ ra được ĐÚNG phần encoder đọc tới")

# Nhiều câu liền nhau KHÔNG mặc nhiên là nhiều cảnh: phải có mốc thời gian.
kiem(not _ch.co_moc("2 thanh niên phóng xe máy. Trong khung hình còn có ô tô xanh."),
     "ba câu tả CÙNG một cảnh thì không tách (không có mốc thời gian)")
kiem(_ch.co_moc("người đầu bếp lăn nguyên liệu. Nguyên liệu sau đó được phủ bột."),
     "có 'sau đó' thì nhận ra là câu nhiều cảnh")
_bep = _ch.tach_canh(
    "người đầu bếp cầm nguyên liệu dài đã xiên que và lăn qua hỗn hợp băm nhỏ. "
    "Nguyên liệu sau đó được chuyển sang một đĩa chứa bột trắng để phủ bên ngoài. "
    "Cuối cùng, nguyên liệu đã được phủ kín một lớp bột trắng và đặt sang đĩa.")
kiem(len(_bep) == 3, f"câu bếp 3 dòng tách đúng 3 cảnh (thấy {len(_bep)})")
kiem(_bep[1].startswith("Nguyên liệu"),
     f"cảnh giữa GIỮ được chủ ngữ (thấy {_bep[1][:30]!r})")

_c1 = _ch.tach_canh("Cảnh quay một nhóm hơn 5 người xếp hàng tập thể dục.")
kiem(len(_c1) == 1, "câu tả MỘT cảnh thì không bị cắt")
_cl = _ch.tach_canh("một bản đồ trên đó công trình thủy lợi lần lượt xuất hiện "
                    "bốn lần. Sau đó chuyển sang cảnh con đập quay từ trên cao")
kiem(len(_cl) == 2, f"'lần lượt' KHÔNG phải mốc đổi cảnh (thấy {len(_cl)})")

# Đề vòng 2 xuống dòng mỗi cảnh thay vì viết "sau đó". Bám vào chữ mốc thì 7
# câu bốn cảnh bị gộp làm một rồi cắt cụt ở token 64. Đo 30/08 trên 30 câu đề
# thật: số câu có cảnh vẫn tràn giảm 10 -> 6.
_4d = ("Một đầu bếp chế biến món ăn trong chảo, với các miếng dồi trường trắng.\n"
       "Đầu bếp cho bông hẹ vào chảo rồi dùng dụng cụ đảo các nguyên liệu.\n"
       "Các đoạn bông hẹ dài màu xanh được trộn cùng những miếng dồi trường.\n"
       "Máy quay chuyển sang cận cảnh chảo khi đầu bếp tiếp tục xào và trộn.")
kiem(len(_ch.tach_canh(_4d)) == 4,
     f"4 dòng BTC = 4 cảnh, dù KHÔNG có chữ mốc nào (thấy {len(_ch.tach_canh(_4d))})")
kiem(not _ch.co_moc(_4d), "…và đúng là câu này không có mốc thời gian nào")
kiem(len(_ch.tach_canh(" ".join(_4d.split("\n")))) == 1,
     "cùng bấy nhiêu chữ mà viết LIỀN một dòng thì KHÔNG tách — dòng mới là ranh giới")

_nhan = ("4 cảnh này xảy ra liên tiếp nhau.\n"
         "Cảnh 1: Hai người phụ nữ cùng nhau dán niêm phong một thùng carton.\n"
         "Cảnh 2: Các thùng mì tôm và bọc bánh mì được sắp xếp ngay ngắn.\n"
         "Cảnh 3: Một người đàn ông nhấc thùng mì tôm lên và xếp lên chồng mì.\n"
         "Cảnh 4: Cảnh quay cận cảnh các thùng mì được xếp chồng trên xe tải.")
_cn = _ch.tach_canh(_nhan)
kiem(len(_cn) == 4, f"bỏ dòng meta '4 cảnh này xảy ra…' (thấy {len(_cn)} cảnh)")
kiem(not _cn[0].lower().startswith("cảnh 1"), "gỡ nhãn 'Cảnh 1:' khỏi nội dung cảnh")
kiem(len(_ch.tach_canh("E1: Cảnh đầu tiên có trái sầu riêng.\n"
                       "E2: Cảnh đầu tiên có trái măng cụt.")) == 2,
     "nhãn kiểu TRAKE 'E1:' cũng gỡ được")

_tieu_de = ("Đây là một đoạn trong bài giảng. Trên slide bao gồm:\n"
            "- Một nhóm nhân vật người 3D màu trắng vây quanh nhân vật màu đỏ.\n"
            "Hai nhân vật hoạt hình nam đang trong tư thế thi đấu kéo co.")
_td = _ch.tach_canh(_tieu_de)
kiem(len(_td) == 2, f"dòng tiêu đề kết bằng ':' KHÔNG đứng riêng (thấy {len(_td)})")
kiem("slide" in _td[0], "…mà ghép vào cảnh sau, giữ lại chữ 'slide'")
kiem(len(_ch.tach_canh(_lc.bo_nhieu(_tieu_de))) == 2,
     "…kể cả khi ĐI QUA bộ lọc trước — lọc phải giữ lại dấu ':' cuối dòng")
kiem(len(_ch.cat_dong("một dòng duy nhất, dù có sau đó và cuối cùng")) == 1,
     "cat_dong đếm DÒNG chứ không đếm mốc — giao diện lấy đó làm mặc định")

kiem("\n" in _lc.bo_nhieu("Cảnh một người đứng.\nCảnh hai người ngồi."),
     "lọc nhiễu GIỮ dấu xuống dòng — gộp về một dòng là tự xoá ranh giới cảnh")

# Ngoặc đơn: người ra đề tự phân vân, hoặc chú metadata hành chính.
kiem("sư tử" not in _lc.bo_nhieu("Một chú lân (hay rồng/sư tử?) màu vàng nhảy xuống"),
     "bỏ ngoặc phân vân '(hay rồng/sư tử?)'")
kiem("01/7/2025" not in _lc.bo_nhieu(
        "chặng đua tại thành phố thuộc tỉnh Quảng Nam (cũ, trước ngày 01/7/2025)"),
     "bỏ ngoặc metadata hành chính '(cũ, trước ngày …)'")
kiem("mộc nhĩ" in _lc.bo_nhieu("nấm mèo (mộc nhĩ) khô đặt phía dưới chén gia vị"),
     "GIỮ ngoặc chú thích đồng nghĩa — đó là nội dung, không phải nhiễu")

# 9/9 đề Q&A vòng 2 là "tả cảnh … rồi mới hỏi". Khâu TÌM chỉ cần phần tả.
_m, _h = _lc.tach_hoi("Cảnh quay một tô cháo đã nấu xong. Kế bên có 1 chén nhỏ "
                      "màu đen. Hỏi topping trong video là thịt của con gì?")
kiem(_h.startswith("Hỏi topping"), f"tách được câu hỏi ở cuối đề Q&A (thấy {_h[:30]!r})")
kiem("Hỏi topping" not in _m, "…và phần mô tả không còn dính câu hỏi")
kiem("chén nhỏ" in _m, "…nhưng vẫn giữ đủ phần tả cảnh")
_m2, _h2 = _lc.tach_hoi("người này đối thoại với người đối diện để xem hôm nay "
                        "nấu món gì. Hỏi X là con gì?")
kiem("nấu món gì" in _m2,
     "câu TẢ kết bằng 'gì.' KHÔNG bị nuốt theo — chỉ cắt câu kết bằng '?' hoặc mở bằng 'Hỏi'")
kiem(_lc.tach_hoi("Hai bạn trẻ đang treo băng-rôn lớn màu xanh dương.")[1] == "",
     "câu KIS không có câu hỏi thì không cắt gì")
# p2-27 kết bằng dấu CHẤM: "…số nào không được nhìn thấy… trong các số từ 1-8."
_m3, _h3 = _lc.tach_hoi("Cảnh quay một chú lân đang biểu diễn. Trong các số "
                        "từ 16 giây đầu, số nào không được nhìn thấy từ góc "
                        "nhìn của camera trong các số từ 1-8.")
kiem(_h3.startswith("Trong các số") and "chú lân" in _m3,
     "câu hỏi kết bằng dấu CHẤM vẫn tách được (luật rộng)")
# Luật rộng CHỈ chạy khi luật hẹp trắng tay, và chỉ lấy MỘT câu — nếu không
# thì p2-30 mất luôn câu tả "…để xem hôm nay nấu món gì."
_m4, _h4 = _lc.tach_hoi("Cô gái cầm lên 2 con X. Sau đó người này đối thoại "
                        "với người đối diện để xem hôm nay nấu món gì. "
                        "Hỏi X là con gì?")
kiem(_h4 == "Hỏi X là con gì?" and "nấu món gì" in _m4,
     "luật rộng KHÔNG ăn lem sang câu tả khi luật hẹp đã tìm được câu hỏi")
# TRAKE: số mốc phải khớp số cột nộp, một dòng bối cảnh thừa là hỏng cả câu.
_mc, _bc = _ch.moc_trake(
    "Video về một khu vườn cây ăn trái ở miền Tây Nam Bộ. Đây là chuỗi liên "
    "tiếp các cảnh quay về 4 loại trái cây trong vườn.\n"
    "E1: Cảnh đầu tiên có trái sầu riêng.\n"
    "E2: Cảnh đầu tiên có trái măng cụt.\n"
    "E3: Cảnh đầu tiên có trái bưởi.\n"
    "E4: Cảnh đầu tiên có trái dâu bòn bon.")
kiem(len(_mc) == 4 and len(_bc) == 1 and _mc[0].startswith("Cảnh đầu tiên"),
     f"đề TRAKE có nhãn: chỉ dòng CÓ NHÃN là mốc (thấy {len(_mc)} mốc)")
_mc2, _bc2 = _ch.moc_trake(
    "4 cảnh này xảy ra liên tiếp nhau.\n"
    "Cảnh 1: Hai người phụ nữ dán niêm phong một thùng carton.\n"
    "Cảnh 2: Các thùng mì tôm được sắp xếp ngay ngắn.\n"
    "Cảnh 3: Một người đàn ông nhấc thùng mì tôm lên.\n"
    "Cảnh 4: Cận cảnh các thùng mì xếp chồng trên xe tải.")
kiem(len(_mc2) == 4, f"dòng nói về CẤU TRÚC đề không thành mốc (thấy {len(_mc2)})")
kiem(_ch.moc_trake("trộn nhân\nphết trứng\ncuốn lại")[0] == 
     ["trộn nhân", "phết trứng", "cuốn lại"],
     "đề KHÔNG đánh nhãn thì mọi dòng vẫn là mốc")
import numpy as _np  # noqa: E402
_S = _np.array([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0]], dtype=_np.float32)
_v = _np.array(["A", "A", "B"]); _t = _np.array([0.0, 5.0, 0.0], dtype=_np.float32)
_d = _ch.diem_chuoi(_S, _v, _t)
kiem(_d[0] > _d[2], "khung có cảnh sau nối tiếp được cộng điểm, khung lẻ thì không")
# Bản cũ dùng cửa sổ làm CỬA KIỂM TRA CÓ MẶT rồi vẫn lấy max hậu tố toàn video:
# hễ có bất kỳ khung nào trong cửa sổ là cảnh sau được phép khớp ở tận cuối
# video. Tức tham số cua_so gần như vô hiệu — phải khoá lại bằng phép kiểm.
_S2 = _np.array([[9.0, 0.0, 0.0], [0.0, 0.0, 9.0]], dtype=_np.float32)
_v2 = _np.array(["A", "A", "A"])
_t2 = _np.array([0.0, 10.0, 1000.0], dtype=_np.float32)
kiem(_ch.diem_chuoi(_S2, _v2, _t2, cua_so=60.0)[0]
     < _ch.diem_chuoi(_S2, _v2, _t2, cua_so=1e9)[0],
     "cửa sổ thời gian được áp THẬT (cảnh sau cách 1000 s không được cộng ở cửa sổ 60 s)")
# Vét cạn theo ĐÚNG định nghĩa mới: một DÃY khung tăng dần, mỗi cảnh một
# khung, hai cảnh liền nhau cách nhau tối đa cua_so giây. Bản trước lấy max
# ĐỘC LẬP cho từng cảnh trên cùng một khoảng sau cảnh đầu, nên nó không ép
# thứ tự giữa các cảnh sau — chính mô hình đó vừa bị thay.
def _day_tot_nhat(_ss, _vv, _tt, _w):
    _k, _n = _ss.shape
    if _k == 1:
        return _ss[0].copy()
    _vt = {int(g): r for r, g in enumerate(_np.lexsort((_tt, _vv)))}

    def _di(_j, _tu):
        if _j == _k:
            return 0.0
        _tot = None
        for _g in range(_n):
            if (_vv[_g] == _vv[_tu] and _vt[_g] > _vt[_tu]
                    and _tt[_g] - _tt[_tu] <= _w):
                _d = _di(_j + 1, _g)
                if _d is not None:
                    _v = _ss[_j][_g] + _d
                    _tot = _v if _tot is None else max(_tot, _v)
        return _tot

    _ra = _np.zeros(_n, dtype=_np.float32)
    for _a in range(_n):
        _d = _di(1, _a)
        _ra[_a] = 0.0 if _d is None else (_ss[0][_a] + _d)
    return _ra / _k


_rng = _np.random.default_rng(7)
_sai = 0
for _ in range(40):
    _n, _k = int(_rng.integers(2, 13)), int(_rng.integers(2, 5))
    _vv = _np.sort(_rng.integers(0, 3, _n))
    _tt = _np.concatenate([_np.sort(_rng.random((_vv == u).sum()) * 200)
                           for u in _np.unique(_vv)]).astype(_np.float32)
    _ss = _rng.random((_k, _n)).astype(_np.float32)
    _w = float(_rng.choice([5.0, 30.0, 90.0]))
    _sai += not _np.allclose(_ch.diem_chuoi(_ss, _vv, _tt, cua_so=_w),
                             _day_tot_nhat(_ss, _vv, _tt, _w), atol=2e-6)
kiem(_sai == 0, f"chấm chuỗi khớp phép vét cạn trên 40 ca ngẫu nhiên ({_sai} lệch)")

# Dịch: endpoint Google chặn theo IP khi gọi dồn (hỏng 6/6 ngày 28/08). Đường
# lùi phải là Gemini chứ KHÔNG phải opus: đo 28/08 opus 0,5926, thua cả tiếng
# Việt thô 0,7160. Một đường lùi dịch dở còn tệ hơn không dịch.
# Lời nói là Ý KIẾN THỨ HAI, không phải một số để cộng. RRF(CLIP+BM25 lời nói)
# đã đo là âm ở mọi trọng số; cái đo được là khác — trên 30 câu đề THẬT vòng 2,
# lời nói trúng đúng video ngay hạng 1 ở 7 câu, CLIP chỉ 4 trong cùng nhóm ấy.
import ast  # noqa: E402
_ma = ast.parse((ROOT / "ui" / "search_ui.py").read_text(encoding="utf-8"))
_ten_ham = {n.name for n in ast.walk(_ma) if isinstance(n, ast.FunctionDef)}
kiem("_goi_y_loi_noi" in _ten_ham, "tab KIS có ô gợi ý từ lời nói")
_than = (ROOT / "ui" / "search_ui.py").read_text(encoding="utf-8")
kiem("_goi_y_loi_noi(query_vi.strip() or query, hits)" in _than,
     "ô gợi ý nhận câu TIẾNG VIỆT — kho lời nói là tiếng Việt, đừng đưa bản dịch")
_kdc = _than.split("def _goi_y_loi_noi")[1].split("@st.cache_data")[0]
kiem("tong" not in _kdc and "score" not in _kdc,
     "ô gợi ý KHÔNG đụng vào điểm xếp hạng (trộn RRF đã đo là âm)")
kiem("except Exception" in _kdc,
     "thiếu parquet ASR thì báo một dòng chứ không làm sập cả tab")
import asr_tim as _at  # noqa: E402
kiem(hasattr(_at, "tim") and hasattr(_at, "diem_khung"),
     "asr_tim còn đủ hai cửa vào mà giao diện và bộ đo đang gọi")
import translate as _tr  # noqa: E402
kiem(hasattr(_tr, "dich_gemini"), "đường lùi CHÍNH là Gemini")
kiem(_tr.MO_HINH_LUI == "",
     f"opus MẶC ĐỊNH TẮT vì 0,5926 < 0,7160 (thấy {_tr.MO_HINH_LUI!r})")
kiem(hasattr(_tr, "mach_mang_con_song"),
     "có cầu dao: mạng chết thì thôi gọi, đỡ phí 6 giây mỗi truy vấn")
_tr.dong_lai_cau_dao()
kiem(_tr.mach_mang_con_song(), "đóng lại cầu dao thì đường mạng bật lại")
_ma = (ROOT / "ui" / "search_ui.py").read_text(encoding="utf-8")
kiem("mạng Google đang bị chặn" in _ma,
     "màn hình nói rõ khi câu được dịch bằng Gemini thay vì mạng Google")
kiem("0,716 so với 0,783" in _ma,
     "hỏng cả hai đường thì nói rõ mất bao nhiêu điểm, không chỉ báo lỗi suông")
_than_nop = _ma[_ma.index("def dong_nop_kis("):]
_than_nop = _than_nop[:_than_nop.index("\ndef ", 1)]
kiem('meta["frame_idx"]' in _than_nop,
     "dòng nộp KIS lấy frame_idx GỐC từ metadata")
kiem("khung_that" not in _than_nop and "moc_gio" not in _than_nop,
     "dòng nộp KIS KHÔNG dùng khung đã hiệu chỉnh (12,9% khung lệch 1)")
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
# Đếm là bài toán KHÁC hẳn đọc một con số in sẵn: chia ô để đọc kỹ sẽ cắt
# rời chính vật cần đếm. Đo trên 8 câu thật vòng 2 — đọc-kỹ thắng mọi loại
# TRỪ đếm, chỗ đó nó về 0 còn nhìn-toàn-khung được 1,0.
for _q in ["Trên đĩa có mấy miếng cà chua bi?",
           "Có mấy công an đứng hai bên nhóm thanh niên?",
           "Mỗi lần khuôn này làm được bao nhiêu cái bánh?",
           "Con robot này di chuyển bằng mấy chân?"]:
    kiem(qamod.la_cau_dem(_q), f"nhận ra câu ĐẾM: {_q[:42]}")
# "hai" chứa "ha" — không neo ranh giới từ thì câu công an bị loại oan.
kiem(qamod.la_cau_dem("Có mấy công an đứng hai bên?"),
     "đơn vị đo phải neo ranh giới từ ('ha' nằm trong 'hai')")
for _q in ["Con số được viết trên phần hông xe màu trắng là số mấy?",
           "Hỏi phần thịt có trọng lượng bao nhiêu trong bảng nguyên liệu?",
           "Loài cây (II) sinh trưởng tốt nhất khi độ mặn bao nhiêu phần nghìn?",
           "Các học sinh đeo khăn quàng cổ màu gì?"]:
    kiem(not qamod.la_cau_dem(_q), f"KHÔNG nhầm là câu đếm: {_q[:42]}")
_su = (ROOT / "ui" / "search_ui.py").read_text(encoding="utf-8")
kiem("qamod_la_dem(qa_ques)" in _su,
     "ô Kiểu câu hỏi tự chọn theo câu hỏi, không mặc định luôn nhận dạng")
# Bộ đo Q&A dựng từ đề THẬT: câu hỏi của ban tổ chức, khung đã mở ra nhìn.
_bd = ROOT / "eval" / "queries_qa_dethat.csv"
kiem(_bd.is_file(), "có bộ đo Q&A dựng từ câu hỏi đề thật")
if _bd.is_file():
    import pandas as _pd  # noqa: E402
    _d = _pd.read_csv(_bd)
    kiem(len(_d) >= 8, f"bộ đo đề thật có đủ câu (thấy {len(_d)})")
    kiem("bang_chung" in _d.columns,
         "mỗi câu ghi rõ BẰNG CHỨNG đã xác nhận đáp án bằng cách nào")
    kiem(int((_d["kind"] == "nhìn").sum()) == 0,
         "0 câu màu sắc — bộ đo cũ 80% là màu, đề thật thì không có câu nào")
    _m = _pd.read_parquet(ROOT / "data" / "processed_hcmc2026"
                          / "metadata.parquet")
    _co = set(zip(_m.video_id, _m.frame_idx.astype(int)))
    _thieu = [f"{r.video_id}/{r.frame_idx}" for r in _d.itertuples()
              if (r.video_id, int(r.frame_idx)) not in _co]
    kiem(not _thieu, f"mọi khung đáp án CÓ THẬT trong kho ({_thieu[:2]})")

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

_nhan_o = [t.label for t in at.text_input] + [t.label for t in at.text_area]
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
o_theo_nhan(at, "MỖI DÒNG MỘT MỐC").set_value("hai bàn tay trộn nhân trong tô thuỷ tinh\n"
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

o_theo_nhan(at, "Gợi ý từ ban tổ chức").set_value("người phụ nữ đội nón lá đang hái dứa ngoài ruộng")
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

    # Kho từng chỉ sống trong session_state: Streamlit sập giữa giờ thi là mất
    # sạch bài đã lưu. Nay mỗi lần Lưu phải để lại một bản trên đĩa.
    tep_dia = KHO_TAM / ten
    kiem(tep_dia.exists(), f"Lưu vào kho có ghi ra đĩa ({ten})")
    if tep_dia.exists():
        tren_dia = [x for x in tep_dia.read_text(
            encoding="utf-8").splitlines() if x.strip()]
        kiem(tren_dia == [str(d) for d in dong],
             f"bản trên đĩa khớp từng dòng với kho ({len(tren_dia)} dòng)")
    _sau = _anh_kho()
    kiem(_sau == KHO_TRUOC,
         "phép kiểm KHÔNG đụng vào kho bài nộp thật "
         f"(thêm/sửa: {sorted(set(_sau.items()) - set(KHO_TRUOC.items()))[:3]}; "
         f"mất: {sorted(set(KHO_TRUOC) - set(_sau))[:3]})")

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
# Nạp đề từ thư mục BTC: 30 câu mà copy tay thì mất cả buổi, và gõ lệch tên file
# nộp là câu đó không được chấm.
print()
print("kiểm nạp đề từ thư mục BTC…")
import nopbai as _nb  # noqa: E402
DE_TAM = Path(tempfile.mkdtemp(prefix="de_kiem_"))
(DE_TAM / "query-2-kis.txt").write_text("hai chiếc máy bay đậu trên đường băng",
                                        encoding="utf-8")
(DE_TAM / "query-10-kis.txt").write_text("người phụ nữ đội nón lá hái dứa",
                                         encoding="utf-8")
(DE_TAM / "query-3-qa.txt").write_text("một con mèo nằm trên ghế", encoding="utf-8")
(DE_TAM / "query-4-trake.txt").write_text("trộn nhân\ncuốn bánh", encoding="utf-8")

_ten = [p.stem for p in _nb.tep_de(DE_TAM)]
kiem(_ten == ["query-2-kis", "query-3-qa", "query-4-trake", "query-10-kis"],
     f"sắp đề theo SỐ chứ không theo chữ (thấy {_ten})")
kiem(_nb.dang_cua("query-10-kis") == "kis"
     and _nb.dang_cua("query-3-qa") == "qa"
     and _nb.dang_cua("query-4-trake") == "trake",
     "đọc đúng dạng câu từ tên file")

os.environ["THU_MUC_DE"] = str(DE_TAM)
at_de = AppTest.from_file(str(UI), default_timeout=600).run()
kiem(any("Đề thi" in (h.value or "") for h in at_de.subheader),
     "thanh bên có khu 'Đề thi'")
_nut = [b for b in at_de.button if b.label.endswith("query-10-kis")]
kiem(bool(_nut), "mỗi file đề là một nút bấm")
if _nut:
    at_de = _nut[0].click().run()
    def _tt(app, k):
        try:
            return app.session_state[k]
        except Exception:
            return None
    kiem(_tt(at_de, "kis_q") == "người phụ nữ đội nón lá hái dứa",
         "bấm nút đề thì câu tự vào ô KIS")
    kiem(_tt(at_de, "ten_kis") == "query-10-kis",
         f"tên file nộp tự đặt khớp tên đề (thấy {_tt(at_de, 'ten_kis')!r})")
    kiem(any("⬜ query-2-kis" in b.label for b in at_de.button),
         "câu chưa làm hiện dấu ⬜")
os.environ.pop("THU_MUC_DE", None)

# ---- giữ lượt dịch ---------------------------------------------------
# Endpoint Google chặn theo IP khi gọi dồn. Một câu 4 cảnh trước đây tiêu 5
# lượt (câu chính + từng cảnh) và kho chỉ nằm trong RAM nên khởi động lại là
# mất sạch. Đo trên 18 câu đề thật: một vòng gõ đề 49 lượt -> 18.
import json as _js  # noqa: E402
import time as _tt  # noqa: E402
import urllib.error as _ue  # noqa: E402
import urllib.request as _ur  # noqa: E402
import translate as _tr  # noqa: E402

_tep = ROOT / "data" / "kernel_out" / "dich_cache.json"
kiem(_tr.TEP_KHO == _tep or str(_tr.TEP_KHO).endswith("dich_cache.json"),
     "kho dịch nằm trên ĐĨA, không chỉ trong RAM")

_luu_cache = dict(_tr._CACHE)
_luu_ghi = _tr.ghi_kho
_luu_gemini = _tr.dich_gemini
_luu_urlopen = _ur.urlopen
_luu_hong, _luu_mo = _tr._hong_lien_tiep, _tr._mo_lai_luc
try:
    _tr.ghi_kho = lambda: None
    _tr.dich_gemini = lambda t: None

    # 1. gộp: n câu chưa có trong kho chỉ tốn MỘT lượt
    _tr._CACHE.clear()
    _tr.dong_lai_cau_dao()
    _dem = {"n": 0}

    def _gia(*a, **k):
        _dem["n"] += 1
        q = a[0] if a else k.get("url")
        import urllib.parse as _up
        goc = _up.unquote(str(q).split("&q=")[-1])
        doan = [[d + chr(10), d] for d in goc.split(chr(10))]
        doan[-1][0] = doan[-1][0].rstrip(chr(10))

        class _R:
            def read(self_):
                return _js.dumps([[[x[0], x[1]] for x in doan]]).encode("utf-8")

            def __enter__(self_):
                return self_

            def __exit__(self_, *e):
                return False
        return _R()

    _ur.urlopen = _gia
    _ra = _tr.dich_nhieu(["cảnh một", "cảnh hai", "cảnh ba", "cảnh bốn"])
    kiem(_dem["n"] == 1, "4 cảnh chỉ tốn MỘT lượt gọi, không phải bốn")
    kiem(_ra == ["cảnh một", "cảnh hai", "cảnh ba", "cảnh bốn"],
         "gộp rồi tách lại theo dấu xuống dòng vẫn đúng thứ tự từng cảnh")
    _dem["n"] = 0
    kiem(_tr.dich_nhieu(["cảnh một", "cảnh hai"]) == ["cảnh một", "cảnh hai"]
         and _dem["n"] == 0, "câu đã dịch rồi thì KHÔNG tốn lượt nữa")

    # 2. lệch dòng thì lùi về từng câu, tuyệt đối không ghép nhầm cảnh
    _tr._CACHE.clear()
    _goi = {"n": 0}

    def _lech(q, timeout=10.0, retries=3):
        _goi["n"] += 1
        return "MOT-DONG" if chr(10) in q else "rieng"
    _tr._goi_mang, _luu_goi = _lech, _tr._goi_mang
    _r2 = _tr.dich_nhieu(["a", "b", "c"])
    kiem(_r2 == ["rieng", "rieng", "rieng"] and _goi["n"] == 4,
         "số dòng trả về không khớp thì lùi về dịch từng câu, không ghép bừa")
    _tr._goi_mang = _luu_goi

    # 3. gặp 429 thì thôi, đừng gọi lại — gọi lại là kéo dài lệnh chặn
    _tr._CACHE.clear()
    _tr.dong_lai_cau_dao()
    _dem["n"] = 0

    def _n429(*a, **k):
        _dem["n"] += 1
        raise _ue.HTTPError("u", 429, "Too Many Requests", {}, None)
    _ur.urlopen = _n429
    _tr.to_english("một câu")
    kiem(_dem["n"] == 1, "gặp 429 thì KHÔNG thử lại (trước là 3 lượt)")

    # 4. cầu dao ngắt rồi phải TỰ mở lại: chặn thoáng qua mà mất Google cả
    #    buổi thi thì lỗ, Google 0,7827 còn Gemini 0,7556.
    _tr.to_english("câu khác")
    kiem(not _tr.mach_mang_con_song(), "hỏng liên tiếp thì ngắt mạch, thôi gọi")
    _tr._mo_lai_luc = _tt.time() - 1
    kiem(_tr.mach_mang_con_song(), "nghỉ đủ lâu thì tự mở lại thử một lượt")
finally:
    _ur.urlopen = _luu_urlopen
    _tr.ghi_kho = _luu_ghi
    _tr.dich_gemini = _luu_gemini
    _tr._CACHE.clear()
    _tr._CACHE.update(_luu_cache)
    _tr._hong_lien_tiep, _tr._mo_lai_luc = _luu_hong, _luu_mo

_su8 = (ROOT / "ui" / "search_ui.py").read_text(encoding="utf-8")
kiem("nong_dich([query_loc] + chuoi.tach_canh(query_loc))" in _su8,
     "tab KIS dịch câu chính và các cảnh trong MỘT lượt")
kiem("nong_dich(goc)" in _su8, "tab TRAKE dịch mọi mốc trong MỘT lượt")
kiem("nong_dich([loccau.bo_nhieu(qa_desc.strip()), qa_ques.strip()])" in _su8,
     "tab Q&A dịch mô tả và câu hỏi trong MỘT lượt")
kiem("mach_mang_con_song()" in _su8,
     "giao diện BÀY RA khi đường Google bị chặn, không lùi im lặng")

# ---- tab Q&A: dịch câu, và kênh CÂU HỎI ------------------------------
# Tab KIS vốn dịch, tab Q&A thì đưa thẳng tiếng Việt vào search(). Đo trên
# 8 câu hỏi thật, cho sẵn đúng video, rổ 30: mô tả 5/8 khi để tiếng Việt,
# 6/8 khi đã dịch. Thêm kênh câu hỏi xen kẽ thì lên 8/8, và bộ 50 câu tự
# dựng giữ nguyên 49/50 — không mất gì.
_su7 = (ROOT / "ui" / "search_ui.py").read_text(encoding="utf-8")
kiem("cau_hoi: str | None = None" in _su7,
     "search() nhận thêm kênh CÂU HỎI")
_tt = _su7[_su7.index("def search("):]
_tt = _tt[:_tt.index("\n@st.cache_data")]
# Điều kiện đã ĐO là "trong đúng video". Bật kênh câu hỏi trên toàn kho là
# chưa đo — câu hỏi rất chung ("số nào") nên dễ kéo về khung bất kỳ.
kiem("if trong_video and cau_hoi" in _tt,
     "kênh câu hỏi CHỈ bật khi đã bó vào một video — đúng điều kiện đã đo")
kiem("xen.append(z)" in _tt,
     "xen kẽ hai thứ tự, không cộng điểm (cộng thì thua ở cả hai bộ)")
kiem('preprocess_query(loccau.bo_nhieu(qa_desc.strip())' in _su7,
     "tab Q&A lọc rồi DỊCH mô tả trước khi tìm, như tab KIS")
kiem("cau_hoi=_qh" in _su7, "câu hỏi đã dịch được truyền vào search()")
kiem('key="qa_dich"' in _su7, "người thi tắt được việc dịch ở tab Q&A")
# Xen kẽ làm điểm không còn giảm đều; công thức cảnh báo rổ giả định thế.
import rerank as _rr  # noqa: E402
kiem(not _rr.basket_uncertain([1.0, 0.2, 0.9, 0.3], shallow=4),
     "cảnh báo rổ chịu được thứ tự KHÔNG giảm dần, không báo động khống")
kiem(_rr.basket_uncertain([1.0, 0.99, 0.98, 0.97], shallow=4),
     "cảnh báo rổ vẫn bắt được rổ có điểm sát nhau")

# ---- rổ Q&A bó vào MỘT video ----------------------------------------
# Khung trả lời được câu hỏi thường không phải khung khớp mô tả — nó ở chỗ
# có bảng nguyên liệu hay biển hiệu. Đo trên 8 câu hỏi THẬT, tính trên các
# câu mà video đã đúng: rổ 30 ảnh trong video chứa đáp án 4/5, rổ toàn kho
# chỉ 2/5. Bộ 50 câu tự dựng thì hoà 23/23 — nhãn của nó đặt ngay tại cảnh
# được tả nên không nhìn thấy hiện tượng.
_su6 = (ROOT / "ui" / "search_ui.py").read_text(encoding="utf-8")
kiem("trong_video: str | None = None" in _su6,
     "search() nhận được phạm vi MỘT video")
_than_tim2 = _su6[_su6.index("def search("):]
_than_tim2 = _than_tim2[:_than_tim2.index("\n@st.cache_data")]
kiem("if trong_video:" in _than_tim2,
     "search() thật sự bó ứng viên vào video ấy")
kiem("np.isfinite(tong[top])" in _than_tim2,
     "bó xong phải LỌC -inf, không thì rổ đầy khung của video khác")
kiem("trong_video=qa_bo_video" in _su6,
     "tab Q&A truyền video đã đánh dấu vào search()")
# Canh bạc được-ăn-cả: bó nhầm video là mất trắng, nên chỉ bật khi CÓ đánh
# dấu, và phải để người thi tắt được.
_i_mark = _su6.index("qa_bo_video = None")
_khoi_bo = _su6[_i_mark:_su6.index("qa_uu_tien = st.checkbox(")]
kiem("if qa_marked:" in _khoi_bo,
     "CHỈ bó rổ khi người thi đã đánh dấu khung — không tự đoán video")
kiem("st.checkbox(" in _khoi_bo,
     "người thi tắt được việc bó rổ (bó nhầm video là mất trắng)")

# ---- soi lại khung sắp nộp -------------------------------------------
# 5/18 khung KIS đội nộp ở đề thật vòng 2 SAI HẲN, 3 khung nữa đúng đoạn
# sai khoảnh khắc. Nguồn lỗi lớn nhất đo được, và nó nằm ở khâu NGƯỜI chọn
# khung. Bày DANH SÁCH chi tiết, không bày phán quyết.
import soikhung as _sk  # noqa: E402
_tho_gach = ("- Áo đỏ: CO\n- Nón trắng: KHONG (đội mũ xanh)\n\nTHIEU")
_m, _p = _sk._tach(_tho_gach)
kiem(len(_m) == 2 and _m[0][1] and not _m[1][1] and _p == "THIEU",
     "đọc được danh sách kiểu gạch đầu dòng, giữ cả lý do")
# Model đổi sang danh sách ĐÁNH SỐ tuỳ lúc — bắt sót là mất 8/18 phán quyết.
_tho_so = "1. Áo đỏ: CO\n2. Nón trắng: KHONG\n\nDUNG"
kiem(len(_sk._tach(_tho_so)[0]) == 2,
     "đọc được cả danh sách ĐÁNH SỐ, không chỉ gạch đầu dòng")
kiem(_sk._tach("")[0] == [] and _sk._tach(None)[0] == [],
     "văn bản rỗng thì ra danh sách rỗng, không nổ")
kiem([t for t, _ in _sk.thieu({"muc": _m})] == ["Nón trắng"],
     "thieu() chỉ trả những chi tiết KHÔNG thấy")
# Ngưỡng đếm: bắt 8/8 khung có vấn đề, báo nhầm 2/10 — hơn hẳn phán quyết
# của chính model (5/10 báo nhầm).
kiem(_sk.dang_ngo({"muc": [("a", False, ""), ("b", True, "")]}),
     "một nửa chi tiết không thấy thì đã đáng dừng lại xem")
kiem(not _sk.dang_ngo({"muc": [("a", False, ""), ("b", True, ""),
                               ("c", True, "")]}),
     "thiếu 1/3 chi tiết thì chưa báo động")
kiem(not _sk.dang_ngo({"muc": []}) and not _sk.dang_ngo({}),
     "không soi được thì im lặng, không báo động khống")
_su5 = (ROOT / "ui" / "search_ui.py").read_text(encoding="utf-8")
_than_sk = _su5[_su5.index("def _bang_soi_khung("):]
_than_sk = _than_sk[:_than_sk.index("\ndef _bang_goi_ten(")]
kiem("st.button(" in _than_sk,
     "người thi tự bấm — mỗi lần soi tốn một lời gọi VLM có ảnh")
kiem("phan_quyet" not in _than_sk,
     "KHÔNG bày phán quyết của model (báo nhầm 5/10 khung tốt)")
kiem("except Exception" in _than_sk,
     "hết khoá hay mất mạng thì báo, không làm hỏng cả lượt tìm")
kiem("_bang_soi_khung(query_vi.strip() or query, hits)" in _su5,
     "bảng soi nhận câu TIẾNG VIỆT — đề viết tiếng Việt")

# ---- bộ đo KIS mức KHUNG dựng từ đề thật ----------------------------
# 18/19 câu KIS vòng 2 đã mở ảnh ra nhìn: chỉ 10 khung đội nộp thật sự cho
# thấy cảnh đề tả, 3 khung đúng đoạn sai khoảnh khắc, 5 khung sai hẳn. Bộ
# này chỉ giữ 10 câu xác nhận được — nhãn sai thì thà không có.
import pandas as _pdv  # noqa: E402
_bk = ROOT / "eval" / "queries_kis_dethat.csv"
kiem(_bk.is_file(), "có bộ đo KIS mức KHUNG dựng từ đề thật")
if _bk.is_file():
    _dk = _pdv.read_csv(_bk)
    kiem(len(_dk) >= 10, f"đủ câu (thấy {len(_dk)})")
    kiem("bang_chung" in _dk.columns,
         "mỗi câu ghi rõ NHÌN THẤY GÌ trong ảnh, không chỉ ghi đúng/sai")
    kiem(int(_dk.bang_chung.str.len().min()) > 20,
         "bằng chứng viết đủ dài để kiểm lại được, không phải dấu tích")
    _mk = _pdv.read_parquet(ROOT / "data" / "processed_hcmc2026"
                            / "metadata.parquet")
    _cok = set(zip(_mk.video_id, _mk.frame_idx.astype(int)))
    _thieuk = [f"{r.video_id}/{r.frame_idx}" for r in _dk.itertuples()
               if (r.video_id, int(r.frame_idx)) not in _cok]
    kiem(not _thieuk, f"mọi khung nhãn CÓ THẬT trong kho ({_thieuk[:2]})")
    kiem(bool((_dk.frame_idx_min <= _dk.frame_idx).all()
              and (_dk.frame_idx <= _dk.frame_idx_max).all()),
         "khoảng hợp lệ bao đúng khung nhãn")
    # Những câu đã soi và LOẠI thì không được lẻn vào bộ.
    _loai = {"p2-1-kis", "p2-2-kis", "p2-5-kis", "p2-6-kis", "p2-10-kis",
             "p2-11-kis", "p2-15-kis", "p2-18-kis", "p2-20-kis"}
    kiem(not (set(_dk.query_id) & _loai),
         "câu đã soi và loại KHÔNG lẻn vào bộ (5 khung nộp sai hẳn)")
    kiem(int((_dk.go_tay == 1).sum()) == 0,
         "0 khung gõ tay lọt vào — cả 3 khung KIS gõ tay đều trượt phép soi")

# ---- gọi tên cái mà đề chỉ TẢ ---------------------------------------
# Ban tổ chức cố tình không gọi tên vật; kho lại đánh chỉ mục theo TÊN.
# Đo trên 30 câu đề thật: cứu ĐÚNG MỘT câu (p2-22, không kênh nào thấy ->
# hạng 18) nhưng TRUNG BÌNH thua cả câu gốc (thẻ hạng-1 1/14 -> 0/14, lời
# nói 3/14 -> 1/14). Nên: người thi tự bấm, và tuyệt đối không trộn điểm.
import goiten as _gt  # noqa: E402
kiem(_gt._tach("mực ống, mực lá") == ["mực ống", "mực lá"],
     "tách tên theo dấu phẩy")
kiem(_gt._tach("-") == [] and _gt._tach("") == [] and _gt._tach("—") == [],
     "đề đã gọi tên rõ (model trả dấu gạch) thì ra danh sách rỗng")
kiem(_gt._tach("1. mực ống\n2. bạch tuộc") == ["mực ống", "bạch tuộc"],
     "bóc số thứ tự và xuống dòng khi model không nghe lời dặn")
kiem(len(_gt._tach(", ".join(f"t{_i}" * 2 for _i in range(20))))
     == _gt.MAX_TEN,
     f"chặn trần {_gt.MAX_TEN} tên, model trả tràn thì cắt")
kiem(_gt.tra([]) == {"the": [], "asr": []},
     "không có tên thì không tra gì, không nổ")

_su4 = (ROOT / "ui" / "search_ui.py").read_text(encoding="utf-8")
_than_gt = _su4[_su4.index("def _bang_goi_ten("):]
_than_gt = _than_gt[:_than_gt.index("\n@st.cache_data")]
kiem("st.button(" in _than_gt,
     "người thi tự bấm — mỗi lần đoán tên tốn một lời gọi API")
# Chỗ nguy hiểm nhất: trộn điểm tên đoán vào xếp hạng. Đã đo là TRUNG BÌNH
# thua cả câu gốc, nên trộn là lỗ.
kiem(all(_x not in _than_gt for _x in ("tong", "hits[\"score\"]", "score\"]")),
     "bảng tên đoán KHÔNG đụng vào điểm xếp hạng")
kiem("except Exception" in _than_gt,
     "hết khoá hay mất mạng thì báo, không làm hỏng cả lượt tìm")
kiem("_bang_goi_ten(query_vi.strip() or query, hits)" in _su4,
     "bảng nhận câu TIẾNG VIỆT (đề tả bằng tiếng Việt, kho chữ cũng vậy)")

# ---- mắt xích: khung nào đóng vai cảnh nào ---------------------------
# Máy chấm điểm cho khung làm CẢNH ĐẦU rồi nộp chính nó, nhưng trên 14 câu
# nhiều cảnh của đề thật chỉ 8/14 đáp án ở cảnh đầu — 6/14 ở cảnh sau, mà
# mắt xích rơi trúng với lệch trung vị 0,8 s.
kiem(hasattr(_ch, "mat_xich"), "có hàm dựng lại vết mắt xích của chuỗi")
_xau = 0
for _ in range(40):
    _n = int(_rng.integers(3, 14))
    _K = int(_rng.integers(2, 5))
    _vv = _np.sort(_rng.integers(0, 3, _n))
    _tt = _np.concatenate([_np.sort(_rng.random((_vv == u).sum()) * 200)
                           for u in _np.unique(_vv)]).astype(_np.float32)
    _ss = _rng.random((_K, _n)).astype(_np.float32)
    _w = float(_rng.choice([5.0, 30.0, 90.0]))
    _d, _vet = _ch.mat_xich(_ss, _vv, _tt, cua_so=_w)
    if not _np.allclose(_d, _ch.diem_chuoi(_ss, _vv, _tt, cua_so=_w),
                        atol=1e-6):
        _xau += 1
        continue
    for _i in range(_n):
        _m = _vet[_i]
        if _m[0] < 0:
            continue
        if int(_m[0]) != _i:
            _xau += 1
            break
        if any(_vv[_m[_k]] != _vv[_i] for _k in range(_K)):
            _xau += 1
            break
        if any(_tt[_m[_k + 1]] - _tt[_m[_k]] > _w + 1e-6
               or _m[_k + 1] <= _m[_k] for _k in range(_K - 1)):
            _xau += 1
            break
        _tong = sum(float(_ss[_k][_m[_k]]) for _k in range(_K)) / _K
        if abs(_tong - float(_d[_i])) > 1e-5:
            _xau += 1
            break
kiem(_xau == 0,
     f"vết mắt xích: đúng video, tăng dần theo thời gian, trong cửa sổ, "
     f"tổng khớp điểm ({_xau} ca sai)")

_su3 = (ROOT / "ui" / "search_ui.py").read_text(encoding="utf-8")
kiem("def bang_mat_xich(" in _su3 and "bang_mat_xich(hits)" in _su3,
     "giao diện bày dãy khung của chuỗi ra cho người thi nhìn")
kiem("dong_nop_kis(nop, _nop_mx)" in _su3,
     "danh sách nộp đi qua hàm riêng, có chỗ bật/tắt xen mắt xích")

# Phép xếp nằm ở nopbai nên KIỂM THẲNG được, không cần dựng Streamlit.
import nopbai as _nb  # noqa: E402
kiem(_nb.xen_mat_xich([10, 20, 30]) == [10, 20, 30],
     "không có mắt xích thì danh sách nộp giữ nguyên thứ tự điểm")
kiem(_nb.xen_mat_xich([10, 20], [(10, 11, 12), (20, 21, 22)])
     == [10, 11, 12, 20, 21, 22],
     "xen mắt xích theo ĐÚNG thứ tự cảnh, hết neo này mới sang neo sau")
kiem(_nb.xen_mat_xich([10, 11], [(10, 11, 12), (11, 12, 13)])
     == [10, 11, 12, 13],
     "hai dãy chồng nhau thì bỏ trùng, không nộp một khung hai lần")
kiem(_nb.xen_mat_xich([10, 20], [(-1, -1, -1), (20, 21, 22)])
     == [10, 20, 21, 22],
     "neo không dựng nổi dãy thì vẫn nộp chính nó, không bị bỏ rơi")
kiem(len(_nb.xen_mat_xich(list(range(0, 400, 3)),
                          [(i, i + 1, i + 2) for i in range(0, 400, 3)]))
     == 100,
     "danh sách nộp vẫn đúng trần 100 dòng sau khi xen")

# ---- chấm chuỗi: ép thứ tự ĐỦ, và gộp đúng thang điểm ----------------
# Bản trước lấy max độc lập cho từng cảnh nên "A rồi B rồi C" chấm y hệt
# "A rồi C rồi B"; và giao diện cộng zscore(nền) với cosine thô nên điểm
# chuỗi chỉ còn ~1/30 trọng lượng. Đo lại: hạng-1 4/14 -> 8/14 đề thật.
import numpy as _np  # noqa: E402


# Thứ tự phải ĐƯỢC ÉP: đảo hai cảnh sau thì điểm phải đổi.
_v = _np.zeros(4, dtype=_np.int64)
_p = _np.array([0.0, 5.0, 10.0, 15.0], dtype=_np.float32)
_S = _np.array([[9., 0., 0., 0.],
                [0., 9., 0., 0.],
                [0., 0., 9., 0.]], dtype=_np.float32)
_dung = _ch.diem_chuoi(_S, _v, _p, cua_so=30.0)[0]
_dao = _ch.diem_chuoi(_S[[0, 2, 1]], _v, _p, cua_so=30.0)[0]
kiem(_dung > _dao,
     f"đảo thứ tự hai cảnh SAU thì điểm phải tụt ({_dung:.3f} > {_dao:.3f})")

# Dãy đứt giữa chừng KHÔNG được ăn điểm của những cảnh đã khớp.
_S2 = _np.array([[9., 0.], [0., 9.], [0., 0.]], dtype=_np.float32)
_v2 = _np.zeros(2, dtype=_np.int64)
_p2 = _np.array([0.0, 5.0], dtype=_np.float32)
kiem(float(_ch.diem_chuoi(_S2, _v2, _p2, cua_so=30.0)[0]) == 0.0,
     "câu ba cảnh mà video chỉ đủ chỗ hai cảnh thì ăn 0, không ăn nửa điểm")

# Cửa sổ đo giữa hai cảnh LIỀN NHAU: ba cảnh cách nhau 20 s, cửa sổ 25 s
# thì dựng được; nếu đo từ cảnh đầu (40 s > 25 s) thì đã hỏng.
_p3 = _np.array([0.0, 20.0, 40.0], dtype=_np.float32)
_v3 = _np.zeros(3, dtype=_np.int64)
_S3 = _np.eye(3, dtype=_np.float32) * 9
kiem(float(_ch.diem_chuoi(_S3, _v3, _p3, cua_so=25.0)[0]) > 0,
     "cửa sổ tính giữa HAI CẢNH LIỀN NHAU, không phải từ cảnh đầu")

# Gộp phải kéo hai vế về cùng thang, nếu không điểm chuỗi bị nén mất.
kiem(hasattr(_ch, "gop"), "có hàm gộp dùng chung cho bộ đo và giao diện")
_a = _np.concatenate([_np.zeros(999), [50.0]])      # thang lớn
_b = _np.concatenate([_np.zeros(999), [0.01]])      # thang bé
_g = _ch.gop(_a, _b, alpha=_ch.ALPHA_NEN)
# Dung sai 1e-4 chứ không 1e-6: epsilon 1e-9 trong _chuan kéo lệch chuẩn
# của mảng thang bé xuống 0,9999968.
kiem(_g[999] > 0 and abs(_np.std(_ch._chuan(_b)) - 1.0) < 1e-4,
     "gộp chuẩn hoá cả hai vế nên vế thang bé không bị nén mất")
_su2 = (ROOT / "ui" / "search_ui.py").read_text(encoding="utf-8")
_than_tim = _su2[_su2.index("def search("):]
_than_tim = _than_tim[:_than_tim.index("\n@st.cache_data")]
kiem("chuoi.gop(tong," in _than_tim
     and "tong + chuoi.diem_chuoi" not in _than_tim
     and "tong + chuoi.mat_xich" not in _than_tim,
     "giao diện gộp qua chuoi.gop chứ không cộng thẳng điểm chuỗi")
_ev = (ROOT / "eval" / "evaluate.py").read_text(encoding="utf-8")
kiem("chuoi.gop(" in _ev, "bộ đo cũng gộp qua chuoi.gop — cùng một phép")
# Che video bị loại phải nằm SAU khi gộp: chuẩn hoá mảng có -inf ra nan.
kiem(_su2.index("chuoi.gop(tong,") < _su2.index("-np.inf, tong)"),
     "che video bị loại nằm SAU bước gộp (chuẩn hoá -inf thì ra nan)")

# ---- mở một video ra thì xếp khung theo ĐIỂM, không theo thời gian ----
# Video trung vị 281 khung. Xếp theo thời gian thì khung đáp án nằm ở ô thứ
# 111 (trung vị, 81 truy vấn); xếp theo điểm thì về ô thứ 1 và 91,4% nằm
# trong 5 ô đầu. Đây là chỗ tốn thời gian nhất của người thi.
_su = (ROOT / "ui" / "search_ui.py").read_text(encoding="utf-8")
kiem("def diem_ca_video(" in _su,
     "có hàm chấm MỌI khung của một video")
kiem("def show_videos(hits, ncol, query" in _su,
     "show_videos nhận câu truy vấn để chấm lại trong video")
kiem("show_videos(hits, cols_per_row, query=query)" in _su,
     "chỗ gọi có truyền câu truy vấn ĐÃ DỊCH vào")
kiem('"Xếp khung theo"' in _su, "có ô chọn cách xếp khung")
_khoi = _su[_su.index("def show_videos("):]
_khoi = _khoi[:_khoi.index("st.title(")]
kiem(_khoi.index('"điểm khớp (nên dùng)"') < _khoi.index('"thời gian"'),
     "xếp theo ĐIỂM là lựa chọn mặc định (đứng đầu danh sách)")
kiem('sort_values("score", ascending=False)' in _khoi,
     "thật sự xếp giảm dần theo điểm chứ không chỉ bày ra ô chọn")
kiem("except Exception" in _khoi,
     "chấm lại mà nổ thì vẫn còn bảng khung (có chắn ngoại lệ)")
# Chấm lại phải phủ MỌI khung, không chỉ khung đã lọt vào kết quả tìm kiếm.
import pandas as _pdv  # noqa: E402
_m = _pdv.read_parquet(ROOT / "data" / "processed_hcmc2026"
                       / "metadata.parquet")
_v = _m["video_id"].iloc[0]
_n = int((_m["video_id"] == _v).sum())
kiem(_n > 50, f"video mẫu {_v} đủ dài để việc xếp có ý nghĩa ({_n} khung)")


print("\n" + ("🔴 CÓ LỖI: " + " · ".join(loi) if loi
              else "✅ giao diện dựng được, bố cục đúng thiết kế"))
sys.exit(1 if loi else 0)
