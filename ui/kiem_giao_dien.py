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
