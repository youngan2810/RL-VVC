# RL-VVC

Kho mã nguồn chính thức của dự án ánh xạ phương pháp RL-RC-DoT sang H.266/VVenC.
Repository sử dụng từ ngày 16/11/2026: **https://github.com/youngan2810/RL-VVC**, nhánh **main**.

## Bắt đầu

- [Hướng dẫn training, input, resume và evaluation](README_RL_RC_DOT.md)
- [Đặc tả state, action, reward và ánh xạ phương pháp](scripts/rl_rc_dot/METHOD_MAPPING.md)
- [Mini Pilot: kiểm tra kỹ thuật trước training](notebooks/RL_RC_DoT_VVC_Mini_Pilot_Kaggle.ipynb)
- [Mini Trial: 20.480 tương tác](notebooks/RL_RC_DoT_VVC_Mini_Trial20480_Kaggle.ipynb)
- [Mini Test30: đánh giá checkpoint đã chọn](notebooks/RL_RC_DoT_VVC_Mini_Test30_Kaggle.ipynb)

Mini sử dụng Train16 / Validation4 / Test30 theo data lock trong source. Kaggle cần MOV BDD100K đã khóa, Internet và GPU T4×2 theo hướng dẫn. Metadata trong source không thay thế video.

## Phiên bản hiện tại

Đã bao gồm sửa lỗi cộng lặp delta QP ở inter QG, hợp đồng action `native-qpa-single-delta/2`. Bắt đầu training bằng `MODE='new'`; không resume checkpoint trước bản sửa. Chưa có bằng chứng chất lượng học hoặc BD-rate của bản sửa từ training thực tế.

Snapshot nhập từ source dự án cũ tại commit `af01dd3ac10a68ff8a4bc963d2e321378706edd8`, giữ nguyên mã codec, RL, test, cấu hình và giấy phép. Notebook được chuyển sang repo này và ghim vào commit mới. Tài liệu `README_RL.md` và các notebook ngoài luồng Mini là phần lưu trữ, không phải hướng dẫn chạy khuyến nghị.

## VVenC và giấy phép

Mã VVenC và thư viện đi kèm giữ nguyên thông tin bản quyền, LICENSE và giấy phép trong `thirdparty/`. [Tài liệu VVenC gốc](README_VVENC.md).
