# RL-VVC

Ánh xạ phương pháp RL-RC-DoT sang H.266/VVenC. Repo chính:
**https://github.com/youngan2810/RL-VVC**, nhánh **main**

Đây là hướng dẫn chính cho input, training, resume và evaluation. Đặc tả
state/action/reward và các khác biệt với paper nằm trong
[METHOD_MAPPING.md](scripts/rl_rc_dot/METHOD_MAPPING.md)

## Phạm vi và phiên bản hiện tại

- Điều khiển bản đồ ΔQP bằng PPO; chưa triển khai phát triển Partition/MTS
- Đã sửa lỗi cộng lặp ΔQP ở inter QG, hợp đồng `native-qpa-single-delta/2`.
  QG con dùng QP nền QPA riêng; mỗi QG chỉ cộng một lần ΔQP của vùng đó
- **Bắt đầu run mới bằng `MODE='new'`**. Không nạp checkpoint trước bản sửa QG
- Kiểm tra local: 66 test, 65 đạt, 1 RGB test bỏ qua vì thiếu shared library
  trên Windows. Test kỹ thuật không chứng minh chất lượng học hoặc BD-rate;
  preflight Kaggle vẫn bắt buộc trước training
- Source nhập từ snapshot `af01dd3ac10a68ff8a4bc963d2e321378706edd8`.
  Notebook đã chuyển sang repo này và ghim commit để tái lập

## So sánh thực nghiệm

Runner hiện đánh giá **VVC** (anchor CTU32) và **VVC_RL**. H264_RL chưa tích hợp
vào runner này. Ưu tiên xin checkpoint của bản tái hiện bên thứ ba để đánh giá
H264_RL; hiện chưa có file checkpoint tải được. Không gọi bản tái hiện là model
chính thức của tác giả paper

Các hệ thống phải dùng cùng video Test, đoạn frame, kích thước, detector,
metric và bitrate thực. Công bố dữ liệu/ngân sách training của từng agent và
kiểm tra không trùng Train/Test. Anchor là **CTU32/DQP2/FPP0**, không dùng các
kết quả stock CTU128 cũ làm đối chứng cùng cấu hình

## Dataset và input

- Full lock: Train65 / Validation7 / Test100
- Mini: **Train16 / Validation4 / Test30**, tập con xác định theo seed, phân
  tầng theo số car boxes; ba tập không trùng nhau. Đây không phải phân tầng
  đã xác minh theo ngày/đêm/thời tiết
- Lock và cấu hình detector đã nằm trong source. Dataset lock SHA256:
  `f7a600f501de622f66ace04bb0b12ec5da0dab988cd54732d614f6b16495aaea`
- Cần input **MOV BDD100K Train00 / Driving Video with Object Tracking**
  chứa các video được chọn. `bdd100k.zip` và `snapshot.zip` cũ chứa metadata,
  không thay thế MOV. Notebook kiểm tra hash MOV trước khi build codec
- Bật Internet và **GPU T4×2**. Setup giữ CUDA Torch đang có, pin source
  VVdeC/YOLO và kiểm tra weights; không so binary Kaggle với SHA Windows cũ

Train Mini chọn ngẫu nhiên đoạn 256 frame bắt đầu tại 0/128/256/384/512.
Validation/Test dùng đoạn đầu theo độ dài cấu hình. Cần ghi rõ giới hạn này
khi báo cáo, không gọi là đánh giá nguyên video.

## Chọn notebook

| Bước | Notebook | Ngân sách mặc định |
|---|---|---|
| Kiểm tra kỹ thuật | [Mini Pilot](notebooks/RL_RC_DoT_VVC_Mini_Pilot_Kaggle.ipynb) | 256 tương tác; Val2 × 2 rate × 64 frame |
| Kiểm tra khả năng học | [Mini Trial](notebooks/RL_RC_DoT_VVC_Mini_Trial20480_Kaggle.ipynb) | 20.480 tương tác; Val4 × 2 rate × 256 frame |
| Đánh giá sau khi chốt checkpoint | [Mini Test30](notebooks/RL_RC_DoT_VVC_Mini_Test30_Kaggle.ipynb) | 30 video × 5 rate × 256 frame; 15 cặp bổ sung/phiên |

Pilot/Trial training dùng hai collector: YOLO trên GPU0/GPU1; PPO mặc định CPU;
VVenC là CPU encoder. Test30 hiện dùng **GPU0 tuần tự**, chưa song song hai GPU.

Phiên đầu dùng `MODE='new'`. Pilot, Trial và `PHASE='full'` trong notebook Mini
có hợp đồng rollout/validation khác nhau: không đổi phase rồi resume checkpoint
của phase cũ. **`PHASE='full'` trong Mini chỉ là Mini chạy dài hơn, vẫn 16/4/30**;
không tự chuyển sang Train65. Muốn mở rộng Full65 phải tạo thí nghiệm riêng.

Mỗi notebook khóa một source commit. Không sửa pin/hyperparameter giữa các
phiên resume. Lưu notebook và log setup của phiên đã chạy để tái lập kết quả.

## Training và cách đánh giá trong phiên

Entry point: `python -m scripts.rl_rc_dot.session`. `train.py` không có
`--trajectory` chuyển sang runner này; pilot DOT_PROXY cũ đã ngừng hỗ trợ.

Training dùng YOLOv5n để so detection trên frame tái tạo với frame gốc làm
pseudo-ground-truth. Reward gốc: `20*Precision - abs(log(prefix_rate/target))`.
PPO nhận 0,05 lần reward; validation/report giữ đơn vị gốc. Mỗi bước là một
picture trong coding order, không phải một video. Không quy về 975 episode
hoặc 3 epoch dataset; `PPO_EPOCHS=3` là lượt tối ưu trên mỗi rollout.

Validation định kỳ so **anchor / zero-action / learned policy**, lưu Precision,
Recall, F1, PSNR, bitrate thực và reward. Chọn best bằng macro mean reward
validation hoàn chỉnh. `latest` phục vụ tiếp tục training; nó có thể kém `best`.
Không dùng Test30/Test100 để chỉnh reward hoặc chọn hyperparameter.

Qua Pilot không đồng nghĩa có hiệu quả học. Trước khi chạy dài, xem KL/clip
fraction, action spread, auxiliary diagnostics, explained variance và xu hướng
validation so với bitrate thực. Không suy chất lượng học chỉ từ test PASS.

## Resume và lưu bằng chứng

Upload gói **`*_KAGGLE_SAFE.zip`** của đúng run; `session_payload.bin` giữ nguyên
ZIP bên trong khi Kaggle giải nén Input. Giữ mọi export cũ để đối chiếu.

- Adapter/2, V3/V4 và DOT_PROXY **không tương thích adapter/3**; không bỏ assert
  để nạp chúng. Với adapter/3 bắt đầu run mới
- Resume giữ source/profile/dataset/seed/workers/rollout/validation/PPO settings;
  được đổi ngân sách bổ sung. Source/config khác sẽ bị từ chối
- Phục hồi model, optimizer, counters và RNG, rồi mở codec episode/rollout mới;
  không phục hồi giữa một picture/reference buffer đang encode
- Dừng mềm dành thời gian export; VM bị tắt đột ngột chỉ giữ checkpoint đã
  commit còn trên đĩa. Không có bảo đảm ZIP kịp tạo khi VM bị kill
- Validation control cache không nằm trong export hiện tại; sau resume control
  có thể được tính lại. Đây là chi phí bổ sung, không phải policy bị mất

Output chính: `method.json`, `session.json`, `dataset_selection.json`,
`progress.json`, `checkpoints/`, `metrics/`, `validations/`, native/trajectory
evidence và SAFE export. SHA/ràng buộc method được kiểm tra trước khi load.

## Test30 và chỉ số báo cáo

Mặc định `TEST_RATES='50,80,120,160,200'`: **150 cặp**. Có thể chọn
`'50,120,200'` để khảo sát nhanh **90 cặp**, không báo BD-rate. Không đổi rates
hay checkpoint khi resume. Độ dài Mini Test cố định 256 frame/video.

Report hiện có: Precision, Recall, F1, PSNR, bitrate thực, paired deltas,
đường RD, BD-Precision và BD-rate nếu đủ năm điểm hợp lệ có miền giao nhau.
Không ngoại suy khi hai đường không giao miền; kết quả không xác định để null.
Không gọi BD-Precision là mAP. Chưa có segmentation/tracking/mAP trong runner.
Các chỉ số detection ở đây là mức bảo toàn pseudo-label, không phải đánh giá
trực tiếp so annotation thật của BDD100K.

Thời gian setup tính trong session budget khi dùng notebook mới. Chọn resume
Input theo đúng checkpoint, source, seed, frame và rate. Test30 hiện tại chỉ đánh giá VVC và VVC_RL; chưa gồm H264_RL.

## Lệnh kiểm tra kỹ thuật

```bash
python -m unittest discover -s scripts/rl_rc_dot -p 'test_*.py' -v
python -m scripts.rl_rc_dot.preflight --encoder /path/vvencapp \
  --decoder /path/vvdecapp --output /path/new_preflight --frames 32
```

Linux/Kaggle runner còn bắt buộc kiểm tra RGB conversion byte-exact với
FFmpeg. Native preflight kiểm tra state, action, CU coverage, AU bits, POC và
decoder parity. Bảng bằng chứng và giới hạn nằm trong
[đặc tả kỹ thuật](scripts/rl_rc_dot/METHOD_MAPPING.md#verification-and-experiment-history).

## Tài liệu và giấy phép

- [Paper RL-RC-DoT](https://arxiv.org/abs/2501.12216)
- [Bản tái hiện độc lập H.264](https://github.com/Puiching-Memory/RL-RC-DoT)
- [Tài liệu build và sử dụng VVenC gốc](https://github.com/fraunhoferhhi/vvenc)
- Giữ nguyên LICENSE, COPYRIGHT, AUTHORS và giấy phép thư viện trong `thirdparty/`

Các README của những giai đoạn thử nghiệm cũ đã được gộp hoặc bỏ. Các comment
còn nhắc `README_RL_RC_DOT.md` trỏ tới hướng dẫn đã gộp vào file này. Notebook
ngoài luồng Mini là tài liệu lưu trữ, không phải luồng chạy khuyến nghị
