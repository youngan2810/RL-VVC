# RL-VVC

Ánh xạ phương pháp RL-RC-DoT sang H.266/VVenC. Repo chính:
**https://github.com/youngan2810/RL-VVC**, nhánh **main**

Đây là hướng dẫn chính cho input, training, resume và evaluation. Đặc tả
state/action/reward và các khác biệt với paper nằm trong
[METHOD_MAPPING.md](scripts/rl_rc_dot/METHOD_MAPPING.md)

## Phạm vi và phiên bản hiện tại

### Sửa riêng action — 17/11/2026

QPDev1 Gaussian đã chạy đủ 20.480 tương tác nhưng mean action dưới 0,084 QP,
nên toàn bộ Validation giữ bitstream như VVC. Giữ bản đó làm đối chứng.
Profile sửa **`qpdev1-qbins`** chỉ đổi đầu action và ánh xạ: PPO MultiCategorical
chọn ΔQP nguyên {-2,-1,0,1,2}; các ô trong CTU32 đồng nhất để không triệt tiêu
trong QG. State, extractor, Critic, Auxiliary, reward, optimizer và dataset giữ nguyên

Dùng [notebook ActionFix Short](notebooks/VVC_RL_QPDev1_ActionFix_Short_Kaggle.ipynb):
`MODE='new'`, 5.120 tương tác đầu, Train16/Val4, T4×2, cùng MOV input cũ.
Không resume checkpoint Gaussian. Chỉ chạy training sau gate map/QG/QP,
bitstream và VVdeC PASS; chưa tự chạy dài hoặc Test30. Validation có Random,
No-op/anchor và policy đã train. SAFE lưu chẩn đoán effective QP và reward

Chi tiết đường action, phương án so sánh, kiểu dữ liệu và giới hạn nằm trong
[audit kỹ thuật](scripts/rl_rc_dot/METHOD_MAPPING.md#integer-action-repair-audit-2026-11-17).
Qua kiểm thử cơ chế không đồng nghĩa học được policy tốt hơn; phải đọc kết quả
short Validation trước khi quyết định training tiếp

### Phương pháp đề xuất QP Development v1

Notebook phát triển khuyến nghị: [VVC_RL QPDev1 Train + Validation](notebooks/VVC_RL_QPDev1_Train_Validation_Kaggle.ipynb)

- **PPO-Clip + GAE**, Gaussian36: một offset toàn frame giới hạn ±2 và 35
  giá trị tạo phần bù không gian giới hạn ±4; tổng ΔQP nằm trong ±6. Actor
  convolution giữ 35 đặc trưng không gian và có hai đầu frame/vùng. Policy
  xác định ban đầu sinh map 0; sampling training có độ lệch chuẩn khởi tạo exp(-1,8)
- State **32 global + 6 bản đồ**: giữ 16+4 cũ, bổ sung độ lớn trung bình/std
  trước chuẩn hóa, target/ngân sách bit, lịch sử AU, QPA tuyệt đối và action
  được yêu cầu trước đó trong coding order. Không dùng detector làm đầu vào policy
- Mục tiêu clip: `mean(frame F1) - 0.1*abs(log(final bitrate / target))`.
  Reward bước dùng `F1/N` cộng chênh lệch chi phí bitrate prefix; với **gamma=1**,
  tổng reward bằng đúng mục tiêu Validation cuối clip. Frame không TP/FP/FN
  có F1=0. F1 là bảo toàn pseudo-label, không phải AP trên annotation thật
- Auxiliary mới: F1 cục bộ trên diện tích box, gồm vùng đối tượng nguồn bị
  bỏ sót và vùng false-positive; target có cùng matching một-một với metric
- Validation có **initial policy / anchor / zero / learned**, chọn checkpoint
  theo mục tiêu clip. Nếu training không cải thiện, best có thể vẫn là initial
- Xuất learning curves, CSV theo video/bitrate và hình **ΔQP yêu cầu / QP thực
  final CU**. Trace final CU giữ ba POC đầu/giữa/cuối mỗi clip Validation để
  giới hạn dung lượng; trajectory/action giữ đủ frame. Không coi ΔQP khác 0
  hoặc test kỹ thuật PASS là hiệu quả detection

Phiên đầu: `AGENT_PROFILE='qpdev1'`, `PHASE='trial'`, `MODE='new'`;
20.480 tương tác, Train16/Val4, 256 frame, Val50/120/200, collector T4×2,
PPO CPU, tối đa 11h45 gồm setup. Không chạy Test30 khi chọn cấu hình

Đối chứng cùng ngân sách: phiên riêng `AGENT_PROFILE='paper'`, `MODE='new'`.
`qpdev-reward` thay reward+aux, `qpdev-state` thêm state+extractor; đây là
so sánh theo giai đoạn, không cô lập từng thay đổi như ablation đầy đủ.
Giữ mốc trước phát triển ở commit `6ea5c054d8435376bf086a620d99b21345c391a7`

Screening đã định trước: best đã train tốt hơn initial và anchor về objective;
clip F1 macro tăng ít nhất 0,2 điểm %, bitrate macro tăng không quá 1%,
p95 tăng không quá 5%, và >1% ô có |ΔQP yêu cầu| ≥0,5. Đây là ngưỡng
thử nghiệm để quyết định chạy tiếp, chưa chứng minh mAP/BD-rate hoặc ý nghĩa
thống kê. Cần kiểm tra nhiều seed và Test chưa dùng để tinh chỉnh

**Không resume checkpoint trước phát triển vào QPDev.** Muốn tiếp tục một run
giữ nguyên profile/phase/source/config, đặt MODE=resume và thêm SAFE đúng run.
SAFE chứa checkpoint, metrics, validations, reports và bằng chứng trajectory.
Các hướng dẫn adapter/3 dưới đây mô tả đối chứng lịch sử; QPDev dùng mục tiêu,
state và action đã nêu ở mục này

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
