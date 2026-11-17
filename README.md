# RL-VVC — Masked Dueling Double DQN (D3QN) for Task-Aware QP

> **Nhánh:** `experiment/d3qn-native-mdp-v2`  
> **Trạng thái:** source nghiên cứu; chưa xác nhận chạy thành công với VVenC/VVdeC/YOLO trên Kaggle, chưa có kết quả D3QN hay chứng minh tốt hơn VVenC QPA.  
> **Thuật toán chính:** **Masked Dueling Double DQN**, không phải PPO.

README này dành cho **nhánh D3QN**. PPO vẫn được giữ trên nhánh [`main`](https://github.com/youngan2810/RL-VVC/tree/main), trong notebook và module cũ để đối chứng lịch sử; không dùng `session.py` hay notebook PPO để huấn luyện D3QN.

**Đặc tả khoa học:** [D3QN_NATIVE_MDP_DESIGN.md](scripts/rl_rc_dot/D3QN_NATIVE_MDP_DESIGN.md).

## 1. Mục tiêu và phạm vi

Trước mắt điều khiển **ΔQP** phục vụ phát hiện xe bằng YOLO trên **VVenC/H.266**. D3QN điều chỉnh quanh native **QPA**; bitstream vẫn được tạo và giải mã bằng codec thực. **Partition (QT/BT/TT)** và **MTS** là hướng phát triển sau, chưa triển khai.

Không mặc định D3QN tốt hơn PPO: muốn so sánh thuật toán, phải huấn luyện PPO với **cùng MDP (state/action/reward)** của D3QN v2.

## 2. Dataset: BDD100K, giống bản PPO về split

| Split | Mini — dùng hiện tại | Full — chưa dùng |
|---|---:|---:|
| **Train** | **16 video** | 65 |
| **Validation** | **4 video** | 7 |
| **Test** | **30 video — giữ nguyên, chưa dùng** | 100 |

- Mini được chọn và khóa từ cùng [`dataset_lock.json`](scripts/rl_rc_dot/assets/dataset_lock.json), **split seed cố định `20261111`** ngay cả khi đổi training seed để chạy nhiều lần; không chia lại hoặc đụng Test để chọn mô hình.
- Các video `.mov` không có trong GitHub. Mount dữ liệu BDD100K tại Kaggle, chẳng hạn `/kaggle/input/datasets/robikscube/driving-video-with-object-tracking`.
- `VideoCatalog` lấy dataset theo `mode='mini'`, kiểm tra Train16/Val4. **Lịch train D3QN mới chọn đủ cả 16 video đúng một lần trong 16 episode đầu** (thứ tự tráo ngẫu nhiên nhưng tái lập), sau đó mới lặp; cửa sổ **256 frame** trong từng video vẫn chọn tại offset 0/128/256/384/512. Trong pilot 5.120 transitions = 20 episode, bảo đảm phủ Train16 nếu hoàn tất đủ.
- Validation mặc định **Val4 × 50/120/200 kbps = 12 video–rate points**, mỗi clip 256 frame đầu. Mỗi point còn chạy nhiều phương pháp đối chứng.
- Tiền xử lý chung: 480×320, 30 fps, YUV420p. Native VVenC: QPA bật, CTU32, MaxCuDQPSubdiv=2, 10-bit nội bộ.
- Full65/Val7 và Test30 **không được phép sử dụng** trong native learning-check environment của nhánh này; chỉ Train16/Val4.

## 3. State – Action – Reward riêng cho D3QN

**State:** 25 global features và 5 maps CTU32 kích thước 10×15: source energy, intra complexity, temporal propagation, native-QPA QP và semantic importance từ YOLO trên **frame nguồn trước encode**. Thêm legal-action mask 25 phần tử. Chi phí chạy YOLO trên nguồn là **chi phí encoder thực** phải báo cáo; không coi đây là semantic feature miễn phí khi triển khai.

**Action:** `Discrete(25)`, mỗi action là cặp `(ΔQP_ROI, ΔQP_BG)` thuộc `{-2,-1,0,+1,+2}²`. Action 12 = `(0,0)` giữ QPA. Action map nguyên CTU32 được chuyển trực tiếp sang 20×30 và gửi qua `NativeEpisode` của VVenC. **Không dùng ánh xạ 36 PPO bins.** Mask loại action tạo QP map trùng nhau; fixed-policy probe dùng action hợp lệ tương đương.

**Reward/objective (giả thuyết khởi tạo, chưa tối ưu):**

    J = mean(frame_F1) - 0.15 * (actual_bits/target_bits)
        - 3.0 * max(0, actual_bits/target_bits - 1.02)

Reward từng picture có tổng đúng bằng objective cuối clip khi `gamma=1`; kiểm tra trong environment. Đây là **soft bitrate penalty**, không phải hard constraint. `frame_F1` là **YOLO pseudo-label retention** (nguồn vs tái tạo), không phải mAP dựa annotation BDD100K.

**D3QN:** dueling value/advantage network, Double-Q target, experience replay, Huber loss, target synchronization, epsilon-greedy, gradient clipping và masking cho cả chọn action lẫn bootstrap. **Tập 25 action và hệ số reward chưa phải thiết kế tối ưu đã được thực nghiệm xác nhận.**

## 4. Các file D3QN (không phải source PPO đổi tên)

| File | Vai trò |
|---|---|
| [`d3qn_method.py`](scripts/rl_rc_dot/d3qn_method.py) | State/action/reward + semantic ROI + mask |
| [`d3qn_agent_v2.py`](scripts/rl_rc_dot/d3qn_agent_v2.py) | Mạng Dueling Double DQN, replay, optimizer, checkpoint |
| [`d3qn_native_env.py`](scripts/rl_rc_dot/d3qn_native_env.py) | Native environment, **không dùng** `VVenCEnv` PPO |
| [`d3qn_native_runner.py`](scripts/rl_rc_dot/d3qn_native_runner.py) | Probe, train/resume, Validation tách biệt, Mini16/Val4 |
| [`test_d3qn_native_v2.py`](scripts/rl_rc_dot/test_d3qn_native_v2.py) | Unit test cho MDP, masking và Double-Q |
| [`D3QN_NATIVE_MDP_DESIGN.md`](scripts/rl_rc_dot/D3QN_NATIVE_MDP_DESIGN.md) | Đặc tả kỹ thuật và giao thức đánh giá |

Hạ tầng dùng chung với hệ thống PPO trước đó chỉ để giữ tính nhất quán codec/benchmark: `data.py` (dataset), `codec.py` (NativeEpisode), `method.detection_assignments`, `action_audit.py`, `evaluate.verify_decoded`, VVenC C++. **PPO agent, PPO rollout, PPO action mapping và PPO reward không được sử dụng để train D3QN.**

## 5. Chạy trên Kaggle

Checkout đúng nhánh D3QN và chuẩn bị VVenC đã sửa native action bridge, VVdeC, YOLOv5, pinned weights, FFmpeg và dataset đã mount. Các đường dẫn `/path/to/...` là placeholder, **phải thay bằng đường dẫn thực**.

### A. Unit test + native preflight

    python -m unittest scripts.rl_rc_dot.test_d3qn_native_v2 -v
    python -m scripts.rl_rc_dot.preflight --encoder /path/to/repaired/vvencapp --decoder /path/to/vvdecapp --output /kaggle/working/d3qn_native_preflight --frames 32

**Chưa có kết quả PASS thực tế cho nhánh D3QN**. Phải kiểm tra native QG/lambda, bitstream, VVdeC parity trước training dài.

### B. Fixed-action probe (chạy trước training)

    python -m scripts.rl_rc_dot.d3qn_native_runner \
      --phase probe \
      --input-root /kaggle/input/datasets/robikscube/driving-video-with-object-tracking \
      --encoder /path/to/repaired/vvencapp --decoder /path/to/vvdecapp \
      --yolo-repo /path/to/pinned/yolov5 --weights /path/to/pinned/yolov5n.pt \
      --output /kaggle/working/d3qn_v2_probe_001 --device cuda:0 \
      --val-videos 4 --val-rates 50,120,200

Runner probe dùng 6 chính sách cố định, nên cấu hình đầy đủ = **4 video × 3 rate × 6 policy = 72 lượt encode**, mỗi lượt 256 frame. Kiểm tra nhanh có thể thêm `--val-videos 1 --val-rates 120` (chỉ smoke test, không phải Validation đầy đủ). Không kết luận cải thiện từ **cùng target bitrate**; phải khớp **bitrate thực**.

### C. D3QN pilot train (chỉ sau khi probe đạt)

    python -m scripts.rl_rc_dot.d3qn_native_runner \
      --phase train \
      --input-root /kaggle/input/datasets/robikscube/driving-video-with-object-tracking \
      --encoder /path/to/repaired/vvencapp --decoder /path/to/vvdecapp \
      --yolo-repo /path/to/pinned/yolov5 --weights /path/to/pinned/yolov5n.pt \
      --output /kaggle/working/d3qn_v2_train_001 --device cuda:0 \
      --train-steps 5120 --val-videos 4 --val-rates 50,120,200

Pilot 5.120 frame transitions tương ứng 20 episode 256-frame nếu chạy hết. Pha `train` **chỉ train và lưu checkpoint sau mỗi episode**, không tự chạy Validation dài. Có thể nối phiên bằng `--checkpoint-in /path/to/checkpoint.pt`, với cùng source/dataset/encoder/weights/protocol; `--train-steps` là số transitions **bổ sung** của phiên. D3QN lưu đầy đủ replay/optimizer/RNG để resume. Phiên có `--max-hours` + `--session-start-utc` sẽ dừng mềm ở **ranh giới episode** để giữ checkpoint; source Kaggle chưa được kiểm chứng thực tế.

### D. Validation D3QN — chạy riêng sau training

```bash
python -m scripts.rl_rc_dot.d3qn_native_runner \
  --phase validate \
  --input-root /kaggle/input/datasets/robikscube/driving-video-with-object-tracking \
  --encoder /path/to/repaired/vvencapp --decoder /path/to/vvdecapp \
  --yolo-repo /path/to/pinned/yolov5 --weights /path/to/pinned/yolov5n.pt \
  --checkpoint-in /path/to/verified/D3QN/checkpoint.pt \
  --output /kaggle/working/d3qn_v2_validation_001 --device cuda:0 \
  --val-videos 4 --val-rates 50,120,200
```

Validation = **4×3×7 = 84 lượt encode**, gồm sáu đối chứng + D3QN, khá nặng; có thể chia `--val-start 0 --val-videos 2`, rồi `--val-start 2 --val-videos 2`, mỗi phiên dùng cùng checkpoint/source và giữ toàn bộ output. Đây là pilot/đánh giá, **chưa phải chứng minh hội tụ hoặc hơn QPA ở bitrate khớp thực**.

## 6. Output, kiểm chứng và giới hạn

- `contract.json`: dataset/source/codec/weights hashes và tham số run.
- `train_metrics.json`, `train_summary.json`, `checkpoint.pt`: chỉ khi `--phase train`.
- `validation.json` và `validation_partial.json`: mỗi video/rate/policy, measured kbps, mean frame F1, objective, VVdeC parity.
- `trajectory.jsonl.gz`, native logs và action audit: AU bits từng POC, lựa chọn QP, reconstruction hash, QG/lambda (Validation).

**Chưa có:** native test PASS trong môi trường Kaggle, best-checkpoint selection theo Val, multi-seed, Test30, ground-truth mAP hoặc BD-rate. D3QN hiện đã có checkpoint/resume theo episode (full replay/optimizer/RNG) nhưng chưa có bằng chứng đã resume thành công trên Kaggle. Để xác định thuật toán D3QN vượt PPO, phải làm ablation với **cùng state/action/reward v2**; các kết quả PPO cũ chỉ là baseline lịch sử.

## 7. Lưu trữ tài liệu PPO

README phương pháp **PPO** trước đây: [nhánh main](https://github.com/youngan2810/RL-VVC/blob/main/README.md). [`METHOD_MAPPING.md`](scripts/rl_rc_dot/METHOD_MAPPING.md), `session.py`, `qp_development.py` và notebook PPO là tài liệu/triển khai **cũ**, không phải entry point cho D3QN v2.

Giữ nguyên LICENSE/COPYRIGHT/AUTHORS và giấy phép thư viện bên thứ ba.


## 8. Learning-check: theo dõi agent có học đúng kỳ vọng không

Notebook pilot theo mốc: [VVC_D3QN_Mini16_Val4_LearningCheck_Kaggle.ipynb](notebooks/VVC_D3QN_Mini16_Val4_LearningCheck_Kaggle.ipynb)
(sẽ được thêm sau khi hoàn thiện notebook). **Không dùng Test30**.

- 16 video Train: 16 episode đầu phủ đủ 16 ID; 20 episode = 5.120 transitions.
- 4 video Val: đo **cùng 4 video** và **120 kbps target** ở các mốc
  1.024 / 2.048 / 3.072 / 4.096 / 5.120 transitions;
  không dùng Validation để cập nhật gradient.
- Baseline Val4 gồm VVenC QPA và heuristic global ΔQP=-1, đo trước train.
- Ở mỗi mốc, đánh giá **chính sách greedy (epsilon=0)** trên Val4,
  đối chiếu same-video F1, actual kbps và objective; các mốc này
  là **diagnostics**, không phải đánh giá Test.
- Theo dõi mean TD loss, gradient norm, valid-action Q-gap,
  số action greedy khác nhau, train vs Val F1/bitrate/objective;
  kiểm tra collapse hoặc chỉ tăng F1 nhờ tăng bitrate.
- Chỉ khi learning-check cho tín hiệu hợp lý mới xem xét đa bitrate
  (50/120/200 kbps trên đúng Val4), multi-seed, so sánh PPO cùng MDP.
- `d3qn_native_runner` hỗ trợ `--val-methods` để
  monitor chỉ `trained_d3qn`; dùng `--phase probe --val-methods
  qpa_anchor,global_minus1` để tạo baseline một lần.

**Cảnh báo:** Kết quả monitor ở cùng target rate **không phải** kết quả
rate-matched/BD-rate. Các threshold chỉ được dùng để quyết định bước
thử nghiệm tiếp theo, không dùng Test30 để tuning.
