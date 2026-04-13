# Underwater Swimmer Tactile Sensing

Computer vision pipelines for soft skin sensor analysis — predicting both **contraction depth** and **crosshead displacement** from raw camera frames using fine-tuned ResNet18 regression models synchronized with MTS load-frame data.

---

## Modules

### [`contraction_prediction/`](contraction_prediction/)

Predicts **skin contraction depth (mm)** from speckle-pattern image analysis.

- Captures synchronized video from a camera-equipped soft skin sensor
- Extracts speckle patterns and correlates frames with MTS crosshead + load data
- Fine-tunes ResNet18 for regression across a ~60 mm contraction range

See [`contraction_prediction/README.md`](contraction_prediction/README.md) for the full pipeline walkthrough.

---

### [`displacement_test/`](displacement_test/)

Predicts **crosshead displacement (mm)** directly from raw video frames, validated against unseen test footage.

#### Pipeline

```
raw data/                        dataset/                     model/
  video_20260403_150500.mp4  →     train/frames/   →   best_model.pth
  DAQ- Crosshead - (Timed).csv     train/dataset.csv    model_stats.json
                                   test/frames/
```

1. **`build_dataset.py`** — Extracts frames from the first **1:19** of video as the training set. Synchronizes each frame with the DAQ crosshead reading (1-second offset between DAQ start and video start). Remaining frames (1:19 onward) are saved unlabeled for test time.

2. **`train_model.py`** — Fine-tunes ResNet18 on 395 labeled frames (80/20 train/val split). Trained with aggressive augmentation given the small dataset size.

3. **`predict.py`** — Runs the trained model on unseen test frames and prints predicted vs. actual displacement per frame.

4. **`make_overlay_video.py`** — Cuts a clip from the original video and burns predicted + actual displacement onto each frame as an overlay.

#### Results — Unseen Frames (1:20–1:40)

| Metric | Value |
|--------|-------|
| Displacement range | 0 to −5.00 mm |
| Val MAE (best checkpoint) | **0.294 mm** |
| Test MAE (frames 1:20–1:40) | **0.350 mm** |
| Max error | 1.080 mm |

The model correctly tracks the full compression and release cycle without ever seeing these frames during training.

#### Overlay Video

[`assets/overlay_1m19s_to_1m39s.mp4`](displacement_test/assets/overlay_1m19s_to_1m39s.mp4) shows the model running on completely unseen footage from 1:19 to 1:39. Each frame displays:

- **Actual displacement** (from DAQ ground truth)
- **Predicted displacement** (from model)
- **Per-frame error**

#### File Structure

```
displacement_test/
├── build_dataset.py          # Extract + label frames from video + DAQ
├── train_model.py            # ResNet18 regression training
├── predict.py                # Predict on test frames, print comparison table
├── make_overlay_video.py     # Burn predicted/actual onto video clip
├── raw data/
│   ├── video_20260403_150500.mp4
│   └── DAQ- Crosshead, … - (Timed).csv
├── dataset/
│   ├── train/                # 395 labeled frames + dataset.csv
│   └── test/                 # 125 unlabeled frames
├── model/                    # Trained weights + normalization stats
└── assets/
    └── overlay_1m19s_to_1m39s.mp4
```

---

## Installation

```bash
pip install opencv-python torch torchvision numpy pandas matplotlib pillow
```
