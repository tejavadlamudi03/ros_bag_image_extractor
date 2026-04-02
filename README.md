# 🎥 ROS2 Bag Image Extractor

A Python script to extract, filter, and classify images from a **ROS2 `.db3` bag file** recorded with an **Intel RealSense** camera. Automatically sorts frames into four categories using blur detection, perceptual hashing, and YOLOv8 object detection.

---

## 📁 Output Structure

\```
Images/
├── sharp/        ← Unique + sharp + contains car/traffic light ✅
├── blurry/       ← Unique + blurry + contains car/traffic light ⚠️
├── duplicates/   ← Near-identical to a previously saved frame 🔁
└── no_object/    ← No car or traffic light detected 🚫
\```

---

## 📦 Requirements

- Python 3.12
- Ubuntu 22.04 (tested)

---

## ⚙️ Installation

\```bash
/usr/bin/python3.12 -m pip install -r requirements.txt --ignore-installed
\```

---

## 🚀 Usage

\```bash
/usr/bin/python3.12 extract_images.py
\```

---

## 🎛️ Parameter Tuning Guide

| Parameter | Tune UP if... | Tune DOWN if... |
|---|---|---|
| \`BLUR_THRESHOLD\` | Too many blurry images pass | Too many sharp images rejected |
| \`REGION_THRESHOLD\` | Partial blur still passes | Too strict on regions |
| \`BLUR_REGION_MAX\` | Partial blur still passes | Too many images rejected |
| \`HASH_THRESHOLD\` | Too many frames marked duplicate | Near-identical frames not caught |
| \`CONF_THRESHOLD\` | False detections appearing | Real objects being missed |

---

## 🤖 YOLO Model

Uses **YOLOv8m** (~91.9% mAP50 on traffic lights). Weights downloaded automatically on first run.

| ID | Class |
|---|---|
| \`2\` | Car |
| \`9\` | Traffic Light |

---

## 📄 License

MIT License
