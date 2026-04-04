# Object Tracking System

Real-time object detection and tracking using YOLOv26 + ONNX Runtime + ByteTrack.


## Features
- Real-time object detection using YOLOv26 exported to ONNX
- Persistent ID assignment across frames using ByteTrack
- Trail visualization showing movement path
- FPS monitoring

## Tech Stack
- Python
- OpenCV
- ONNX Runtime
- ByteTrack
- PyTorch

## Project Structure
```
object-tracking/
├── models/          ← place yolo26n.onnx here
├── src/
│   ├── __init__.py
│   ├── detector.py  ← preprocessing and detection
│   ├── tracker.py   ← tracking and visualization
│   └── utils.py     ← video I/O utilities
├── main.py          ← entry point
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup
```bash
git clone https://github.com/SalehAli2/Object_Tracking.git
cd Object_Tracking
pip install -r requirements.txt
```

Download `yolo26n.onnx` by exporting it yourself:
```python
from ultralytics import YOLO
model = YOLO('yolo26n.pt')
model.export(format='onnx', opset=17)
```
Place the exported file in the `models/` folder.

## Run
```bash
python main.py
```

## Model
- YOLOv26n exported to ONNX format
- Input: 640x640
- Output: bounding boxes in corner format [x1, y1, x2, y2, conf, class]
