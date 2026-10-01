# Smart Classroom AI Exam Proctoring

A prototype AI‑based exam proctoring system that uses a webcam and computer‑vision models to monitor students in real time.

## Features
- Real‑time webcam monitoring
- Face‑based student detection
- Multiple student identification
- Student tracking (lightweight centroid/face‑based)
- Head‑turn based looking‑around detection
- Mobile phone detection
- Real‑time warning / alert system
- Fully offline / local processing

## Tech Stack
- Python 3.10+
- OpenCV
- YOLO (Ultralytics)
- Computer Vision utilities

## Project Structure
```
Smart-Classroom-Cheating-Detection/
├─ app.py                 # Main Streamlit / OpenCV application entry point
├─ models/
│   └─ best.pt            # Trained YOLO weights (~5 MB)
├─ dataset/
│   └─ data.yaml          # Dataset configuration (if needed)
├─ requirements.txt       # Python dependencies
└─ README.md
```

## Getting Started

### 1. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
python app.py
```
A webcam is required. The app will open a window showing the live feed with detections and alerts.

## Current Limitations
- Tracking uses a simple centroid/face‑based approach; identities can swap when faces overlap.
- Mobile‑phone detection depends on the trained YOLO classes and may produce false positives/negatives.
- Tested on a single‑camera setup; multi‑camera support is not implemented.
- This is a **prototype** for research / demonstration purposes only.

## License
MIT (or specify your license)