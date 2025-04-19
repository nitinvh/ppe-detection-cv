# PPE Detection with YOLOv3 and Deep SORT

## Overview
Industrial environments often mandate the use of Personal Protective Equipment (PPE) such as helmets, harnesses, and goggles to protect worker safety. However, continuous human supervision can be prone to oversight, especially during long shifts. This project introduces an AI-driven approach using real-time video analysis to detect and track individuals not adhering to PPE requirements.

The system utilizes pre-installed CCTV cameras to automatically identify workers without proper safety gear and issues live alerts to supervisors, reducing the possibility of human error.

## Key Features
- Real-time input via connected cameras
- Helmet detection using YOLOv3
- Multi-person tracking using Deep SORT with unique IDs
- Alerts triggered for individuals without helmets after a set duration (e.g., 5 seconds)
- Option to display alerts in a global message console

## How It Works
1. **Video Feed**: Captured via USB cameras
2. **Object Detection**: YOLOv3 identifies individuals with and without helmets
3. **Tracking**: Deep SORT assigns consistent IDs across frames
4. **Alerts**: Notifications appear if a person is repeatedly detected without PPE


## Current Capabilities
- Helmet detection model is trained and operational
- USB cameras are supported (other input types to be added)
- Modular tracking setup
- Alerts integrated into GUI interface

### Planned Improvements
- Detection of additional PPE (e.g., goggles, vests)
- Support for RTSP/HTTP-based camera streams
- Mobile notifications (e.g., SMS or app-based alerts)

## Getting Started

### Environment Setup
Using `conda` is the preferred way to manage dependencies:
```bash
# Option 1: Create environment from YAML
conda env create -f environment.yml

# Option 2: Create using requirements.txt
conda create --name helmet-detection --file requirements.txt

# Activate environment
conda activate helmet-detection
```

### Required Files
Download the following files and place them in the root project directory:
- [YOLO Weights - full_yolo3_helmet_and_person.h5](https://1drv.ms/u/c/024d7625f12b47b2/QbJHK_Eldk0ggAL3AQAAAAAAExKZFaGcssUM5Q)
- [Deep SORT Embedding Model - mars-small128.pb](https://1drv.ms/u/c/024d7625f12b47b2/QbJHK_Eldk0ggAL4AQAAAAAA8rfjUd8TxK6_-Q)

### Run the Application
**With GUI (max 2 cameras):**
```bash
python predict_gui.py -c config.json -n 2
```

**Without GUI (unlimited cameras):**
```bash
python predict.py -c config.json -n 4
```

## Training the Detection Model

### 1. Dataset Preparation
- Collect images of people with and without helmets.
- Use [LabelImg](https://github.com/tzutalin/labelImg) to annotate in Pascal VOC format.

Directory Structure:
```
train_image_folder/
train_annot_folder/
valid_image_folder/
valid_annot_folder/
```

Each annotation file should match its image file by name.

### 2. Configuration
Edit the `config.json` file to include dataset paths, anchor boxes, label classes, and training hyperparameters. Example:
```json
{
  "model": {
    "min_input_size": 288,
    "max_input_size": 448,
    "anchors": [...],
    "labels": ["helmet", "person with helmet", "person without helmet"]
  },
  "train": {
    "train_image_folder": "train_image_folder/",
    "train_annot_folder": "train_annot_folder/",
    ...
  }
}
```

Download backend weights:
- [backend.h5](https://1drv.ms/u/c/024d7625f12b47b2/QbJHK_Eldk0ggAL5AQAAAAAA1JJB2XEu27RBmw) and place it in the root directory.

### 3. (Optional) Generate Custom Anchors
```bash
python gen_anchors.py -c config.json
```
Paste the output anchors into `config.json`.

### 4. Train the Model
```bash
python train.py -c config.json
```
The best model will be saved to the specified filename in `saved_weights_name`.

### 5. Live Inference
Use webcam feed to test live detection:
```bash
python predict.py -c config.json -n 1
```

## Credits
- YOLOv3 adaptation: [experiencor/keras-yolo3](https://github.com/experiencor/keras-yolo3)
- Deep SORT: [nwojke/deep_sort](https://github.com/nwojke/deep_sort)
- Training Data: [rekon/keras-yolo2](https://github.com/rekon/keras-yolo2)

---
Contributions, ideas, and enhancements are highly appreciated to take this project further!

