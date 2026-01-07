# Real-time Assistive Object Detection Application

A real-time assistive application for visually impaired students that uses a webcam to detect objects in the environment and provides audible voice feedback.

## Features

- **Real-time Object Detection**: Uses YOLOv8n (pre-trained COCO model) for fast and accurate object detection
- **Audio Feedback**: Provides voice announcements of detected objects with their position (Left/Center/Right)
- **Smart Speech Cooldown**: Prevents repetitive announcements - waits 5 seconds before repeating the same object
- **Non-blocking TTS**: Text-to-speech runs in a separate thread to avoid blocking video feed
- **Confidence Filtering**: Only processes detections with confidence > 0.6 to reduce false positives
- **Performance Optimized**: Processes every 5th frame by default to maintain smooth video feed

## Requirements

- Python 3.10 or higher
- Webcam
- Windows/Linux/macOS

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

The first time you run the application, YOLOv8n model weights will be automatically downloaded.

## Usage

Run the application:

```bash
python main.py
```

### Controls

- **'q' key**: Quit the application

### Configuration

You can modify the following parameters in `main.py`:

- `camera_index`: Change if you have multiple cameras (default: 0)
- `frame_skip`: Process every Nth frame (default: 5 for better performance)
- `confidence_threshold`: Minimum confidence for detections (default: 0.6)
- `speech_cooldown`: Seconds to wait before repeating same object (default: 5.0)

## How It Works

1. **Initialization**: Loads YOLOv8n model and initializes webcam
2. **Frame Capture**: Captures frames from webcam
3. **Detection**: Runs YOLO inference on frames (every 5th frame by default)
4. **Filtering**: Filters detections by confidence threshold (>0.6)
5. **Position Estimation**: Determines if object is Left, Center, or Right based on bounding box position
6. **Audio Feedback**: Speaks the object name and position (e.g., "Chair Center")
7. **Cooldown Management**: Tracks when objects were last spoken to prevent repetition

## Technical Details

- **Model**: YOLOv8n (nano) - fastest YOLOv8 variant, suitable for real-time applications
- **TTS Engine**: pyttsx3 (offline, low-latency, cross-platform)
- **Threading**: TTS runs in separate daemon thread to avoid blocking video processing
- **Position Logic**: Frame divided into thirds (Left: 0-33%, Center: 33-66%, Right: 66-100%)

## Troubleshooting

- **Webcam not detected**: Try changing `camera_index` to 1 or 2
- **Low FPS**: Increase `frame_skip` value (e.g., 10 instead of 5)
- **Too many false positives**: Increase `confidence_threshold` (e.g., 0.7 or 0.8)
- **TTS not working**: Ensure pyttsx3 is properly installed and system has TTS voices available

## License

This project is provided as-is for educational and assistive purposes.


