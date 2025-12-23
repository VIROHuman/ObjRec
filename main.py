"""
Real-time Assistive Object Detection Application for Visually Impaired Students
Uses YOLOv8n to detect objects via webcam and provides audible feedback.
"""

import cv2
import time
import threading
from collections import defaultdict
from ultralytics import YOLO
import pyttsx3
import numpy as np


class ObjectDetectionApp:
    def __init__(self, camera_index=0, frame_skip=5, confidence_threshold=0.6, speech_cooldown=5.0):
        """
        Initialize the object detection application.
        
        Args:
            camera_index: Index of the webcam (default: 0)
            frame_skip: Process every Nth frame (default: 5 for performance)
            confidence_threshold: Minimum confidence for detections (default: 0.6)
            speech_cooldown: Seconds to wait before repeating same object (default: 5.0)
        """
        self.camera_index = camera_index
        self.frame_skip = frame_skip
        self.confidence_threshold = confidence_threshold
        self.speech_cooldown = speech_cooldown
        
        # Initialize YOLO model
        print("Loading YOLOv8n model...")
        self.model = YOLO('yolov8n.pt')  # Downloads automatically if not present
        print("Model loaded successfully!")
        
        # Initialize TTS engine
        print("Initializing text-to-speech engine...")
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speech rate (words per minute)
        self.tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        
        # Try to set a more natural voice (if available)
        voices = self.tts_engine.getProperty('voices')
        if voices:
            # Prefer female voice if available, otherwise use first available
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            else:
                self.tts_engine.setProperty('voice', voices[0].id)
        
        # Speech cooldown tracking
        self.last_spoken = {}  # {object_name: last_time_spoken}
        self.speech_lock = threading.Lock()
        self.is_speaking = False
        
        # Frame counter for skipping frames
        self.frame_count = 0
        
        # Initialize webcam
        self.cap = None
        self.init_camera()
    
    def init_camera(self):
        """Initialize the webcam."""
        print(f"Initializing webcam (index {self.camera_index})...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open webcam at index {self.camera_index}")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Webcam initialized successfully!")
    
    def get_position(self, bbox_center_x, frame_width):
        """
        Determine if object is on Left, Center, or Right of the frame.
        
        Args:
            bbox_center_x: X coordinate of bounding box center
            frame_width: Width of the frame
            
        Returns:
            str: 'Left', 'Center', or 'Right'
        """
        third = frame_width / 3
        if bbox_center_x < third:
            return "Left"
        elif bbox_center_x < 2 * third:
            return "Center"
        else:
            return "Right"
    
    def speak(self, text):
        """
        Speak text in a non-blocking manner using a separate thread.
        
        Args:
            text: Text to speak
        """
        def speak_thread():
            with self.speech_lock:
                if self.is_speaking:
                    return  # Skip if already speaking
                self.is_speaking = True
            
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            finally:
                with self.speech_lock:
                    self.is_speaking = False
        
        # Start speaking in a separate thread
        thread = threading.Thread(target=speak_thread, daemon=True)
        thread.start()
    
    def should_speak(self, object_name):
        """
        Check if we should speak the object name based on cooldown.
        
        Args:
            object_name: Name of the detected object
            
        Returns:
            bool: True if we should speak, False otherwise
        """
        current_time = time.time()
        
        with self.speech_lock:
            if object_name in self.last_spoken:
                time_since_last = current_time - self.last_spoken[object_name]
                if time_since_last < self.speech_cooldown:
                    return False  # Still in cooldown
            
            # Update last spoken time
            self.last_spoken[object_name] = current_time
            return True
    
    def process_detections(self, results, frame):
        """
        Process YOLO detection results and provide audio feedback.
        
        Args:
            results: YOLO detection results
            frame: Current video frame
        """
        frame_width = frame.shape[1]
        detected_objects = []
        
        # Process each detection
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get confidence and class
                    confidence = float(box.conf[0])
                    if confidence < self.confidence_threshold:
                        continue
                    
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox_center_x = (x1 + x2) / 2
                    
                    # Determine position
                    position = self.get_position(bbox_center_x, frame_width)
                    
                    # Store detection info
                    detected_objects.append({
                        'name': class_name,
                        'confidence': confidence,
                        'position': position,
                        'bbox': (int(x1), int(y1), int(x2), int(y2))
                    })
        
        # Sort by confidence (highest first) and speak the most confident detection
        if detected_objects:
            detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
            top_detection = detected_objects[0]
            
            # Create speech text with position
            object_name = top_detection['name']
            position = top_detection['position']
            speech_text = f"{object_name} {position}"
            
            # Check cooldown and speak
            if self.should_speak(object_name):
                print(f"Speaking: {speech_text} (confidence: {top_detection['confidence']:.2f})")
                self.speak(speech_text)
        
        return detected_objects
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame: Video frame
            detections: List of detection dictionaries
            
        Returns:
            frame: Frame with drawn detections
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['name']
            confidence = det['confidence']
            position = det['position']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label text
            label = f"{class_name} {confidence:.2f} ({position})"
            
            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return frame
    
    def run(self):
        """Main application loop."""
        print("\n" + "="*50)
        print("Object Detection Application Started")
        print("Press 'q' to quit")
        print("="*50 + "\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from webcam")
                    break
                
                # Process every Nth frame for performance
                self.frame_count += 1
                if self.frame_count % self.frame_skip != 0:
                    # Still show the frame even if not processing
                    cv2.imshow('Object Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                # Run YOLO inference
                results = self.model(frame, verbose=False)
                
                # Process detections and get audio feedback
                detections = self.process_detections(results, frame)
                
                # Draw detections on frame
                frame_with_detections = self.draw_detections(frame.copy(), detections)
                
                # Display frame
                cv2.imshow('Object Detection', frame_with_detections)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")


def main():
    """Main entry point."""
    try:
        # Create and run the application
        app = ObjectDetectionApp(
            camera_index=0,        # Change if you have multiple cameras
            frame_skip=5,          # Process every 5th frame (adjust for performance)
            confidence_threshold=0.6,  # Minimum confidence for detections
            speech_cooldown=5.0    # Seconds between repeating same object
        )
        app.run()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

