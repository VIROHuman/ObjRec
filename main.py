"""
Advanced Assistive Object Detection for Visually Impaired
Features: Preemption, Crowd Grouping, Persistence Tracking, Low Light Detection
"""

import cv2
import time
import threading
import queue
import subprocess
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import easyocr
import msvcrt
import sys

# --- 1. ROBUST AUDIO WORKER WITH PREEMPTION ---
class TTSWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.start()

    def run(self):
        while True:
            text = self.queue.get()
            if text is None: 
                break
            
            try:
                # Use Windows PowerShell to speak. This CANNOT freeze Python.
                escaped_text = text.replace("'", "''")
                cmd = f'PowerShell -Command "Add-Type –AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{escaped_text}\');"'
                subprocess.run(cmd, shell=True, capture_output=True)
            except Exception as e:
                print(f"Error in speech loop: {e}")
            finally:
                self.queue.task_done()

    def speak(self, text, preempt=False):
        """
        Add text to speech queue.
        
        Args:
            text: Text to speak
            preempt: If True, clear queue and speak immediately (for high priority warnings)
        """
        with self.lock:
            if preempt:
                # Clear the queue for high-priority warnings
                while not self.queue.empty():
                    try:
                        self.queue.get_nowait()
                        self.queue.task_done()
                    except queue.Empty:
                        break
            
            # Add to queue (limit size to prevent lag)
            if self.queue.qsize() < 3:
                self.queue.put(text)

# --- 2. MAIN APPLICATION WITH ADVANCED FEATURES ---
class SmartVisionApp:
    def __init__(self, source=0):
        """
        Initialize the Smart Vision application.
        
        Args:
            source: Video source - can be an integer (0 for webcam) or a string URL (for DroidCam)
                    Examples: 0, 1, "http://192.168.1.100:4747/video"
        """
        # Configuration
        self.conf_threshold = 0.6
        self.frame_skip = 5  # Process every 5th frame
        
        # High Priority Objects (trigger preemption)
        self.high_priority = {'car', 'bus', 'truck', 'stairs', 'traffic light'}
        
        # Priority Map: Lower number = Higher Priority
        self.priority_map = {
            'car': 1, 'bus': 1, 'truck': 1, 'stairs': 1, 'traffic light': 1,
            'person': 2,
            'chair': 3, 'desk': 3, 'door': 3, 'couch': 3,
            'cell phone': 4, 'bottle': 4, 'book': 4
        }

        # State tracking
        self.tts = TTSWorker()
        self.model = YOLO('yolov8n.pt')
        
        # Initialize video capture (supports both webcam index and DroidCam URL)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        
        # Print source information
        if isinstance(source, str):
            print(f"Connected to DroidCam: {source}")
        else:
            print(f"Connected to webcam: {source}")
        
        # Audio Cooldown
        self.last_spoken_time = defaultdict(float) 
        self.cooldown_seconds = 4.0
        
        # Persistence Tracking (Anti-Jitter): Track consecutive detections
        self.detection_counters = defaultdict(int)  # {label: consecutive_frame_count}
        self.persistence_threshold = 3  # Must be detected 3 consecutive frames
        
        # Low Light Mode
        self.low_light_threshold = 30  # Brightness threshold (0-255)
        self.low_light_cooldown = 10.0  # Seconds between low light warnings
        self.last_low_light_warning = 0
        
        # Crowd Detection
        self.crowd_threshold = 3  # Number of people to consider a crowd
        self.crowd_mode_timer = 0  # Hysteresis timer for crowd mode (frames)

        # Mode Management
        self.mode = 'detection'  # 'detection', 'search', or 'ocr'
        
        # Search Mode
        self.search_targets = ['person', 'cell phone', 'bottle', 'chair', 'cup']
        self.target_index = 0  # Initialize target index
        self.search_cooldown = 2.0  # Faster cooldown for search mode
        self.last_search_announcement = 0
        
        # OCR Mode
        print("Initializing OCR reader...")
        self.ocr_reader = easyocr.Reader(['en'])  # Initialize once
        print("OCR reader ready.")

        # Visual setup
        self.cap.set(3, 640)  # Width
        self.cap.set(4, 480)  # Height
        
        # Check if OpenCV GUI is available
        self.gui_available = self._check_gui_support()
        
        print("System Ready. Press 'q' to exit.")
        print("Controls: 's' = Toggle Search Mode, 'n' = Next Target (in Search Mode), 'r' = OCR Mode, 'q' = Quit")
        if not self.gui_available:
            print("Warning: OpenCV GUI not available. Running in headless mode.")
            print("Keyboard input will work via console.")
    
    def _check_gui_support(self):
        """Check if OpenCV GUI functions are available."""
        try:
            # Try to create a test window
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow("_test_", test_img)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            return True
        except:
            return False

    def calculate_brightness(self, frame):
        """
        Calculate average brightness of the frame.
        
        Args:
            frame: BGR image frame
            
        Returns:
            float: Average brightness (0-255)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate mean brightness
        return np.mean(gray)

    def get_clock_position(self, center_x, frame_width):
        """Maps X-coordinate to Clock Position (10, 11, 12, 1, 2)"""
        segment = frame_width / 5
        if center_x < segment: 
            return "at 10 o'clock"
        elif center_x < segment * 2: 
            return "at 11 o'clock"
        elif center_x < segment * 3: 
            return "at 12 o'clock"
        elif center_x < segment * 4: 
            return "at 1 o'clock"
        else: 
            return "at 2 o'clock"
    
    def get_position_lrc(self, center_x, frame_width):
        """
        Get Left/Right/Center position for Search Mode.
        
        Args:
            center_x: X coordinate of object center
            frame_width: Width of the frame
            
        Returns:
            str: 'Left', 'Right', or 'Center'
        """
        third = frame_width / 3
        if center_x < third:
            return "Left"
        elif center_x < 2 * third:
            return "Center"
        else:
            return "Right"

    def estimate_distance(self, box_height, frame_height):
        """Estimates distance based on how tall the object is in the frame"""
        ratio = box_height / frame_height
        if ratio > 0.75: 
            return "Very Close"
        if ratio > 0.5: 
            return "Near"
        return ""  # If far, don't mention distance to keep speech short

    def process_frame(self, frame):
        """
        Process frame with all advanced features:
        - Low light detection
        - Mode-specific processing (detection/search/ocr)
        - Crowd grouping
        - Persistence tracking
        - Priority-based preemption
        """
        frame_h, frame_w = frame.shape[:2]
        
        # --- LOW LIGHT DETECTION ---
        brightness = self.calculate_brightness(frame)
        current_time = time.time()
        
        if brightness < self.low_light_threshold:
            # Check cooldown for low light warning
            if current_time - self.last_low_light_warning > self.low_light_cooldown:
                print("Low light detected - stopping detection")
                self.tts.speak("Environment too dark", preempt=True)
                self.last_low_light_warning = current_time
            # Don't process detections in low light
            cv2.putText(frame, f"Low Light: {brightness:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Mode: {self.mode.upper()}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return frame
        
        # Display current mode
        cv2.putText(frame, f"Mode: {self.mode.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # --- SEARCH MODE PROCESSING ---
        if self.mode == 'search':
            return self.process_search_mode(frame, current_time)
        
        # --- OCR MODE PROCESSING (handled in run loop, not here) ---
        if self.mode == 'ocr':
            cv2.putText(frame, "OCR Mode - Press 'r' to read text", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            return frame
        
        # --- DEFAULT DETECTION MODE ---
        return self.process_detection_mode(frame, current_time)
    
    def process_search_mode(self, frame, current_time):
        """Process frame in Search Mode - look for specific target only."""
        frame_h, frame_w = frame.shape[:2]
        target = self.search_targets[self.target_index]
        
        # Display search target
        cv2.putText(frame, f"Searching: {target}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Run detection
        results = self.model(frame, verbose=False, stream=True)
        
        target_detections = []
        
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue
                
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                
                # Only process the target object
                if label != target:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) / 2
                
                # Get Left/Right/Center position
                position = self.get_position_lrc(center_x, frame_w)
                
                target_detections.append({
                    'label': label,
                    'conf': conf,
                    'position': position,
                    'box': (x1, y1, x2, y2),
                    'center_x': center_x
                })
        
        # Process search results
        if target_detections:
            # Use the detection with highest confidence
            best_detection = max(target_detections, key=lambda x: x['conf'])
            position = best_detection['position']
            
            # Check cooldown
            if current_time - self.last_search_announcement > self.search_cooldown:
                if position == "Center":
                    text = f"Found {target} Center"
                elif position == "Left":
                    text = "Turn Left"
                else:  # Right
                    text = "Turn Right"
                
                print(f"Search Mode: {text}")
                self.tts.speak(text, preempt=True)
                self.last_search_announcement = current_time
            
            # Draw detection
            x1, y1, x2, y2 = best_detection['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"{target} {position}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # No target found - draw indicator
            cv2.putText(frame, f"No {target} detected", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame
    
    def process_detection_mode(self, frame, current_time):
        """Process frame in default Detection Mode - all existing logic."""
        frame_h, frame_w = frame.shape[:2]
        
        # --- OBJECT DETECTION ---
        results = self.model(frame, verbose=False, stream=True)
        
        detections = []
        person_count = 0
        current_labels = set()  # Track what's detected this frame
        
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.conf_threshold: 
                    continue
                
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Count persons for crowd detection
                if label == 'person':
                    person_count += 1
                
                # Calculate spatial info
                center_x = (x1 + x2) / 2
                box_h = y2 - y1
                
                detections.append({
                    'label': label,
                    'conf': conf,
                    'priority': self.priority_map.get(label, 5),
                    'box': (x1, y1, x2, y2),
                    'clock': self.get_clock_position(center_x, frame_w),
                    'dist': self.estimate_distance(box_h, frame_h)
                })
                current_labels.add(label)
        
        # --- CROWD GROUPING WITH HYSTERESIS TIMER ---
        # If people_count >= 4, activate crowd mode timer (30 frames = ~1-2 seconds)
        if person_count >= 4:
            self.crowd_mode_timer = 30
            print(f"Crowd detected: {person_count} people - Timer set to 30 frames")
        
        # If timer is active, force crowd mode (remove all individual person detections)
        if self.crowd_mode_timer > 0:
            # Remove all individual person detections to prevent flickering
            detections = [d for d in detections if d['label'] != 'person']
            # Add crowd as high priority
            detections.append({
                'label': 'Crowd ahead',
                'conf': 0.9,  # High confidence for crowd
                'priority': 1,  # High priority
                'box': (0, 0, frame_w, frame_h),  # Full frame
                'clock': 'at 12 o\'clock',
                'dist': ''
            })
            # Decrement timer
            self.crowd_mode_timer -= 1
            print(f"Crowd mode active (Timer: {self.crowd_mode_timer}) - Individual persons removed")
        elif person_count > self.crowd_threshold:
            # Normal crowd detection (when timer is not active)
            detections = [d for d in detections if d['label'] != 'person']
            detections.append({
                'label': 'Crowd ahead',
                'conf': 0.9,
                'priority': 1,
                'box': (0, 0, frame_w, frame_h),
                'clock': 'at 12 o\'clock',
                'dist': ''
            })
            print(f"Crowd detected: {person_count} people")
        
        # --- PERSISTENCE TRACKING (Anti-Jitter) ---
        # Update detection counters
        for label in current_labels:
            self.detection_counters[label] += 1
        # Reset counters for objects not detected this frame
        for label in list(self.detection_counters.keys()):
            if label not in current_labels:
                self.detection_counters[label] = 0
        
        # Filter detections by persistence (must be detected 3 consecutive frames)
        persistent_detections = []
        for det in detections:
            label = det['label']
            if self.detection_counters[label] >= self.persistence_threshold:
                persistent_detections.append(det)
        
        if not persistent_detections:
            return frame
        
        # --- PRIORITY SORTING ---
        persistent_detections.sort(key=lambda x: (x['priority'], -x['conf']))
        
        # --- ANNOUNCEMENT WITH PREEMPTION ---
        top_obj = persistent_detections[0]
        is_high_priority = top_obj['label'] in self.high_priority or 'Crowd' in top_obj['label']
        
        self.announce(top_obj, preempt=is_high_priority)
        
        # --- DRAW DETECTIONS ---
        for d in persistent_detections:
            x1, y1, x2, y2 = d['box']
            # High priority = green, others = blue
            color = (0, 255, 0) if d['priority'] <= 1 else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Show label and persistence count
            label_text = f"{d['label']} ({self.detection_counters[d['label']]})"
            cv2.putText(frame, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display brightness
        brightness = self.calculate_brightness(frame)
        cv2.putText(frame, f"Brightness: {brightness:.1f}", (10, frame_h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def process_ocr(self, frame):
        """
        Process OCR on current frame with ROI cropping and preprocessing.
        Optimized for reading newspaper text without DroidCam overlay.
        
        Args:
            frame: Current video frame
            
        Returns:
            str: Detected text or None
        """
        try:
            # --- STEP 1: ROI CROP (Center 60% - remove 20% from each side) ---
            frame_h, frame_w = frame.shape[:2]
            
            # Calculate crop boundaries (center 60%)
            top_margin = int(frame_h * 0.2)      # Remove top 20%
            bottom_margin = int(frame_h * 0.8)   # Keep until 80% (bottom 20% removed)
            left_margin = int(frame_w * 0.2)     # Remove left 20%
            right_margin = int(frame_w * 0.8)    # Keep until 80% (right 20% removed)
            
            # Crop to center region
            roi = frame[top_margin:bottom_margin, left_margin:right_margin]
            
            # --- STEP 2: IMAGE PREPROCESSING ---
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply Binary Threshold using Otsu's method
            # This makes text black and background white, improving OCR accuracy
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # --- STEP 3: RUN OCR ON PROCESSED IMAGE ---
            results = self.ocr_reader.readtext(binary)
            
            if not results:
                return None
            
            # --- STEP 4: GARBAGE FILTER ---
            filtered_texts = []
            for result in results:
                text = result[1].strip()
                
                # Filter out lines containing "DroidCam" (case-insensitive)
                if 'droidcam' in text.lower():
                    continue
                
                # Filter out single-character results (noise)
                if len(text) <= 1:
                    continue
                
                filtered_texts.append(text)
            
            if not filtered_texts:
                return None
            
            # Join filtered results into a single string
            combined_text = " ".join(filtered_texts)
            return combined_text
            
        except Exception as e:
            print(f"OCR Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def announce(self, obj, preempt=False):
        """
        Announce detected object with optional preemption.
        
        Args:
            obj: Detection dictionary
            preempt: If True, clear queue and speak immediately
        """
        current_time = time.time()
        label = obj['label']
        
        # Check cooldown (unless preempting)
        if not preempt:
            if current_time - self.last_spoken_time[label] < self.cooldown_seconds:
                return
        
        # Construct natural sentence
        parts = []
        if obj['dist']:
            parts.append(obj['dist'])
        parts.append(label)
        parts.append(obj['clock'])
        
        text = " ".join(parts).strip()
        print(f"Speaking: {text} {'[PREEMPT]' if preempt else ''}")
        
        self.tts.speak(text, preempt=preempt)
        self.last_spoken_time[label] = current_time

    def _get_keyboard_input(self):
        """
        Get keyboard input - works with or without OpenCV GUI.
        Returns character code or -1 if no key pressed.
        """
        if self.gui_available:
            try:
                return cv2.waitKey(1) & 0xFF
            except:
                # Fallback if waitKey fails
                self.gui_available = False
                return -1
        else:
            # Use msvcrt for Windows console input (non-blocking)
            if msvcrt.kbhit():
                key = msvcrt.getch()
                return ord(key) if isinstance(key, bytes) else ord(key)
            return -1
    
    def _display_frame(self, frame, window_name="Smart Vision"):
        """Display frame if GUI is available."""
        if self.gui_available:
            try:
                cv2.imshow(window_name, frame)
            except:
                self.gui_available = False
    
    def _cleanup_windows(self):
        """Clean up OpenCV windows if GUI is available."""
        if self.gui_available:
            try:
                cv2.destroyAllWindows()
            except:
                pass
    
    def run(self):
        frame_count = 0
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret: 
                    break
                
                # Handle keyboard input
                key = self._get_keyboard_input()
                
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    # Master toggle for Search Mode
                    if self.mode == 'detection':
                        # Switch to search mode
                        self.mode = 'search'
                        target = self.search_targets[self.target_index]
                        announcement = f"Search Mode Active. Looking for {target}"
                        print(announcement)
                        self.tts.speak(announcement, preempt=True)
                    elif self.mode == 'search':
                        # Switch back to detection mode
                        self.mode = 'detection'
                        announcement = "Back to Normal Mode"
                        print(announcement)
                        self.tts.speak(announcement, preempt=True)
                elif key == ord('n') or key == ord('N'):
                    # Next target - only works in search mode
                    if self.mode == 'search':
                        # Cycle to next target
                        self.target_index = (self.target_index + 1) % len(self.search_targets)
                        target = self.search_targets[self.target_index]
                        announcement = f"Looking for {target}"
                        print(announcement)
                        self.tts.speak(announcement, preempt=True)
                    # Do nothing if not in search mode
                elif key == ord('r') or key == ord('R'):
                    # Trigger OCR Mode
                    if self.mode != 'ocr':
                        self.mode = 'ocr'
                        print("OCR Mode activated")
                        self.tts.speak("Reading text", preempt=True)
                        
                        # Process OCR on current frame
                        text = self.process_ocr(frame)
                        if text:
                            print(f"OCR Result: {text}")
                            self.tts.speak(text, preempt=False)
                        else:
                            print("No text found")
                            self.tts.speak("No text found", preempt=False)
                        
                        # Return to detection mode after OCR
                        self.mode = 'detection'
                
                # Performance Optimization: Process every Nth frame
                frame_count += 1
                if frame_count % self.frame_skip == 0:
                    frame = self.process_frame(frame)
                    self._display_frame(frame)
                else:
                    # Just show the video without processing to keep FPS high
                    self._display_frame(frame)
        finally:
            self.cap.release()
            self._cleanup_windows()

if __name__ == "__main__":
    # DroidCam URL - Using IP from your DroidCam app
    # Available IPs:
    # - WiFi IP: 192.168.1.8
    # - Device IPs: 100.123.251.230 or 10.171.3.216
    # Using WiFi IP (192.168.1.8) - change to other IPs if needed
    droidcam_url = "http://192.168.1.8:4747/video"
    # Alternatives:
    # droidcam_url = "http://10.171.3.216:4747/video"
    # droidcam_url = "http://100.123.251.230:4747/video"
    
    # Choose video source:
    # - Use 0 for local webcam
    # - Use droidcam_url for DroidCam wireless camera
    video_source = droidcam_url  # Changed to use DroidCam - set to 0 for webcam
    
    app = SmartVisionApp(source=video_source)
    app.run()
