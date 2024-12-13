from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                           QPushButton, QWidget, QHBoxLayout, QComboBox, 
                           QStatusBar, QScrollArea, QGridLayout, QSlider,
                           QFrame, QSpinBox, QSizePolicy, QTabWidget, QProgressBar)
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QPainter
from PyQt5.QtCore import QTimer, Qt, QSize, pyqtSignal, QThread
from scipy.io import wavfile
import sys
import cv2
import numpy as np
import mediapipe as mp
import pyaudio
from datetime import datetime
import os
import time
import speech_recognition as sr
import threading
import wave
import pyaudio


class BackgroundPreviewWidget(QWidget):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setFixedSize(160, 90)  # 16:9 aspect ratio
        self.image_path = image_path
        self.selected = False
        self.scroll_area = parent  # Store reference to scroll area
        
        try:
            # Handle both file paths and numpy arrays
            if isinstance(image_path, str):
                self.image = cv2.imread(image_path)
                if self.image is not None:
                    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            else:
                # Handle numpy array input
                self.image = image_path.copy()
                if self.image.shape[2] == 3:  # BGR to RGB
                    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            
            # Resize image if it exists
            if self.image is not None:
                self.image = cv2.resize(self.image, (160, 90))
        except Exception as e:
            print(f"Preview widget initialization error: {e}")
            self.image = None
    
    def paintEvent(self, event):
        if self.image is None:
            return
            
        painter = QPainter(self)
        height, width, channel = self.image.shape
        bytes_per_line = 3 * width
        qt_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        painter.drawImage(self.rect(), qt_image)
        
        # Draw border if selected
        if self.selected:
            painter.setPen(Qt.blue)
            painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
    
    def mousePressEvent(self, event):
        if self.scroll_area and hasattr(self.scroll_area, 'select_background'):
            self.scroll_area.select_background(self)

class BackgroundPreviewScrollArea(QScrollArea):
    backgroundSelected = pyqtSignal(int)  # Add signal
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMinimumHeight(110)
        self.setMaximumHeight(110)
        
        # Container widget
        container = QWidget()
        self.layout = QHBoxLayout(container)
        self.layout.setSpacing(10)
        self.layout.setAlignment(Qt.AlignLeft)
        self.setWidget(container)
        
        self.previews = []
        self.selected_preview = None
    
    def add_preview(self, image_path):
        try:
            preview = BackgroundPreviewWidget(image_path, self)  # Pass self as parent
            self.layout.addWidget(preview)
            self.previews.append(preview)
            return preview
        except Exception as e:
            print(f"Error adding preview: {e}")
            return None
    
    def select_background(self, preview):
        try:
            if self.selected_preview:
                self.selected_preview.selected = False
                self.selected_preview.update()
            
            preview.selected = True
            preview.update()
            self.selected_preview = preview
            
            # Notify main window
            if preview in self.previews and self.main_window:
                index = self.previews.index(preview)
                if hasattr(self.main_window, 'background_selected'):
                    self.main_window.background_selected(index)
                else:
                    print("Main window does not have background_selected method")
        except Exception as e:
            print(f"Error selecting background: {e}")
            
class EffectsWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        layout = QVBoxLayout(self)
        
        # Blur control
        blur_layout = QHBoxLayout()
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(1, 99)
        self.blur_slider.setValue(21)
        self.blur_slider.setTickPosition(QSlider.TicksBelow)
        self.blur_slider.setTickInterval(10)
        blur_layout.addWidget(QLabel("Blur:"))
        blur_layout.addWidget(self.blur_slider)
        
        # Segmentation threshold control
        threshold_layout = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 99)
        self.threshold_slider.setValue(60)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        threshold_layout.addWidget(QLabel("Threshold:"))
        threshold_layout.addWidget(self.threshold_slider)
        
        # AR size control
        ar_size_layout = QHBoxLayout()
        self.ar_size_spin = QSpinBox()
        self.ar_size_spin.setRange(50, 200)
        self.ar_size_spin.setValue(100)
        self.ar_size_spin.setSuffix("%")
        ar_size_layout.addWidget(QLabel("AR Size:"))
        ar_size_layout.addWidget(self.ar_size_spin)
        
        # Add all controls to main layout
        layout.addLayout(blur_layout)
        layout.addLayout(threshold_layout)
        layout.addLayout(ar_size_layout)


class VirtualBackgroundApp(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            self.init_variables()
            self.init_processing_tools()
            self.init_camera()
            self.load_resources()
            self.initUI()
            self.setup_audio_visualization()  
            self.setup_voice_commands()    
            self.setup_video_processing()
        except Exception as e:
            print(f"Initialization error: {e}")
            raise
    
    def initUI(self):
        """Initialize the user interface"""
        try:
            self.setWindowTitle("Enhanced Virtual Background App")
            self.setGeometry(100, 100, 1280, 900)
            
            # Main widget and layout
            main_widget = QWidget()
            self.setCentralWidget(main_widget)
            layout = QVBoxLayout(main_widget)
            
            # Video display
            self.setup_video_display(layout)
            
            # Background preview
            self.setup_preview_area(layout)
            
            # Controls
            self.setup_controls(layout)
            
            # Status bar
            self.setup_status_bar()
            
            print("UI initialized successfully")
        except Exception as e:
            print(f"UI initialization error: {e}")
            raise


    def init_video_background(self):
        """Initialize video background"""
        try:
            if self.video_backgrounds:
                # Close existing video if any
                if hasattr(self, 'current_video') and self.current_video is not None:
                    self.current_video.release()
                    
                # Open new video
                video_path = self.video_backgrounds[self.current_video_index]
                self.current_video = cv2.VideoCapture(video_path)
                if not self.current_video.isOpened():
                    print(f"Failed to open video: {video_path}")
                else:
                    print(f"Successfully opened video: {video_path}")
        except Exception as e:
            print(f"Video initialization error: {e}")


    def setup_audio_visualization(self):
        """Setup audio visualization"""
        try:
            # Create audio level widget
            audio_container = QWidget()
            audio_layout = QVBoxLayout(audio_container)
            
            # Create label
            mic_label = QLabel("Mic")
            mic_label.setAlignment(Qt.AlignCenter)
            audio_layout.addWidget(mic_label)
            
            # Create level indicator
            self.audio_level = QProgressBar()
            self.audio_level.setOrientation(Qt.Vertical)
            self.audio_level.setRange(0, 100)
            self.audio_level.setValue(0)
            self.audio_level.setFixedWidth(20)
            self.audio_level.setFixedHeight(100)
            self.audio_level.setTextVisible(False)
            audio_layout.addWidget(self.audio_level)
            
            # Add spacer
            audio_layout.addStretch()
            
            # Add to controls layout
            self.right_controls.addWidget(audio_container)
            
            # Start audio monitoring
            self.start_audio_monitoring()
            print("Audio visualization setup complete")
        except Exception as e:
            print(f"Audio visualization setup error: {e}")

    def start_audio_monitoring(self):
        """Start monitoring audio levels"""
        try:
            self.audio_thread = threading.Thread(target=self.monitor_audio, daemon=True)
            self.audio_thread.start()
            print("Audio monitoring started")
        except Exception as e:
            print(f"Audio monitoring error: {e}")

    def monitor_audio(self):
        """Monitor audio levels"""
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=44100,
                        input=True,
                        frames_per_buffer=1024,
                        stream_callback=self.audio_callback)
            
            stream.start_stream()
            while stream.is_active():
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Audio monitoring error: {e}")

    def audio_callback(self, in_data, frame_count, time_info, status):
        """Handle audio data for visualization"""
        try:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            level = int(np.abs(audio_data).mean() * 100)
            self.audio_level.setValue(level)
            return (in_data, pyaudio.paContinue)
        except Exception as e:
            print(f"Audio callback error: {e}")
            return (in_data, pyaudio.paContinue)

    def setup_voice_commands(self):
        """Setup voice command recognition"""
        try:
            self.recognizer = sr.Recognizer()
            # Set sensitivity
            self.recognizer.energy_threshold = 4000
            self.recognizer.dynamic_energy_threshold = True
            
            # Start voice command thread
            self.voice_command_thread = threading.Thread(target=self.listen_for_commands, daemon=True)
            self.voice_command_thread.start()
            print("Voice commands initialized")
        except Exception as e:
            print(f"Voice command setup error: {e}")

    def listen_for_commands(self):
        """Listen for voice commands"""
        while True:
            try:
                with sr.Microphone() as source:
                    print("Listening for commands...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=2)
                    
                    try:
                        text = self.recognizer.recognize_google(audio).lower()
                        print(f"Heard: {text}")
                        
                        if "next" in text:
                            if self.bg_combo.currentText() == "Video Backgrounds":
                                self.next_video_background()
                            else:
                                self.next_background()
                            print("Switching to next background")
                        elif "previous" in text or "back" in text:
                            if self.bg_combo.currentText() == "Video Backgrounds":
                                self.previous_video_background()
                            else:
                                self.previous_background()
                            print("Switching to previous background")
                            
                    except sr.UnknownValueError:
                        print("Could not understand audio")
                    except sr.RequestError as e:
                        print(f"Could not request results; {e}")
                        
            except Exception as e:
                print(f"Voice recognition error: {e}")
            time.sleep(0.1)

    def previous_background(self):
        """Switch to previous background"""
        try:
            if self.backgrounds:
                self.current_bg_index = (self.current_bg_index - 1) % len(self.backgrounds)
        except Exception as e:
            print(f"Previous background error: {e}")

    def previous_video_background(self):
        """Switch to previous video background"""
        try:
            if self.video_backgrounds:
                self.current_video_index = (self.current_video_index - 1) % len(self.video_backgrounds)
                self.init_video_background()
        except Exception as e:
            print(f"Previous video error: {e}")

    def change_background_mode(self, index):
        """Handle background mode changes"""
        try:
            mode = self.bg_combo.currentText()
            
            if mode == "Video Backgrounds":
                self.current_video_index = 0
                self.init_video_background()
                print("Switched to video background mode")
            else:
                # Clean up video if switching away from video mode
                if hasattr(self, 'current_video') and self.current_video is not None:
                    self.current_video.release()
                    self.current_video = None
                    
            self.statusBar.showMessage(f"Switched to {mode}")
        except Exception as e:
            print(f"Mode change error: {e}")


    def setup_video_display(self, layout):
        """Set up the main video display area"""
        try:
            self.video_label = QLabel()
            self.video_label.setAlignment(Qt.AlignCenter)
            self.video_label.setMinimumSize(1280, 720)
            layout.addWidget(self.video_label)
        except Exception as e:
            print(f"Video display setup error: {e}")
            raise

    def setup_preview_area(self, layout):
        """Set up the background preview area with separate sections"""
        try:
            # Create tab widget for different background types
            self.preview_tabs = QTabWidget()
            layout.addWidget(self.preview_tabs)
            
            # Static backgrounds tab
            self.static_preview_area = BackgroundPreviewScrollArea(self)
            self.preview_tabs.addTab(self.static_preview_area, "Static Backgrounds")
            
            # Video backgrounds tab
            self.video_preview_area = BackgroundPreviewScrollArea(self)
            self.preview_tabs.addTab(self.video_preview_area, "Video Backgrounds")
            
            # Add static backgrounds to preview
            for bg in self.backgrounds:
                if bg is not None and isinstance(bg, np.ndarray):
                    try:
                        self.static_preview_area.add_preview(bg)
                    except Exception as e:
                        print(f"Error adding background preview: {e}")
            
            # Add video previews
            for video_path in self.video_backgrounds:
                try:
                    # Get first frame of video for preview
                    cap = cv2.VideoCapture(video_path)
                    ret, frame = cap.read()
                    if ret:
                        preview = self.video_preview_area.add_preview(frame)
                        preview.video_path = video_path  # Store video path
                    cap.release()
                except Exception as e:
                    print(f"Error adding video preview: {e}")
                    
        except Exception as e:
            print(f"Preview area setup error: {e}")
            raise
    

    def setup_voice_commands(self):
        """Initialize voice command recognition"""
        try:
            self.recognizer = sr.Recognizer()
            self.voice_thread = threading.Thread(target=self.listen_for_commands, daemon=True)
            self.voice_thread.start()
            print("Voice commands initialized")
        except Exception as e:
            print(f"Voice command setup error: {e}")

    def listen_for_commands(self):
        """Listen for voice commands"""
        while True:
            try:
                with sr.Microphone() as source:
                    print("Listening...")
                    self.recognizer.adjust_for_ambient_noise(source)
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=1)
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"Heard: {text}")
                    
                    if "next" in text:
                        self.next_background()
                        print("Switching to next background")
                    
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except Exception as e:
                print(f"Voice recognition error: {e}")
            time.sleep(0.1)

    def get_video_background(self, h, w):
        """Get current video background frame"""
        try:
            if not self.video_backgrounds:
                return np.ones((h, w, 3), dtype=np.uint8) * [0, 120, 255]

            # Check if video needs to be initialized
            if not hasattr(self, 'current_video') or self.current_video is None:
                self.init_video_background()
            
            if self.current_video is not None and self.current_video.isOpened():
                ret, frame = self.current_video.read()
                if not ret:
                    self.current_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.current_video.read()
                
                if ret and frame is not None:
                    return cv2.resize(frame, (w, h))
            
            return np.ones((h, w, 3), dtype=np.uint8) * [0, 120, 255]
            
        except Exception as e:
            print(f"Video background error: {e}")
            return np.ones((h, w, 3), dtype=np.uint8) * [0, 120, 255]

    def change_background_mode(self, index):
        """Handle background mode changes"""
        try:
            # Clean up previous video
            if hasattr(self, 'current_video') and self.current_video is not None:
                self.current_video.release()
                self.current_video = None
            
            mode = self.bg_combo.currentText()
            
            if mode == "Static Backgrounds":
                self.preview_area.setVisible(True)
                self.effects_widget.blur_slider.setEnabled(False)
                # Show only static background previews
                for i, preview in enumerate(self.preview_area.previews):
                    preview.setVisible(i < len(self.backgrounds))
                    
            elif mode == "Video Backgrounds":
                self.preview_area.setVisible(True)
                self.effects_widget.blur_slider.setEnabled(False)
                # Show only video previews
                for i, preview in enumerate(self.preview_area.previews):
                    preview.setVisible(i >= len(self.backgrounds))
                if not hasattr(self, 'current_video_path'):
                    self.current_video_path = self.video_backgrounds[0] if self.video_backgrounds else None
                
            elif mode == "Blur Background":
                self.preview_area.setVisible(False)
                self.effects_widget.blur_slider.setEnabled(True)
                
            elif mode == "Portrait Mode":
                self.preview_area.setVisible(False)
                self.effects_widget.blur_slider.setEnabled(True)
            
            self.statusBar.showMessage(f"Switched to {mode}")
            
        except Exception as e:
            print(f"Mode change error: {e}")

    def background_selected(self, index):
        """Handle background selection"""
        try:
            mode = self.bg_combo.currentText()
            if mode == "Static Backgrounds":
                if 0 <= index < len(self.backgrounds):
                    self.current_bg_index = index
                    print(f"Selected static background {index}")
            elif mode == "Video Backgrounds":
                video_index = index - len(self.backgrounds)
                if 0 <= video_index < len(self.video_backgrounds):
                    self.current_video_index = video_index
                    self.current_video_path = self.video_backgrounds[video_index]
                    print(f"Selected video background {video_index}")
        except Exception as e:
            print(f"Background selection error: {e}")

    def setup_controls(self, layout):
        """Set up all control elements"""
        try:
            controls_container = QWidget()
            controls_layout = QHBoxLayout(controls_container)
            
            # Left controls
            self.setup_left_controls(controls_layout)
            
            # Effects widget
            self.setup_effects_widget(controls_layout)
            
            # Right controls
            self.setup_right_controls(controls_layout)
            
            layout.addWidget(controls_container)
        except Exception as e:
            print(f"Controls setup error: {e}")
            raise

    def setup_left_controls(self, parent_layout):
        """Set up left side controls"""
        try:
            left_controls = QVBoxLayout()
            
            # Background type selection
            self.bg_combo = QComboBox()
            self.bg_combo.addItems(["Static Backgrounds", "Video Backgrounds", "Blur Background", "Portrait Mode"])
            self.bg_combo.currentIndexChanged.connect(self.change_background_mode)
            left_controls.addWidget(self.bg_combo)
            
            # AR controls
            self.ar_toggle = QPushButton("Toggle AR Effects")
            self.ar_toggle.setCheckable(True)
            self.ar_toggle.clicked.connect(self.toggle_ar_effects)
            left_controls.addWidget(self.ar_toggle)
            
            parent_layout.addLayout(left_controls)
        except Exception as e:
            print(f"Left controls setup error: {e}")
            raise

    def setup_effects_widget(self, parent_layout):
        """Set up effects controls"""
        try:
            self.effects_widget = EffectsWidget()
            self.effects_widget.blur_slider.valueChanged.connect(self.update_blur)
            self.effects_widget.threshold_slider.valueChanged.connect(self.update_threshold)
            self.effects_widget.ar_size_spin.valueChanged.connect(self.update_ar_size)
            parent_layout.addWidget(self.effects_widget)
        except Exception as e:
            print(f"Effects widget setup error: {e}")
            raise

    def setup_right_controls(self, parent_layout):
        """Set up right side controls"""
        try:
            right_controls = QVBoxLayout()
            
            # Recording controls
            self.record_btn = QPushButton("Start Recording")
            self.record_btn.clicked.connect(self.toggle_recording)
            right_controls.addWidget(self.record_btn)
            
            # Screenshot button
            self.screenshot_btn = QPushButton("Take Screenshot")
            self.screenshot_btn.clicked.connect(self.take_screenshot)
            right_controls.addWidget(self.screenshot_btn)
            
            parent_layout.addLayout(right_controls)
        except Exception as e:
            print(f"Right controls setup error: {e}")
            raise

    def setup_status_bar(self):
        """Set up the status bar"""
        try:
            self.statusBar = QStatusBar()
            self.setStatusBar(self.statusBar)
            self.statusBar.showMessage("Ready")
        except Exception as e:
            print(f"Status bar setup error: {e}")
            raise

    def setup_video_processing(self):
        """Set up video processing timer"""
        try:
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(33)  # ~30 FPS
            print("Video processing setup complete")
        except Exception as e:
            print(f"Video processing setup error: {e}")
            raise



    def toggle_ar_effects(self):
        """Toggle AR effects on/off"""
        try:
            self.show_ar = not self.show_ar
            self.ar_toggle.setChecked(self.show_ar)
            self.statusBar.showMessage(f"AR Effects: {'On' if self.show_ar else 'Off'}")
        except Exception as e:
            print(f"AR toggle error: {e}")

    def update_blur(self, value):
        """Update blur amount"""
        try:
            self.blur_amount = value
        except Exception as e:
            print(f"Blur update error: {e}")

    def update_threshold(self, value):
        """Update segmentation threshold"""
        try:
            self.segmentation_threshold = value / 100.0
        except Exception as e:
            print(f"Threshold update error: {e}")

    def update_ar_size(self, value):
        """Update AR prop size"""
        try:
            self.ar_scale_factor = value / 100.0
        except Exception as e:
            print(f"AR size update error: {e}")

    def toggle_recording(self):
        """Toggle recording state"""
        try:
            if not self.is_recording:
                self.start_recording()
            else:
                self.stop_recording()
        except Exception as e:
            print(f"Recording toggle error: {e}")

    def take_screenshot(self):
        """Take and save screenshot"""
        try:
            if hasattr(self, 'video_label') and self.video_label.pixmap():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
                filepath = os.path.join('screenshots', filename)
                
                if self.video_label.pixmap().save(filepath):
                    self.statusBar.showMessage(f"Screenshot saved: {filename}")
                else:
                    self.statusBar.showMessage("Failed to save screenshot")
        except Exception as e:
            print(f"Screenshot error: {e}")
            self.statusBar.showMessage("Screenshot error occurred")

    
    def setup_audio_visualization(self):
        """Setup audio level visualization"""
        try:
            # Create audio level indicator
            self.audio_level = QProgressBar()
            self.audio_level.setOrientation(Qt.Vertical)
            self.audio_level.setRange(0, 100)
            self.audio_level.setValue(0)
            self.audio_level.setTextVisible(False)
            self.audio_level.setFixedWidth(20)
            
            # Add to UI
            audio_container = QWidget()
            audio_layout = QVBoxLayout(audio_container)
            audio_layout.addWidget(QLabel("Mic"))
            audio_layout.addWidget(self.audio_level)
            self.controls_layout.addWidget(audio_container)
            
            # Setup audio monitor
            self.audio_thread = threading.Thread(target=self.monitor_audio_level, daemon=True)
            self.audio_thread.start()
            
        except Exception as e:
            print(f"Audio visualization setup error: {e}")

    def monitor_audio_level(self):
        """Monitor audio level for visualization"""
        try:
            with sr.Microphone() as source:
                while True:
                    if hasattr(self, 'audio_level'):
                        # Get audio level
                        audio = self.recognizer.listen(source, timeout=0.1, phrase_time_limit=0.1)
                        level = np.abs(np.frombuffer(audio.get_raw_data(), np.int16)).mean()
                        normalized_level = min(100, int(level / 100))
                        self.audio_level.setValue(normalized_level)
                    time.sleep(0.1)
        except Exception as e:
            print(f"Audio monitoring error: {e}")

    def start_recording(self):
        """Start recording with audio"""
        try:
            if not self.is_recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join('recordings', f"recording_{timestamp}.mp4")
                
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Try 'avc1' if this doesn't work
                self.recording_data = cv2.VideoWriter(filename, fourcc, 30.0, (1280, 720))
                
                # Initialize audio recording
                self.audio_frames = []
                self.audio_recording = True
                
                # Start audio recording thread
                self.audio_thread = threading.Thread(target=self.record_audio)
                self.audio_thread.start()
                
                self.is_recording = True
                self.record_btn.setText("Stop Recording")
                self.statusBar.showMessage("Recording started...")
        except Exception as e:
            print(f"Recording start error: {e}")

    def record_audio(self):
        """Record audio"""
        try:
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 2
            RATE = 44100
            
            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
            
            while self.audio_recording:
                data = stream.read(CHUNK)
                self.audio_frames.append(data)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            print(f"Audio recording error: {e}")

    def stop_recording(self):
        """Stop recording and save"""
        try:
            if self.is_recording:
                self.is_recording = False
                self.audio_recording = False
                
                if hasattr(self, 'audio_thread'):
                    self.audio_thread.join()
                
                if self.recording_data:
                    self.recording_data.release()
                
                # Save audio
                if hasattr(self, 'audio_frames') and self.audio_frames:
                    audio_filename = os.path.join('recordings', 'temp_audio.wav')
                    wf = wave.open(audio_filename, 'wb')
                    wf.setnchannels(2)
                    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
                    wf.setframerate(44100)
                    wf.writeframes(b''.join(self.audio_frames))
                    wf.close()
                    
                    # Combine audio and video
                    self.combine_audio_video()
                
                self.record_btn.setText("Start Recording")
                self.statusBar.showMessage("Recording saved")
        except Exception as e:
            print(f"Recording stop error: {e}")

    def combine_audio_video(self):
        """Combine audio and video using ffmpeg"""
        try:
            import ffmpeg
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_video = ffmpeg.input(os.path.join('recordings', 'temp_video.mp4'))
            input_audio = ffmpeg.input(os.path.join('recordings', 'temp_audio.wav'))
            
            output_filename = os.path.join('recordings', f"final_recording_{timestamp}.mp4")
            ffmpeg.output(input_video, input_audio, output_filename).run(overwrite_output=True)
            
            # Clean up temporary files
            os.remove(os.path.join('recordings', 'temp_video.mp4'))
            os.remove(os.path.join('recordings', 'temp_audio.wav'))
        except Exception as e:
            print(f"Combine audio/video error: {e}")


    def setup_speech_recognition(self):
        """Setup speech recognition"""
        try:
            self.recognizer = sr.Recognizer()
            self.speech_thread = threading.Thread(target=self.listen_for_commands, daemon=True)
            self.speech_thread.start()
            print("Speech recognition initialized")
        except Exception as e:
            print(f"Speech recognition setup error: {e}")

    def listen_for_commands(self):
        """Listen for voice commands"""
        while True:
            try:
                with sr.Microphone() as source:
                    print("Listening for commands...")
                    self.recognizer.adjust_for_ambient_noise(source)
                    audio = self.recognizer.listen(source)
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"Heard: {text}")
                    
                    if "next" in text:
                        print("Executing next command...")
                        mode = self.bg_combo.currentText()
                        if mode == "Static Backgrounds":
                            self.next_background()
                        elif mode == "Video Backgrounds":
                            self.next_video_background()
                    
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
            except sr.UnknownValueError:
                print("Could not understand audio")
            except Exception as e:
                print(f"Speech recognition error: {e}")
            time.sleep(0.1)



    def record_frame(self, frame):
        """Record a single frame"""
        try:
            if self.recording_data:
                self.recording_data.write(frame)
        except Exception as e:
            print(f"Frame recording error: {e}")
            self.stop_recording()

    def update_frame(self):
        """Process and update each video frame"""
        if not self.cap.isOpened():
            return

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Failed to read frame")
                return

            # Process frame
            frame = cv2.flip(frame, 1)  # Mirror effect
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process face mesh for AR effects
            face_results = self.face_mesh.process(frame_rgb)
            
            # Process segmentation
            segmentation_result = self.selfie_segmentation.process(frame_rgb)
            
            if segmentation_result.segmentation_mask is not None:
                # Create mask
                mask = segmentation_result.segmentation_mask
                mask = np.stack((mask,) * 3, axis=-1)
                
                # Get current background
                background = self.get_current_background(frame)
                if background is None:
                    background = np.ones_like(frame) * [0, 120, 255]
                
                # Create output image
                output_image = np.where(mask > self.segmentation_threshold, frame, background)
                
                # Add AR effects if enabled
                if self.show_ar:
                    output_image = self.add_ar_effects(output_image, face_results)
                
                # Convert to Qt format and display
                qt_image = self.convert_to_qt_format(output_image)
                if qt_image is not None:
                    self.video_label.setPixmap(qt_image)
                
                # Handle recording
                if self.is_recording and self.recording_data is not None:
                    self.record_frame(output_image)
                    
        except Exception as e:
            print(f"Frame processing error: {e}")

    def get_current_background(self, frame):
        """Get the current background based on selected mode"""
        try:
            mode = self.bg_combo.currentText()
            h, w = frame.shape[:2]
            
            if mode == "Blur Background":
                blur_size = self.blur_amount * 2 + 1
                return cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
                
            elif mode == "Video Backgrounds":
                h, w = frame.shape[:2]
                return self.get_video_background(h, w)
                
            elif mode == "Portrait Mode":
                return frame
                
            else:  # Static Backgrounds
                if self.backgrounds and self.current_bg_index < len(self.backgrounds):
                    bg = self.backgrounds[self.current_bg_index]
                    if bg is not None:
                        return cv2.resize(bg, (w, h))
            
            # Return default background if current background is invalid
            return np.ones((h, w, 3), dtype=np.uint8) * [0, 120, 255]
            
        except Exception as e:
            print(f"Background error: {e}")
            return None

    def overlay_image_alpha(self, background, overlay, x, y):
        """Overlay an RGBA image onto a background"""
        try:
            if overlay.shape[2] != 4:  # Must have an alpha channel
                return background
                
            if x >= background.shape[1] or y >= background.shape[0]:
                return background

            h, w = overlay.shape[:2]
            
            # Crop if necessary
            if y + h > background.shape[0]:
                h = background.shape[0] - y
            if x + w > background.shape[1]:
                w = background.shape[1] - x
                
            # Get the overlaying region
            overlay_image = overlay[:h, :w]
            alpha_channel = overlay_image[:, :, 3] / 255.0
            alpha_3channel = np.stack([alpha_channel] * 3, axis=-1)
            
            # Get the background region
            background_section = background[y:y+h, x:x+w]
            
            # Blend the images
            try:
                background[y:y+h, x:x+w] = background_section * (1 - alpha_3channel) + \
                                        overlay_image[:, :, :3] * alpha_3channel
            except ValueError as e:
                print(f"Blending error: {e}")
                
            return background
            
        except Exception as e:
            print(f"Overlay error: {e}")
            return background


    def add_ar_effects(self, image, face_results):
        """Add AR effects based on face detection"""
        if not self.ar_props or self.current_ar_effect >= len(self.ar_props):
            return image
            
        try:
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # Get image dimensions
                    h, w = image.shape[:2]
                    
                    # Get key facial landmarks
                    nose_tip = face_landmarks.landmark[4]
                    forehead = face_landmarks.landmark[10]
                    left_temple = face_landmarks.landmark[234]
                    right_temple = face_landmarks.landmark[454]
                    
                    # Calculate face width and prop scale
                    face_width = abs(right_temple.x - left_temple.x) * w
                    prop_scale = face_width * 1.8 * (self.ar_scale_factor / 100)
                    
                    # Calculate position (centered above head)
                    x = int(nose_tip.x * w - prop_scale / 2)
                    y = int(forehead.y * h - prop_scale)
                    
                    # Ensure coordinates are valid
                    x = max(0, min(x, w))
                    y = max(0, y)
                    
                    # Get and resize prop
                    prop = self.ar_props[self.current_ar_effect]
                    if prop is not None and prop.shape[2] == 4:  # Ensure RGBA
                        prop_h, prop_w = prop.shape[:2]
                        scale_factor = prop_scale / prop_w
                        new_width = int(prop_w * scale_factor)
                        new_height = int(prop_h * scale_factor)
                        
                        try:
                            prop_resized = cv2.resize(prop, (new_width, new_height))
                            image = self.overlay_image_alpha(image, prop_resized, x, y)
                        except Exception as e:
                            print(f"Prop resize error: {e}")
            
            return image
            
        except Exception as e:
            print(f"AR effect error: {e}")
            return image

    def convert_to_qt_format(self, image):
        """Convert OpenCV image to Qt format"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            return QPixmap.fromImage(qt_image)
        except Exception as e:
            print(f"Format conversion error: {e}")
            return None

    def init_variables(self):
        """Initialize all variables with default values"""
        try:
            # Core variables
            self.current_bg_index = 0
            self.current_video_index = 0
            self.current_ar_effect = 0
            self.is_recording = False
            self.show_ar = True
            self.blur_amount = 21
            self.segmentation_threshold = 0.6
            self.recording_data = None
            self.ar_scale_factor = 1.0
            
            # Initialize lists
            self.backgrounds = []
            self.video_backgrounds = []
            self.ar_props = []
            
            # Create default background
            self.create_default_backgrounds()
            
            print("Variables initialized successfully")
        except Exception as e:
            print(f"Variable initialization error: {e}")
            raise

    def create_default_backgrounds(self):
        """Create default solid color and gradient backgrounds"""
        try:
            # Solid colors (BGR format)
            colors = [
                ([255, 140, 0], "Orange"),    # Orange
                ([0, 255, 0], "Green"),       # Green
                ([255, 0, 0], "Blue"),        # Blue
                ([0, 0, 255], "Red"),         # Red
                ([255, 255, 0], "Cyan"),      # Cyan
                ([255, 0, 255], "Yellow"),    # Yellow
                ([128, 0, 128], "Purple"),    # Purple
                ([0, 128, 128], "Brown"),     # Brown
            ]
            
            # Create solid color backgrounds
            for color, name in colors:
                bg = np.ones((720, 1280, 3), dtype=np.uint8)
                bg[:] = color
                self.backgrounds.append(bg)
                print(f"Created {name} background")

            # Create gradient backgrounds
            # Horizontal gradient
            gradient_h = np.zeros((720, 1280, 3), dtype=np.uint8)
            for i in range(1280):
                gradient_h[:, i] = [int(255 * i / 1280), 0, 0]  # Blue gradient
            self.backgrounds.append(gradient_h)
            
            # Vertical gradient
            gradient_v = np.zeros((720, 1280, 3), dtype=np.uint8)
            for i in range(720):
                gradient_v[i, :] = [0, int(255 * i / 720), 0]  # Green gradient
            self.backgrounds.append(gradient_v)
            
            # Diagonal gradient (three points)
            gradient_d = np.zeros((720, 1280, 3), dtype=np.uint8)
            for i in range(720):
                for j in range(1280):
                    # Calculate distances from three points
                    d1 = np.sqrt((i/720)**2 + (j/1280)**2)
                    d2 = np.sqrt(((i-720)/720)**2 + ((j-1280)/1280)**2)
                    d3 = np.sqrt(((i-360)/720)**2 + ((j-640)/1280)**2)
                    gradient_d[i, j] = [
                        int(255 * d1),
                        int(255 * d2),
                        int(255 * d3)
                    ]
            self.backgrounds.append(gradient_d)
            
            print("Created gradient backgrounds")
            
        except Exception as e:
            print(f"Error creating default backgrounds: {e}")
            raise


    def add_ar_effects(self, image, frame_rgb):
        """Add AR effects with head tracking"""
        if not self.ar_props or not self.show_ar:
            return image
            
        try:
            results = self.face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get face dimensions
                    h, w = image.shape[:2]
                    
                    # Get key points for head tracking
                    nose = face_landmarks.landmark[1]  # Nose tip
                    left_eye = face_landmarks.landmark[33]  # Left eye
                    right_eye = face_landmarks.landmark[263]  # Right eye
                    
                    # Calculate face dimensions and position
                    face_width = int(abs(right_eye.x - left_eye.x) * w * 2)
                    face_height = int(face_width * 0.8)  # Approximate height
                    
                    # Calculate AR prop position (above head)
                    x = int(nose.x * w - face_width / 2)
                    y = int(nose.y * h - face_height * 1.5)
                    
                    # Ensure coordinates are valid
                    x = max(0, min(x, w - face_width))
                    y = max(0, min(y, h - face_height))
                    
                    if self.ar_props and len(self.ar_props) > self.current_ar_effect:
                        prop = self.ar_props[self.current_ar_effect]
                        try:
                            prop_resized = cv2.resize(prop, (face_width, face_height))
                            image = self.overlay_image_alpha(image, prop_resized, x, y)
                        except Exception as e:
                            print(f"AR prop resize error: {e}")
            
            return image
            
        except Exception as e:
            print(f"AR effect error: {e}")
            return image

    def load_ar_props(self):
        """Load and validate AR props"""
        try:
            ar_folder = "ar_props"
            for file in os.listdir(ar_folder):
                if file.lower().endswith('.png'):
                    try:
                        prop_path = os.path.join(ar_folder, file)
                        # Load image with alpha channel
                        prop = cv2.imread(prop_path, cv2.IMREAD_UNCHANGED)
                        
                        if prop is not None and prop.shape[2] == 4:  # Ensure RGBA format
                            self.ar_props.append(prop)
                            print(f"Successfully loaded AR prop: {file}")
                        else:
                            print(f"Invalid AR prop (must be RGBA): {file}")
                    except Exception as e:
                        print(f"Error loading AR prop {file}: {e}")
            
            print(f"Loaded {len(self.ar_props)} AR props")
        except Exception as e:
            print(f"AR prop loading error: {e}")
            raise

    def load_video_backgrounds(self):
        """Load and validate video backgrounds"""
        try:
            video_folder = "video_backgrounds"
            for file in os.listdir(video_folder):
                if file.lower().endswith(('.mp4', '.avi')):
                    try:
                        video_path = os.path.join(video_folder, file)
                        # Test if video is readable
                        test_cap = cv2.VideoCapture(video_path)
                        if test_cap.isOpened():
                            self.video_backgrounds.append(video_path)
                            test_cap.release()
                            print(f"Successfully loaded video background: {file}")
                        else:
                            print(f"Invalid video file: {file}")
                    except Exception as e:
                        print(f"Error loading video {file}: {e}")
            
            print(f"Loaded {len(self.video_backgrounds)} video backgrounds")
        except Exception as e:
            print(f"Video background loading error: {e}")
            raise

    def get_video_background(self, h, w):
        """Get current video background frame"""
        try:
            if not self.video_backgrounds:
                return np.ones((h, w, 3), dtype=np.uint8) * [0, 120, 255]
            
            if not hasattr(self, 'current_video') or self.current_video is None:
                self.current_video = cv2.VideoCapture(self.video_backgrounds[self.current_video_index])
            
            ret, video_frame = self.current_video.read()
            if not ret:
                self.current_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, video_frame = self.current_video.read()
            
            if ret and video_frame is not None:
                return cv2.resize(video_frame, (w, h))
            
            return np.ones((h, w, 3), dtype=np.uint8) * [0, 120, 255]
            
        except Exception as e:
            print(f"Video background error: {e}")
            return np.ones((h, w, 3), dtype=np.uint8) * [0, 120, 255]
        
    
    def change_background_mode(self, index):
        """Handle background mode changes"""
        try:
            # Clean up previous video
            if hasattr(self, 'current_video') and self.current_video is not None:
                self.current_video.release()
                self.current_video = None
            
            # Get selected mode
            mode = self.bg_combo.currentText()
            
            # Update UI based on mode
            if mode == "Static Backgrounds":
                self.preview_area.setVisible(True)
                self.effects_widget.blur_slider.setEnabled(False)
                self.current_bg_index = 0  # Reset to first background
            elif mode == "Video Backgrounds":
                self.preview_area.setVisible(False)
                self.effects_widget.blur_slider.setEnabled(False)
                self.current_video_index = 0  # Reset to first video
                self.init_video_background()
            elif mode == "Blur Background":
                self.preview_area.setVisible(False)
                self.effects_widget.blur_slider.setEnabled(True)
            elif mode == "Portrait Mode":
                self.preview_area.setVisible(False)
                self.effects_widget.blur_slider.setEnabled(True)
            
            self.statusBar.showMessage(f"Switched to {mode}")
        except Exception as e:
            print(f"Mode change error: {e}")

    def init_video_background(self):
        """Initialize current video background"""
        try:
            if self.video_backgrounds:
                if hasattr(self, 'current_video') and self.current_video is not None:
                    self.current_video.release()
                
                video_path = self.video_backgrounds[self.current_video_index]
                self.current_video = cv2.VideoCapture(video_path)
                print(f"Initialized video: {video_path}")
                
                if not self.current_video.isOpened():
                    print(f"Failed to open video: {video_path}")
        except Exception as e:
            print(f"Video init error: {e}")

    def background_selected(self, index):
        """Handle background selection"""
        try:
            if 0 <= index < len(self.backgrounds):
                self.current_bg_index = index
                self.bg_combo.setCurrentText("Static Backgrounds")
                print(f"Selected background {index}")
        except Exception as e:
            print(f"Background selection error: {e}")

    def next_background(self):
        """Switch to next background"""
        try:
            mode = self.bg_combo.currentText()
            if mode == "Static Backgrounds":
                self.current_bg_index = (self.current_bg_index + 1) % len(self.backgrounds)
            elif mode == "Video Backgrounds":
                self.current_video_index = (self.current_video_index + 1) % len(self.video_backgrounds)
                self.init_video_background()
        except Exception as e:
            print(f"Next background error: {e}")

    def prev_background(self):
        """Switch to previous background"""
        try:
            mode = self.bg_combo.currentText()
            if mode == "Static Backgrounds":
                self.current_bg_index = (self.current_bg_index - 1) % len(self.backgrounds)
            elif mode == "Video Backgrounds":
                self.current_video_index = (self.current_video_index - 1) % len(self.video_backgrounds)
                self.init_video_background()
        except Exception as e:
            print(f"Previous background error: {e}")

    def init_processing_tools(self):
        """Initialize MediaPipe tools"""
        try:
            # Initialize MediaPipe Selfie Segmentation
            self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
            self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
            
            # Initialize MediaPipe Face Mesh for better AR positioning
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            print("Processing tools initialized successfully")
        except Exception as e:
            print(f"Processing tools initialization error: {e}")
            raise

    def init_camera(self):
        """Initialize and configure camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.try_alternate_camera()
            
            self.configure_camera()
            print("Camera initialized successfully")
        except Exception as e:
            print(f"Camera initialization error: {e}")
            raise

    def try_alternate_camera(self):
        """Try different camera indices"""
        for i in range(1, 5):
            try:
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    print(f"Successfully opened camera at index {i}")
                    return
            except Exception as e:
                print(f"Error trying camera index {i}: {e}")
        
        raise RuntimeError("No camera found. Please check your camera connection.")

    def configure_camera(self):
        """Configure camera properties"""
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Verify camera settings
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera configured: {actual_width}x{actual_height} @ {actual_fps}fps")
        except Exception as e:
            print(f"Camera configuration error: {e}")
            raise

    def load_resources(self):
        """Load all required resources"""
        try:
            # Create necessary directories
            self.create_directories()
            
            # Load different types of resources
            self.load_backgrounds()
            self.load_video_backgrounds()
            self.load_ar_props()
            
            print("All resources loaded successfully")
        except Exception as e:
            print(f"Resource loading error: {e}")
            raise

    def create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = ['backgrounds', 'video_backgrounds', 'ar_props', 'screenshots', 'recordings']
        for directory in directories:
            try:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    print(f"Created directory: {directory}")
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")
                raise

    def load_backgrounds(self):
        """Load and validate static backgrounds"""
        try:
            bg_folder = "backgrounds"
            for file in os.listdir(bg_folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img_path = os.path.join(bg_folder, file)
                        img = cv2.imread(img_path)
                        if img is not None and img.shape[2] == 3:  # Ensure valid color image
                            img = cv2.resize(img, (1280, 720))
                            self.backgrounds.append(img)
                            print(f"Successfully loaded background: {file}")
                        else:
                            print(f"Invalid background format: {file}")
                    except Exception as e:
                        print(f"Error loading background {file}: {e}")
            
            print(f"Loaded {len(self.backgrounds)} backgrounds")
        except Exception as e:
            print(f"Background loading error: {e}")
            raise




if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        
        # Set application style
        app.setStyle('Fusion')
        
        # Set dark theme
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        
        app.setPalette(palette)
        
        # Create and show main window
        window = VirtualBackgroundApp()
        window.show()
        
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Application Error: {e}")
        sys.exit(1)        