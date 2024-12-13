# VoiceControlled-BackgroundChanger 🎥

A sophisticated Python application that revolutionizes your video call experience with real-time background replacement, voice control, AR effects, and audio visualization. Perfect for virtual meetings, content creation, and streaming.

## 🌟 Key Features

### Background Management
- **Real-time Background Replacement**: Seamless background switching during live video
- **Multiple Background Types**: 
  - Static image backgrounds
  - Video backgrounds
  - Blur effect
  - Portrait mode with depth effect
- **Voice Control**: Change backgrounds using voice commands
  - "Next" - Switch to next background
  - "Previous" - Switch to previous background

### AR Effects
- **Real-time Head Tracking**: AR props follow your head movements
- **Customizable Props**: Support for custom AR overlay elements
- **Size Controls**: Adjust AR prop size in real-time
- **Position Tracking**: Natural movement and positioning

### Audio Features
- **Voice Command Recognition**: Hands-free control
- **Real-time Audio Visualization**: See your microphone input levels
- **Integrated Audio Processing**: Clean audio recording capabilities

### Recording & Output
- **High-Quality Recording**: Combined audio/video recording
- **Screenshot Function**: Capture moments instantly
- **Multiple Output Formats**: Support for various video codecs
- **Customizable Quality**: Adjustable recording parameters

## 🛠️ Technology Stack

- **Core Technologies**:
  - Python 3.8+
  - OpenCV
  - MediaPipe
  - PyQt5
  - Speech Recognition
  - PyAudio
  - NumPy

## 🚀 Getting Started

### Prerequisites

```bash
# Required software
- Python 3.8 or higher
- pip package manager
- Webcam
- Microphone
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/officialjwise/VoiceControlled-BackgroundChanger.git
cd VoiceControlled-BackgroundChanger
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up project structure:
```
project_folder/
├── backgrounds/         # Background images (.jpg/.png)
├── video_backgrounds/   # Video files (.mp4/.avi)
├── ar_props/           # AR props (.png with transparency)
├── screenshots/        # Generated screenshots
├── recordings/         # Saved recordings
└── main.py            # Application entry point
```

### Configuration

1. Add background images to `backgrounds/`
2. Add video files to `video_backgrounds/`
3. Add AR props to `ar_props/`

### Running the Application

```bash
python main.py
```

## 💡 Usage Guide

### Basic Controls
- **Background Selection**: Use dropdown menu or voice commands
- **AR Effects**: Toggle button in control panel
- **Recording**: Start/Stop button for video capture
- **Screenshots**: Capture button for still images

### Voice Commands
- Say "Next" to switch to next background
- Say "Previous" to return to previous background

### Keyboard Shortcuts
- `Esc` - Exit application
- `Space` - Toggle AR effects
- `S` - Take screenshot
- `R` - Start/Stop recording

## 📂 Project Structure

```
src/
├── main.py                    # Entry point
├── background_handler.py      # Background processing
├── ar_effects.py             # AR implementation
├── audio_handler.py          # Audio processing
└── ui/                       # UI components
    ├── main_window.py
    ├── controls.py
    └── previews.py
```

## 🎯 Troubleshooting

### Common Issues & Solutions

1. **Camera Access**
   - Ensure no other applications are using the camera
   - Check camera permissions
   - Try restarting the application

2. **Voice Commands**
   - Verify microphone permissions
   - Check internet connection
   - Speak clearly and maintain proper distance

3. **Video Playback**
   - Ensure video codec compatibility (H.264 recommended)
   - Check video file integrity
   - Verify supported format (.mp4/.avi)

## 🤝 Contributing

We welcome contributions! See our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License. See [LICENSE](LICENSE) file for details.

## 🙌 Acknowledgments

- **OpenCV Team**: For computer vision capabilities
- **MediaPipe**: For face detection and segmentation
- **PyQt**: For the GUI framework
- **Python Community**: For various helpful resources
- **All Contributors**: Who help improve this project

## 📞 Contact & Support

- **Developer**: Official J-Wise
- **Twitter/X**: [@daniel4_kodua](https://x.com/daniel4_kodua)
- **Project Repository**: [VoiceControlled-BackgroundChanger](https://github.com/officialjwise/VoiceControlled-BackgroundChanger)

## ✨ Future Plans

- [ ] Additional voice commands
- [ ] More AR effects
- [ ] Custom background filters
- [ ] Stream integration
- [ ] Multi-language support

---

<div align="center">
⌨️ with ❤️ by Official J-Wise 🚀

If you found this project helpful, please consider giving it a ⭐️!
</div>
