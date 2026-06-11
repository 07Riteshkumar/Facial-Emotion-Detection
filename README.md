# Facial-Emotion-Detection
Smart Facial Emotion Detection
🎭 Facial Emotion Detection

Real-time emotion recognition from webcam, image, or video — powered by DeepFace & OpenCV


<img width="1200" height="700" alt="664d6fec8ce56bb678b9d215_Facial emotion recognition" src="https://github.com/user-attachments/assets/8fabc507-58c6-4554-80d9-a9614c159904" />


📌 Overview

Facial Emotion Detection is a lightweight Python application that detects human faces in real time and classifies their emotional state using deep learning. It supports three input modes — live webcam, image file, and video file — and overlays a colour-coded bounding box, dominant emotion label, and a live confidence bar chart directly onto the video feed.

Built for clarity and ease of use, every function is documented and the code is structured so that even beginners can read, understand, and extend it.


✨ Features


🎥 Live webcam emotion detection with real-time FPS counter
🖼️ Single image analysis with annotated output window
📹 Video file processing frame-by-frame with overlay
📊 Per-face confidence bar chart showing all 7 emotion scores
🎨 Colour-coded bounding boxes per emotion
💾 Snapshot saving (press S during webcam mode)
🛡️ Secure file handling — path traversal protection + extension allowlist
⚡ Frame-skip optimisation — analyses every 5th frame for smooth playback
✅ Python 3.10 – 3.13 compatible (64-bit, Windows / macOS / Linux)



🧠 Detected Emotions

EmotionColour😄 HappyGreen😢 SadBlue😠 AngryRed😲 SurpriseYellow😨 FearPurple🤢 DisgustOrange😐 NeutralGrey


🗂️ Project Structure

facial-emotion-detection/
│
├── emotion_detector.py   # Main application — all logic lives here
├── requirements.txt      # Python dependencies
└── README.md             # This file


⚙️ Requirements


Python 3.10 – 3.13 (64-bit)
Webcam (for live mode)
~500 MB disk space (TensorFlow + model weights downloaded on first run)



🚀 Installation & Setup

1 — Clone the repository

bashgit clone https://github.com/your-username/facial-emotion-detection.git
cd facial-emotion-detection

2 — (Recommended) Create a virtual environment

bashpython -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

3 — Install dependencies

bashpip install -r requirements.txt


Note: On first run DeepFace automatically downloads the emotion model weights (~100 MB). An internet connection is required once.




▶️ Usage

Webcam (default)

bashpython emotion_detector.py

Image file

bashpython emotion_detector.py --mode image --source photo.jpg

Video file

bashpython emotion_detector.py --mode video --source clip.mp4

Second webcam / external camera

bashpython emotion_detector.py --camera 1

All arguments

--mode    webcam | image | video   (default: webcam)
--source  path to image or video file
--camera  camera device index      (default: 0)


⌨️ Keyboard Controls

KeyActionQ or ESCQuit the applicationSSave a snapshot as snapshot_N.jpg (webcam mode)Any keyClose window (image mode)


🛠️ PyCharm Setup


Open PyCharm → File → Open → select the project folder
Go to File → Settings → Project → Python Interpreter
Click Add Interpreter → select Python 3.13 (64-bit)
Open the Terminal tab inside PyCharm and run:


bash   pip install -r requirements.txt


Right-click emotion_detector.py → Run 'emotion_detector'


To pass arguments: Run → Edit Configurations → Script parameters

--mode image --source C:\Users\you\photo.jpg


🧩 How It Works

Webcam / Image / Video
        │
        ▼
  Read frame (OpenCV)
        │
        ▼  every 5th frame
  DeepFace.analyze()
        │
        ▼
  Dominant emotion + confidence scores (7 emotions)
        │
        ▼
  Draw bounding box + label + bar chart (OpenCV)
        │
        ▼
  Display / Save

The core pipeline:


analyse_frame() — sends a BGR frame to DeepFace, returns face regions and emotion scores
draw_results() — draws bounding boxes, labels, and bar charts onto the frame
run_webcam() — main loop for live camera input
run_image() — loads, analyses, and displays a single image
run_video() — iterates through a video file frame by frame



🛡️ Security


os.path.realpath() resolves all user-supplied paths, preventing directory traversal attacks
An extension allowlist restricts input to known-safe image and video formats (.jpg, .png, .mp4, .avi, etc.)
All processing runs 100% locally — no data is sent to any server



📦 Dependencies

PackagePurposeopencv-python >= 4.10Webcam / image / video I/O and drawingdeepface >= 0.0.93Face detection and emotion recognitiontensorflow >= 2.17Deep learning backend (Python 3.13 compatible)numpy >= 1.26Array operationsPillow >= 10.0Image preprocessing


🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.


Fork the repository
Create your feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request



📄 License

This project is licensed under the MIT License — see the LICENSE file for details.


🙏 Acknowledgements


DeepFace by Sefik Ilkin Serengil
OpenCV — Open Source Computer Vision Library
TensorFlow by Google Brain Team
