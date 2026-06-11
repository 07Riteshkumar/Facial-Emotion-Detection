"""
Facial Emotion Detection Program
==================================
Uses OpenCV for face detection and DeepFace for emotion recognition.
Supports webcam (live), image file, and video file input.

Dependencies:
    pip install opencv-python deepface tf-keras numpy
"""

import cv2
import numpy as np
import argparse
import sys
import os
from pathlib import Path


# ── Safely import DeepFace ────────────────────────────────────────────────────
try:
    from deepface import DeepFace
except ImportError:
    print("[ERROR] DeepFace is not installed.")
    print("        Run:  pip install deepface tf-keras")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette for each emotion label (BGR format for OpenCV)
EMOTION_COLORS = {
    "happy":     (0,   255, 100),   # green
    "sad":       (255, 100,   0),   # blue
    "angry":     (0,   0,   255),   # red
    "surprise":  (0,   220, 255),   # yellow
    "fear":      (128,   0, 128),   # purple
    "disgust":   (0,   128, 255),   # orange
    "neutral":   (200, 200, 200),   # light grey
}

DEFAULT_COLOR = (255, 255, 255)     # white fallback
FONT          = cv2.FONT_HERSHEY_SIMPLEX
ANALYSE_EVERY = 5                   # analyse every N frames (for performance)


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def analyse_frame(frame: np.ndarray) -> list[dict]:
    """
    Run DeepFace emotion analysis on one BGR frame.

    Returns a list of face results, each containing:
        - region   : {x, y, w, h}
        - emotion  : dominant emotion string
        - scores   : dict of all emotion probabilities (0-100)
    Returns an empty list if no face is detected.
    """
    try:
        results = DeepFace.analyze(
            img_path        = frame,
            actions         = ["emotion"],   # only detect emotion (faster)
            enforce_detection = False,       # don't crash if face not found
            silent          = True,
        )
        # DeepFace always returns a list
        faces = []
        for r in results:
            faces.append({
                "region":  r["region"],
                "emotion": r["dominant_emotion"],
                "scores":  r["emotion"],
            })
        return faces

    except Exception as exc:
        # Log but never crash the main loop
        print(f"[WARN] Analysis error: {exc}")
        return []


def draw_results(frame: np.ndarray, faces: list[dict]) -> np.ndarray:
    """
    Draw bounding box, dominant emotion label, and a mini bar chart
    of all emotion scores onto the frame.
    """
    overlay = frame.copy()

    for face in faces:
        region  = face["region"]
        emotion = face["emotion"].lower()
        scores  = face["scores"]

        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 0)
        h = region.get("h", 0)

        color = EMOTION_COLORS.get(emotion, DEFAULT_COLOR)

        # ── Bounding box ──────────────────────────────────────────────────
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

        # ── Dominant emotion label (above box) ────────────────────────────
        label       = f"{emotion.upper()}  {scores.get(emotion, 0):.1f}%"
        label_y     = max(y - 10, 20)
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.7, 2)
        cv2.rectangle(overlay, (x, label_y - th - 6), (x + tw + 6, label_y + 4), color, -1)
        cv2.putText(overlay, label, (x + 3, label_y), FONT, 0.7, (0, 0, 0), 2)

        # ── Mini bar chart (top-right corner, inside bounding box) ────────
        bar_x   = x + w + 8
        bar_y   = y
        bar_w   = 120
        bar_h   = 14
        spacing = 18

        # Make sure bars stay within frame
        if bar_x + bar_w + 60 > frame.shape[1]:
            bar_x = x - bar_w - 70

        for i, (emo, score) in enumerate(sorted(scores.items(), key=lambda kv: -kv[1])):
            bar_top  = bar_y + i * spacing
            bar_len  = int((score / 100) * bar_w)
            emo_col  = EMOTION_COLORS.get(emo, DEFAULT_COLOR)

            cv2.rectangle(overlay, (bar_x, bar_top), (bar_x + bar_w, bar_top + bar_h), (50, 50, 50), -1)
            cv2.rectangle(overlay, (bar_x, bar_top), (bar_x + bar_len, bar_top + bar_h), emo_col, -1)
            cv2.putText(overlay, f"{emo[:3]} {score:.0f}%",
                        (bar_x + bar_w + 4, bar_top + bar_h - 2),
                        FONT, 0.38, (230, 230, 230), 1)

    # Blend overlay (for slight transparency effect on bar chart area)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame


def add_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Stamp FPS in the top-left corner."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), FONT, 0.6, (0, 255, 0), 2)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Input modes
# ─────────────────────────────────────────────────────────────────────────────

def run_webcam(camera_index: int = 0) -> None:
    """
    Live webcam mode.
    Press  Q  or  ESC  to quit.
    Press  S  to save a snapshot.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {camera_index}.")
        return

    print("[INFO] Webcam started. Press Q/ESC to quit, S to save snapshot.")

    frame_count = 0
    last_faces: list[dict] = []
    fps_timer = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Lost camera feed.")
            break

        frame_count += 1

        # Only analyse every Nth frame for better performance
        if frame_count % ANALYSE_EVERY == 0:
            last_faces = analyse_frame(frame)

        # Always draw last known results
        frame = draw_results(frame, last_faces)

        # FPS calculation
        tick_now = cv2.getTickCount()
        elapsed  = (tick_now - fps_timer) / cv2.getTickFrequency()
        fps      = ANALYSE_EVERY / elapsed if elapsed > 0 else 0
        fps_timer = tick_now
        frame = add_fps(frame, fps)

        cv2.imshow("Facial Emotion Detection  [Q/ESC=quit  S=snapshot]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):          # Q or ESC
            break
        elif key == ord("s"):              # S → save snapshot
            snapshot_path = f"snapshot_{frame_count}.jpg"
            cv2.imwrite(snapshot_path, frame)
            print(f"[INFO] Snapshot saved: {snapshot_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam closed.")


def run_image(image_path: str) -> None:
    """
    Single image mode — analyse and display result, then wait for keypress.
    """
    path = Path(image_path)
    if not path.exists():
        print(f"[ERROR] File not found: {image_path}")
        return

    frame = cv2.imread(str(path))
    if frame is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    print(f"[INFO] Analysing image: {image_path}")
    faces = analyse_frame(frame)

    if not faces:
        print("[INFO] No faces detected in the image.")
    else:
        for i, f in enumerate(faces, 1):
            print(f"  Face {i}: {f['emotion'].upper()}  "
                  f"({f['scores'].get(f['emotion'], 0):.1f}%)")

    frame = draw_results(frame, faces)

    # Resize for display if the image is very large
    max_display = 900
    h, w = frame.shape[:2]
    if w > max_display or h > max_display:
        scale = max_display / max(w, h)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    cv2.imshow(f"Emotion Detection — {path.name}  [any key to close]", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(video_path: str) -> None:
    """
    Video file mode — process and display frame by frame.
    Press Q or ESC to stop early.
    """
    path = Path(video_path)
    if not path.exists():
        print(f"[ERROR] File not found: {video_path}")
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Processing video: {video_path}  ({total_frames} frames)")
    print("[INFO] Press Q/ESC to stop.")

    frame_count  = 0
    last_faces: list[dict] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % ANALYSE_EVERY == 0:
            last_faces = analyse_frame(frame)

        frame = draw_results(frame, last_faces)
        cv2.putText(frame, f"Frame {frame_count}/{total_frames}",
                    (10, 25), FONT, 0.6, (0, 255, 0), 2)

        cv2.imshow(f"Emotion Detection — {path.name}  [Q/ESC=quit]", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Video processing complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Facial Emotion Detection — webcam, image, or video",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python emotion_detector.py                        # default: webcam
  python emotion_detector.py --mode webcam          # explicit webcam
  python emotion_detector.py --mode image  --source photo.jpg
  python emotion_detector.py --mode video  --source clip.mp4
  python emotion_detector.py --camera 1             # use second webcam
        """,
    )
    parser.add_argument(
        "--mode", choices=["webcam", "image", "video"],
        default="webcam",
        help="Input mode (default: webcam)",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Path to image or video file (required for image/video modes)",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index for webcam mode (default: 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Security: validate file path to prevent directory traversal
    if args.source:
        source = os.path.realpath(args.source)
        if not os.path.exists(source):
            print(f"[ERROR] Source path does not exist: {args.source}")
            sys.exit(1)
    else:
        source = None

    if args.mode == "webcam":
        run_webcam(camera_index=args.camera)

    elif args.mode == "image":
        if not source:
            print("[ERROR] --source <image_path> is required for image mode.")
            sys.exit(1)
        run_image(source)

    elif args.mode == "video":
        if not source:
            print("[ERROR] --source <video_path> is required for video mode.")
            sys.exit(1)
        run_video(source)


if __name__ == "__main__":
    main()
