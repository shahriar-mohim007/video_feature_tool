import cv2
import json
import numpy as np
import pytesseract
from PIL import Image


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    print(f"Video file '{video_path}' loaded successfully.")
    return cap


def detect_shot_cuts(cap, threshold=30):

    hard_cut_count = 0
    previous_frame_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if previous_frame_gray is not None:
            frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
            mean_diff = cv2.mean(frame_diff)[0]

            if mean_diff > threshold:
                hard_cut_count += 1

        previous_frame_gray = current_frame_gray

    extracted_features = {
        "description": "This feature detects how many hard cuts (scene changes) are present in the video.",
        "hard_cut_count": hard_cut_count
    }

    print(json.dumps(extracted_features, indent=4))


def motion_analysis(cap):

    if not cap.isOpened():
        print("Error: VideoCapture is not opened for motion analysis.")
        return

    # Rewind video to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    motion_magnitudes = []

    ret, frame1 = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            next_gray,
            np.zeros_like(prev_gray, dtype=np.float32),
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_magnitudes.append(float(mag.mean()))
        prev_gray = next_gray

    average_motion = float(np.mean(motion_magnitudes)) if motion_magnitudes else 0.0

    extracted_features = {
        "description": "This feature calculates average motion using optical flow.",
        "average_motion": average_motion
    }

    print(json.dumps(extracted_features, indent=4))


def text_detection(cap):

    if not cap.isOpened():
        print("Error: VideoCapture is not opened for text detection.")
        return


    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    total_frames = 0
    frames_with_text = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pil_image = Image.fromarray(gray)
        text = pytesseract.image_to_string(pil_image)

        if text.strip():
            frames_with_text += 1

    text_present_ratio = frames_with_text / total_frames if total_frames > 0 else 0.0

    extracted_features = {
        "description": "This feature calculates the ratio of frames containing text in the video.",
        "text_present_ratio": text_present_ratio
    }

    print(json.dumps(extracted_features, indent=4))


def main():
    video_path = "samples/video.mp4"


    cap = load_video(video_path)


    detect_shot_cuts(cap)


    motion_analysis(cap)


    text_detection(cap)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
