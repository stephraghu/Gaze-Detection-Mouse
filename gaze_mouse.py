import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import os
import sys
import time

# Constants
SMOOTHING_FACTOR = 4
BLINK_THRESHOLD = 0.004
CLICK_THRESHOLD_FRAMES = 15
CAMERA_INDEX = 1

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# Left iris landmarks
LEFT_IRIS = [473, 474, 475, 476, 477]

# Left eye landmarks
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

# Extra eye landmarks for drawing the eye
LEFT_EYE_OUTER = 263
LEFT_EYE_INNER = 362
LEFT_EYE_UPPER = 386
LEFT_EYE_LOWER = 374


def get_iris_center(landmarks, iris_indices, frame_w, frame_h):
    x_coords = []
    y_coords = []

    for idx in iris_indices:
        lm = landmarks[idx]
        x_coords.append(lm.x * frame_w)
        y_coords.append(lm.y * frame_h)

    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))

    return center_x, center_y


def detect_blink(landmarks, top_idx, bottom_idx):
    top = landmarks[top_idx]
    bottom = landmarks[bottom_idx]

    distance = abs(top.y - bottom.y)

    return distance < BLINK_THRESHOLD


def map_to_screen(
    iris_x,
    iris_y,
    frame_w,
    frame_h,
    screen_w,
    screen_h
):
    norm_x = iris_x / frame_w
    norm_y = iris_y / frame_h

    scale_x = 2.5
    scale_y = 2.5

    mapped_x = (norm_x - 0.5) * scale_x + 0.5
    mapped_y = (norm_y - 0.5) * scale_y + 0.5

    mapped_x = np.clip(mapped_x, 0, 1)
    mapped_y = np.clip(mapped_y, 0, 1)

    return (
        int(mapped_x * screen_w),
        int(mapped_y * screen_h)
    )


def landmark_to_pixel(landmark, frame_w, frame_h):
    return (
        int(landmark.x * frame_w),
        int(landmark.y * frame_h)
    )


def draw_eye_tracking(frame, landmarks, iris_x, iris_y):
    frame_h, frame_w, _ = frame.shape

    # Draw the iris landmarks
    for idx in LEFT_IRIS:
        point = landmark_to_pixel(
            landmarks[idx],
            frame_w,
            frame_h
        )

        cv2.circle(
            frame,
            point,
            3,
            (255, 100, 0),
            -1
        )

    # Draw eye outline
    eye_points = [
        LEFT_EYE_OUTER,
        LEFT_EYE_UPPER,
        LEFT_EYE_INNER,
        LEFT_EYE_LOWER,
        LEFT_EYE_OUTER
    ]

    for i in range(len(eye_points) - 1):
        p1 = landmark_to_pixel(
            landmarks[eye_points[i]],
            frame_w,
            frame_h
        )

        p2 = landmark_to_pixel(
            landmarks[eye_points[i + 1]],
            frame_w,
            frame_h
        )

        cv2.line(
            frame,
            p1,
            p2,
            (0, 255, 255),
            2
        )


    # Draw iris center
    cv2.circle(
        frame,
        (iris_x, iris_y),
        12,
        (0, 255, 0),
        2
    )

    cv2.circle(
        frame,
        (iris_x, iris_y),
        4,
        (0, 255, 0),
        -1
    )


    # Crosshair around iris
    cv2.line(
        frame,
        (iris_x - 20, iris_y),
        (iris_x - 8, iris_y),
        (0, 255, 0),
        2
    )

    cv2.line(
        frame,
        (iris_x + 8, iris_y),
        (iris_x + 20, iris_y),
        (0, 255, 0),
        2
    )

    cv2.line(
        frame,
        (iris_x, iris_y - 20),
        (iris_x, iris_y - 8),
        (0, 255, 0),
        2
    )

    cv2.line(
        frame,
        (iris_x, iris_y + 8),
        (iris_x, iris_y + 20),
        (0, 255, 0),
        2
    )


def main():

    if not os.path.exists(MODEL_PATH):
        print("ERROR: face_landmarker.task not found.")
        print("Expected location:")
        print(MODEL_PATH)
        sys.exit(1)


    # Screen size
    screen_w, screen_h = pyautogui.size()

    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = True


    # MediaPipe setup
    BaseOptions = mp.tasks.BaseOptions
    FaceLandMarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )


    # Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    # Optional camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Tracking variables
    smooth_x = screen_w // 2
    smooth_y = screen_h // 2

    click_cooldown = 0
    last_timestamp_ms = 0

    # MediaPipe tracker
    with FaceLandMarker.create_from_options(options) as face_landmarker:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # Mirror camera
            frame = cv2.flip(frame, 1)

            frame_h, frame_w, _ = frame.shape

            # Convert frame for MediaPipe
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Timestamp
            timestamp_ms = int(time.time() * 1000)

            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1

            last_timestamp_ms = timestamp_ms

            # Detect face
            result = face_landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # Status panel
            status_text = "NO FACE DETECTED"
            status_color = (0, 0, 255)

            if result.face_landmarks:

                landmarks = result.face_landmarks[0]

                # Get iris position
                iris_x, iris_y = get_iris_center(
                    landmarks,
                    LEFT_IRIS,
                    frame_w,
                    frame_h
                )


                # Map iris to screen
                target_x, target_y = map_to_screen(
                    iris_x,
                    iris_y,
                    frame_w,
                    frame_h,
                    screen_w,
                    screen_h
                )


                # Smooth cursor
                smooth_x += (
                    target_x - smooth_x
                ) / SMOOTHING_FACTOR

                smooth_y += (
                    target_y - smooth_y
                ) / SMOOTHING_FACTOR


                # Move mouse
                try:
                    pyautogui.moveTo(
                        int(smooth_x),
                        int(smooth_y)
                    )

                except pyautogui.FailSafeException:
                    smooth_x = screen_w // 2
                    smooth_y = screen_h // 2


                # Blink / click
                if click_cooldown > 0:
                    click_cooldown -= 1

                blinking = detect_blink(
                    landmarks,
                    LEFT_EYE_TOP,
                    LEFT_EYE_BOTTOM
                )

                if blinking:

                    status_text = "BLINK - CLICK"
                    status_color = (255, 0, 255)

                    if click_cooldown == 0:
                        pyautogui.click()
                        click_cooldown = CLICK_THRESHOLD_FRAMES

                else:
                    status_text = "EYE TRACKING ACTIVE"
                    status_color = (0, 255, 0)

                # Draw eye tracking
                draw_eye_tracking(
                    frame,
                    landmarks,
                    iris_x,
                    iris_y
                )


                # Draw line showing tracking direction
                cv2.line(
                    frame,
                    (iris_x, iris_y),
                    (
                        int(
                            iris_x +
                            (iris_x - frame_w / 2) * 0.8
                        ),
                        int(
                            iris_y +
                            (iris_y - frame_h / 2) * 0.8
                        )
                    ),
                    (0, 200, 255),
                    2
                )


                # Tracking information
                cv2.putText(
                    frame,
                    f"Iris: ({iris_x}, {iris_y})",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Cursor: ({int(smooth_x)}, {int(smooth_y)})",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )


            # Status text
            cv2.rectangle(
                frame,
                (10, 10),
                (330, 55),
                (20, 20, 20),
                -1
            )

            cv2.putText(
                frame,
                status_text,
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                status_color,
                2
            )


            # Instructions
            cv2.putText(
                frame,
                "q = quit",
                (20, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )


            # Show camera
            cv2.imshow(
                "Eye Tracking Camera",
                frame
            )

            # Quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
