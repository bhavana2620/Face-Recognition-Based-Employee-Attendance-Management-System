"""
Face Recognition Engine Module
Handles OpenCV operations: face detection, dataset collection,
LBPH model training, and real-time streaming recognition.
"""

import os
import cv2
import numpy as np
from PIL import Image
from models import log_attendance, get_all_employees, get_employee_by_id

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")
HAAR_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Ensure runtime directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)

face_detector = cv2.CascadeClassifier(HAAR_FILE)


def generate_face_dataset(emp_id, max_samples=60):
    """
    Captures face frames from the webcam, crops them, converts to grayscale,
    and stores them in dataset/ as User.<int_id>.<sample_no>.jpg.
    Returns generator yielding progress percentage.
    """
    # Map alphanumeric employee_id to an integer hash for LBPH compatibility
    int_id = abs(hash(emp_id)) % (10**8)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam is not accessible.")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100)
        )

        for x, y, w, h in faces:
            count += 1
            face_img = gray[y : y + h, x : x + w]
            file_name = f"User.{int_id}.{emp_id}.{count}.jpg"
            file_path = os.path.join(DATASET_DIR, file_name)
            cv2.imwrite(file_path, face_img)
            break  # Take only primary face per frame

        if count >= max_samples:
            break

        # Safety escape
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return count


def train_model():
    """
    Reads all grayscale face images in dataset/, trains an
    LBPH Face Recognizer, and saves the weights to trainer/trainer.yml.
    """
    image_paths = [
        os.path.join(DATASET_DIR, f)
        for f in os.listdir(DATASET_DIR)
        if f.endswith(".jpg")
    ]
    if not image_paths:
        return False, "No face samples found in the dataset folder."

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_samples = []
    ids = []

    for img_path in image_paths:
        try:
            pil_img = Image.open(img_path).convert("L")
            img_numpy = np.array(pil_img, "uint8")

            # Extract numeric id from filename: User.<int_id>.<emp_id>.<count>.jpg
            filename = os.path.split(img_path)[-1]
            parts = filename.split(".")
            int_id = int(parts[1])

            faces = face_detector.detectMultiScale(img_numpy)
            for x, y, w, h in faces:
                face_samples.append(img_numpy[y : y + h, x : x + w])
                ids.append(int_id)
        except Exception:
            continue

    if not face_samples:
        return (
            False,
            "Could not extract facial features from provided dataset.",
        )

    recognizer.train(face_samples, np.array(ids))
    recognizer.write(TRAINER_FILE)

    return (
        True,
        f"Model trained successfully on {len(face_samples)} face images.",
    )


def build_id_mapping():
    """
    Maps LBPH integer IDs back to string Employee IDs and metadata.
    """
    employees = get_all_employees()
    mapping = {}
    for emp in employees:
        int_id = abs(hash(emp["employee_id"])) % (10**8)
        mapping[int_id] = {
            "employee_id": emp["employee_id"],
            "name": emp["name"],
            "department": emp["department"],
        }
    return mapping


def generate_live_recognition_stream():
    """
    Generator function that captures frames, performs detection and recognition,
    draws overlays, updates attendance records, and yields MJPEG frames.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if not os.path.exists(TRAINER_FILE):
        # Stream error message image if no model exists
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            blank_frame,
            "Model not trained. Train model first!",
            (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        _, buffer = cv2.imencode(".jpg", blank_frame)
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )
        return

    recognizer.read(TRAINER_FILE)
    id_map = build_id_mapping()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100)
        )

        for x, y, w, h in faces:
            roi_gray = gray[y : y + h, x : x + w]
            predicted_id, confidence = recognizer.predict(roi_gray)

            # In OpenCV LBPH: lower confidence number means better match (0 is exact match)
            # Threshold: confidence < 75 is considered a reliable match
            if confidence < 75 and predicted_id in id_map:
                emp_info = id_map[predicted_id]
                emp_id = emp_info["employee_id"]
                name = emp_info["name"]
                conf_score = f"{round(100 - confidence, 1)}%"

                # Color: Green for recognized
                box_color = (0, 255, 0)
                display_text = f"{name} ({emp_id})"
                score_text = f"Conf: {conf_score}"

                # Trigger automated attendance marking
                log_attendance(emp_id)
            else:
                box_color = (0, 0, 255)  # Red for unknown
                display_text = "Unknown Person"
                score_text = "N/A"

            # Draw bounding box and header text
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.rectangle(
                frame, (x, y - 35), (x + w, y), box_color, cv2.FILLED
            )
            cv2.putText(
                frame,
                display_text,
                (x + 6, y - 18),
                cv2.FONT_HERSHEY_DUPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                frame,
                score_text,
                (x + 6, y - 4),
                cv2.FONT_HERSHEY_DUPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

        # Encode frame as JPEG
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

    cap.release()