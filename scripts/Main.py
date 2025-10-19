
import os
import time
import cv2
import numpy as np
import face_recognition
import pickle
import dlib

# ===== GPIO shim: chạy được cả trên PC (không có RPi.GPIO) =====
try:
    import RPi.GPIO as _GPIO
    class GPIOWrap:
        GPIO = _GPIO
        RPI = True
        def __init__(self):
            self.GPIO.setmode(self.GPIO.BCM)
            self.GPIO.setwarnings(False)
            self.GPIO.setup(17, self.GPIO.OUT, initial=self.GPIO.LOW)   # relay/LED
            self.GPIO.setup(27, self.GPIO.IN, pull_up_down=self.GPIO.PUD_DOWN)  # button
        def cleanup(self):
            self.GPIO.cleanup()
        def door_on(self, sec=5):
            self.GPIO.output(17, self.GPIO.HIGH)
            time.sleep(sec)
            self.GPIO.output(17, self.GPIO.LOW)
        def button_pressed(self):
            return self.GPIO.input(27) == self.GPIO.HIGH
    HW = GPIOWrap()
except Exception:
    class _Dummy:
        RPI = False
        def cleanup(self): pass
        def door_on(self, sec=5):
            print(f"[SIM] Door open {sec}s (no GPIO).")
        def button_pressed(self): return False
    HW = _Dummy()

# ===== Đường dẫn & cấu hình =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_FILE = os.environ.get("ENCODINGS_FILE", os.path.join(BASE_DIR, "face_encodings.pkl"))
GUEST_IMAGE = os.environ.get("GUEST_IMAGE", os.path.join(BASE_DIR, "guest_image.jpeg"))
PREDICTOR_PATH = os.environ.get("DLIB_LANDMARKS", os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat"))

SKIP_FRAMES = 2          # xử lý 1/2 khung hình
RESIZE_SCALE = 0.5       # resize 50% để tăng tốc
TOLERANCE = 0.28         # ngưỡng so khớp mặt

# ===== Email (tùy chọn). Nếu không set ENV thì bỏ qua =====
SENDER_EMAIL    = os.environ.get("SENDER_EMAIL", None)
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", None)  # app password
RECEIVER_EMAIL  = os.environ.get("RECEIVER_EMAIL", None)

def send_notice(image_path):
    if not (SENDER_EMAIL and SENDER_PASSWORD and RECEIVER_EMAIL):
        print("[WARN] Email ENV not set -> skip sending notice.")
        return
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_PASSWORD)

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = "Intruder detected."
    msg.attach(MIMEText("An unknown person was detected at your door."))

    if os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            img = MIMEImage(f.read(), name=os.path.basename(image_path))
        msg.attach(img)

    server.send_message(msg)
    server.quit()
    print("[INFO] Notice sent.")

def takephoto(frame, save_path=GUEST_IMAGE):
    cv2.imwrite(save_path, frame)
    print(f"[INFO] Photo saved: {save_path}")

# ===== Liveness (nháy mắt) với dlib 68 landmarks =====
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(f"Missing landmarks model: {PREDICTOR_PATH}")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def blink_detection(frame, blink_state):
    # blink_state: dict with keys {'consec_below', 'blinks'}
    EAR_THRESHOLD = 0.25
    MIN_CONSEC = 3  # số khung liên tiếp coi là nhắm mắt
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    if len(faces) == 0:
        # không thấy mặt => reset nhẹ
        blink_state['consec_below'] = 0
        return blink_state

    for face in faces:
        lm = predictor(gray, face)
        left = np.array([(lm.part(i).x, lm.part(i).y) for i in range(36, 42)])
        right = np.array([(lm.part(i).x, lm.part(i).y) for i in range(42, 48)])
        ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
        if ear < EAR_THRESHOLD:
            blink_state['consec_below'] += 1
        else:
            if blink_state['consec_below'] >= MIN_CONSEC:
                blink_state['blinks'] += 1    # ghi nhận 1 nháy hoàn chỉnh
            blink_state['consec_below'] = 0
    return blink_state

# ===== Load encodings =====
if not os.path.exists(ENCODINGS_FILE):
    raise FileNotFoundError(f"Encodings file not found: {ENCODINGS_FILE}")
with open(ENCODINGS_FILE, 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)
print(f"[INFO] Loaded {len(known_face_names)} enrolled encodings.")

def verify_window(cap, seconds=10):
    """Cửa sổ xác thực trong 'seconds' giây. Trả về tên nếu hợp lệ, else 'Unknown'."""
    name = "Unknown"
    end_t = time.time() + seconds
    blink_state = {'consec_below': 0, 'blinks': 0}

    while time.time() < end_t:
        ok, frame = cap.read()
        if not ok:
            break

        # tăng tốc
        small = cv2.resize(frame, (0,0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs_small = face_recognition.face_locations(rgb_small, model="hog")
        encs_small = face_recognition.face_encodings(rgb_small, locs_small)

        locs, encs = [], []
        for (top, right, bottom, left), enc in zip(locs_small, encs_small):
            # scale to full frame
            top = int(top/RESIZE_SCALE); right = int(right/RESIZE_SCALE)
            bottom = int(bottom/RESIZE_SCALE); left = int(left/RESIZE_SCALE)
            locs.append((top, right, bottom, left))
            encs.append(enc)

        # liveness cập nhật theo frame đầy đủ
        blink_state = blink_detection(frame, blink_state)

        for (top, right, bottom, left), enc in zip(locs, encs):
            distances = face_recognition.face_distance(known_face_encodings, enc)
            if len(distances) == 0: continue
            idx = int(np.argmin(distances))
            match = distances[idx] <= TOLERANCE
            nm = known_face_names[idx] if match else "Unknown"

            # điều kiện mở cửa: đã có ít nhất 1 blink hoàn chỉnh
            if match and blink_state['blinks'] >= 1:
                name = nm

            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, nm, (left+6, bottom-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)

        cv2.putText(frame, f"Blinks:{blink_state['blinks']}  Press 'q' to quit, 'b' to (re)start",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return "QUIT"

    return name

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera 0.")

    print("[INFO] Ready. Press 'b' to start verification window (or button on Pi). Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.putText(frame, "Press 'b' to start verify (10s). 'q' to quit.",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        trigger = (key == ord('b')) or (getattr(HW, "RPI", False) and HW.button_pressed())
        if key == ord('q'):
            break

        if trigger:
            who = verify_window(cap, seconds=10)
            if who == "QUIT":
                break
            if who != "Unknown":
                print(f"[INFO] Welcome Home {who}")
                HW.door_on(5)
            else:
                print("[INFO] Unknown visitor. Capturing & sending notice.")
                takephoto(frame)
                try:
                    send_notice(GUEST_IMAGE)
                except Exception as e:
                    print(f"[WARN] send_notice failed: {e}")

    cap.release()
    cv2.destroyAllWindows()
    HW.cleanup()
    print("[INFO] Exit.")

if __name__ == "__main__":
    main()
