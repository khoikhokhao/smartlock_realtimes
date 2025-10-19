import os
import time
import cv2
import numpy as np
import face_recognition
import pickle
import dlib

# GPIO shim (giống Main.py)
try:
    import RPi.GPIO as _GPIO
    class GPIOWrap:
        GPIO = _GPIO
        RPI = True
        def __init__(self):
            self.GPIO.setmode(self.GPIO.BCM)
            self.GPIO.setwarnings(False)
            self.GPIO.setup(17, self.GPIO.OUT, initial=self.GPIO.LOW)
            self.GPIO.setup(27, self.GPIO.IN, pull_up_down=self.GPIO.PUD_DOWN)
        def cleanup(self): self.GPIO.cleanup()
        def door_on(self, sec=5):
            self.GPIO.output(17, self.GPIO.HIGH); time.sleep(sec); self.GPIO.output(17, self.GPIO.LOW)
        def button_pressed(self): return self.GPIO.input(27) == self.GPIO.HIGH
    HW = GPIOWrap()
except Exception:
    class _Dummy:
        RPI=False
        def cleanup(self): pass
        def door_on(self, sec=5): print(f"[SIM] Door open {sec}s")
        def button_pressed(self): return False
    HW = _Dummy()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODINGS_FILE = os.environ.get("ENCODINGS_FILE", os.path.join(BASE_DIR, "face_encodings.pkl"))
PREDICTOR_PATH = os.environ.get("DLIB_LANDMARKS", os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat"))
TOLERANCE = 0.28

# --- dlib landmarks ---
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(f"Missing landmarks model: {PREDICTOR_PATH}")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def ear(eye):
    A = np.linalg.norm(eye[1] - eye[5]); B = np.linalg.norm(eye[2] - eye[4]); C = np.linalg.norm(eye[0] - eye[3])
    return (A+B)/(2.0*C)

def update_blink(frame, state):
    TH=0.25; MINC=3
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    if len(faces)==0:
        state['consec']=0; return state
    for f in faces:
        lm = predictor(gray, f)
        L = np.array([(lm.part(i).x, lm.part(i).y) for i in range(36,42)])
        R = np.array([(lm.part(i).x, lm.part(i).y) for i in range(42,48)])
        if (ear(L)+ear(R))/2.0 < TH:
            state['consec']+=1
        else:
            if state['consec']>=MINC: state['blinks']+=1
            state['consec']=0
    return state

# --- encodings ---
if not os.path.exists(ENCODINGS_FILE):
    raise FileNotFoundError(f"Encodings file not found: {ENCODINGS_FILE}")
with open(ENCODINGS_FILE, 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)
print(f"[INFO] Loaded {len(known_face_names)} encodings.")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera 0.")
    print("[INFO] Ready. Press 'b' to start 10s verify (or Pi button). 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok: break
        cv2.putText(frame, "Press 'b' to verify (10s). 'q' to quit.",
                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        trigger = (key == ord('b')) or (getattr(HW, "RPI", False) and HW.button_pressed())
        if key == ord('q'): break

        if trigger:
            name = "Unknown"
            end_t = time.time()+10
            state = {'consec':0,'blinks':0}
            while time.time()<end_t:
                ok, frame = cap.read()
                if not ok: break
                # detect & encode on full frame (đơn giản)
                locs = face_recognition.face_locations(frame, model="hog")
                encs  = face_recognition.face_encodings(frame, locs)
                state = update_blink(frame, state)

                for (top, right, bottom, left), enc in zip(locs, encs):
                    dists = face_recognition.face_distance(known_face_encodings, enc)
                    if len(dists)==0: continue
                    idx = int(np.argmin(dists))
                    match = dists[idx] <= TOLERANCE
                    nm = known_face_names[idx] if match else "Unknown"
                    if match and state['blinks']>=1:
                        name = nm
                    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                    cv2.putText(frame, nm, (left+6, bottom-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)

                cv2.putText(frame, f"Blinks:{state['blinks']}  (q=quit)",
                            (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.imshow("Camera Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release(); cv2.destroyAllWindows(); HW.cleanup(); return

            if name != "Unknown":
                print(f"[INFO] Welcome Home {name}")
                HW.door_on(5)
            else:
                print("[INFO] Unknown. (No email in this simple script)")

    cap.release()
    cv2.destroyAllWindows()
    HW.cleanup()
    print("[INFO] Exit.")

if __name__ == "__main__":
    main()
