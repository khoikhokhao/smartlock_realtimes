import face_recognition
import cv2
import os
import pickle
import time
import numpy as np
# Định nghĩa đường dẫn file encodings
encodings_file = os.path.join(os.path.dirname(__file__), "face_encodings.pkl")

# Hàm để lưu encodings
def save_encodings_to_file(encodings, names, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump((encodings, names), f)

# Tải encodings hiện có
known_face_encodings, known_face_names = [], []
if os.path.exists(encodings_file):
    with open(encodings_file, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    print(f"Loaded {len(known_face_names)} existing faces.")

# Đảm bảo thư mục cha tồn tại
encodings_dir = os.path.dirname(encodings_file)
if not os.path.exists(encodings_dir):
    os.makedirs(encodings_dir)
    print(f"Created directory: {encodings_dir}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open USB camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("USB Camera started.")
time.sleep(2)

print("\n--- Bắt đầu thêm khuôn mặt mới ---")
print("Đảm bảo khuôn mặt ở giữa khung hình và rõ ràng.")
print("Nhấn 'c' để bắt đầu chụp (sẽ chụp 10 ảnh), 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error capturing frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    if len(face_locations) == 1:
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, "Press 'c' to CAPTURE (10 pics)", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Please show one clear face.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Enroll Face", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        if len(face_locations) == 1:
            name = input("Enter the name for this person: ")
            if not name:
                print("Name cannot be empty. Aborting capture.")
                continue

            print(f"Starting capture for {name}. Please hold still...")
            captured_encodings = []

            for i in range(10):
                print(f"Capturing image {i+1}/10...")
                time.sleep(0.5)

                ret_capture, capture_frame = cap.read()
                if not ret_capture:
                    print(f"Error capturing frame {i+1}. Skipping.")
                    continue

                rgb_capture_frame = cv2.cvtColor(capture_frame, cv2.COLOR_BGR2RGB)
                capture_face_locations = face_recognition.face_locations(rgb_capture_frame, model="hog")

                if len(capture_face_locations) == 1:
                    face_encoding = face_recognition.face_encodings(rgb_capture_frame, capture_face_locations)[0]
                    captured_encodings.append(face_encoding)

                    top, right, bottom, left = capture_face_locations[0]
                    cv2.rectangle(capture_frame, (left, top), (right, bottom), (0, 255, 255), 2)
                    cv2.putText(capture_frame, f"Captured {i+1}/10", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imshow("Enroll Face", capture_frame)
                    cv2.waitKey(200)
                else:
                    print(f"No single face detected in image {i+1}. Skipping.")
                    cv2.putText(capture_frame, "No face detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imshow("Enroll Face", capture_frame)
                    cv2.waitKey(500)

            if len(captured_encodings) > 0:
                # Lưu tất cả 10 encoding riêng biệt thay vì average
                for encoding in captured_encodings:
                    known_face_encodings.append(encoding)
                    known_face_names.append(name)

                print(f"Added {len(captured_encodings)} encodings for {name}.")
                save_encodings_to_file(known_face_encodings, known_face_names, encodings_file)
                print(f"Encodings saved. Total encodings: {len(known_face_names)}")
            else:
                print("No valid faces captured for enrollment. Try again.")
        else:
            print("Please ensure only one clear face is in the frame before pressing 'c'.")

    elif key == ord('q'):
        print("Exiting enrollment.")
        break

print("Cleaning up resources...")
cap.release()
cv2.destroyAllWindows()
print("Enrollment complete.")
