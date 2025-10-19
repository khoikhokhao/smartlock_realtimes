# Smart Door Lock (Webcam Version)

> Bản chạy bằng **webcam** (PC hoặc Raspberry Pi). Không bắt buộc có GPIO.  
> Hệ thống nhận diện khuôn mặt + kiểm tra nháy mắt (liveness). Người lạ sẽ bị chụp ảnh và có thể gửi cảnh báo qua Email/Discord Webhook.

## ✨ Tính năng
- **Đăng ký khuôn mặt (Enroll)**: chụp 10 ảnh/encoding cho mỗi người để tăng độ ổn định.
- **Nhận diện real-time bằng webcam**: so khớp embedding 128D (dlib ResNet) với dữ liệu đã đăng ký.
- **Liveness check (nháy mắt)**: dùng landmark 68 điểm để tính EAR → xác minh người thật.
- **Cảnh báo**:
  - **Email** (SMTP Outlook/Yahoo hoặc Gmail + App Password).
  - **Discord Webhook** kèm ảnh người lạ.
- **Đánh giá mô hình**: script vẽ ROC, tính AUC, EER, Precision/Recall/F1.

## 🧱 Kiến trúc & mô hình
- **Phát hiện khuôn mặt**: `face_recognition.face_locations` (HOG).
- **Mã hoá khuôn mặt**: ResNet-34 pretrained của dlib → vector **128 chiều**.
- **So khớp**: khoảng cách Euclidean; ngưỡng **tolerance** mặc định `0.28` (an toàn).
- **Liveness**: `shape_predictor_68_face_landmarks.dat` (dlib) → 68 landmarks → **EAR** để bắt nháy mắt.

## 📂 Cấu trúc dự án
```
model/
├─ enroll_faces_10encodings.py
├─ Main.py
├─ delete_faces.py
├─ evaluate_recognition.py
├─ shape_predictor_68_face_landmarks.dat
├─ face_encodings.pkl
├─ guest_image.jpeg
├─ requirements.txt
└─ README.md
```

## 🛠 Cài đặt

### 1) Tạo & kích hoạt môi trường ảo
**Windows PowerShell**
```powershell
cd C:\Users\Admin\Desktop\model
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2) Cài thư viện
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3) Tải model landmarks
Đảm bảo file `shape_predictor_68_face_landmarks.dat` có trong thư mục dự án.

## ▶️ Chạy thử (Webcam)
```bash
python enroll_faces_10encodings.py
python Main.py
```

## 🔔 Cấu hình cảnh báo (tuỳ chọn)
### Email
```powershell
$env:SENDER_EMAIL="your_email@example.com"
$env:SENDER_PASSWORD="your_password"
$env:RECEIVER_EMAIL="destination@example.com"
```
### Discord Webhook
```powershell
$env:DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
```

## 📏 Đánh giá mô hình
```bash
python evaluate_recognition.py
```

## ⚙️ Tham số chính
- `TOLERANCE = 0.28`
- `verify_window(..., seconds=10)`
- `HW.door_on(5)`
