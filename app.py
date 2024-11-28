from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
from ultralytics import YOLO
from collections import defaultdict

app = Flask(__name__)

# Load the YOLO model
model = YOLO('yolo11m.pt')
class_list = model.names
video_file = 'apel.mp4'

cap = None
# Koordinat garis diagonal
x1_line, y1_line = 21, 238
x2_line, y2_line = 763, 352

# Hitung kemiringan (m) dan y-intercept (c) dari garis
m_line = (y2_line - y1_line) / (x2_line - x1_line)
c_line = y1_line - m_line * x1_line

# Dictionary to store object counts by class
class_counts = defaultdict(int)
crossed_ids = set()

# Variabel global untuk mengontrol status deteksi
detection_running = False

# Fungsi untuk membuka ulang video
def reset_video():
    global cap
    cap.release()  # Tutup video saat ini
    cap = cv2.VideoCapture(video_file)  # Buka kembali video

# Fungsi untuk memulai deteksi
def start_detection():
    # global cap
    # if cap is None:
    #     cap = cv2.VideoCapture(video_file)
    global cap, detection_running, class_counts
    class_counts = defaultdict(int)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(video_file)
    detection_running = True

def stop_detection():
    global cap, detection_running
    detection_running = False  # Menghentikan loop di generate_frames()
    if cap is not None:
        cap.release()  # Lepaskan video capture


@app.route('/')
def index():
    return render_template('home.html') 


@app.route('/detection', methods=['GET', 'POST'])
def detection():
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'start':
            start_detection()  # Mulai deteksi saat tombol ditekan
        elif action == 'stop':
            stop_detection()
    return render_template('detection.html', class_counts = class_counts['apple'])

def generate_frames():
    global cap
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if it ends
            continue

        results = model.track(frame, persist=True, classes=[47])

        if results[0].boxes.data is not None:
            boxes = results[0].boxes.xyxy.cpu()
            # Pastikan id tersedia sebelum akses
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
            else:
                track_ids = [None] * len(boxes)  # Jika id tidak ada, isi dengan None
            class_indices = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu()

            cv2.line(frame, (x1_line, y1_line), (x2_line, y2_line), (0, 0, 255), 3)

            for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                class_name = class_list[class_idx]

                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                y_on_line = m_line * cx + c_line
                if cy > y_on_line and track_id not in crossed_ids:
                    crossed_ids.add(track_id)
                    class_counts[class_name] += 1

            # Display counting data on the frame
            y_offset = 30
            for class_name, count in class_counts.items():
                cv2.putText(frame, f"{class_name}: {count}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y_offset += 30

        # Encode the frame and return it as byte stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    if cap is None:
        return redirect(url_for('detection'))  # Jika video belum dimulai, redirect ke halaman utama
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
