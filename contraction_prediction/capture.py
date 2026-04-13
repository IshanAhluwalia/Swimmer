"""
Camera Video Capture Script
- Opens a live preview from the connected camera
- Press S to start recording, press S again to stop
- Press Q or ESC to quit
- Records at a constant TARGET_FPS
- Saves per-frame timestamps to a matching CSV file in real time
"""

import csv
import cv2
import os
import time
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "skin_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_FPS = 30.0
FRAME_INTERVAL = 1.0 / TARGET_FPS

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open camera. Try changing the camera index in capture.py.")
    exit(1)

cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera opened ({width}x{height} @ {TARGET_FPS:.1f} fps).")
print("Press S to start/stop recording, Q or ESC to quit.")

recording = False
writer = None
csv_file = None
csv_writer = None
video_path = None
frame_count = 0
next_frame_time = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from camera.")
        break

    now = time.monotonic()

    if recording and now >= next_frame_time:
        writer.write(frame)
        csv_writer.writerow([frame_count, datetime.now().isoformat(timespec='milliseconds')])
        csv_file.flush()
        frame_count += 1
        # Advance deadline; clamp so we don't spiral trying to catch up
        next_frame_time = max(next_frame_time + FRAME_INTERVAL, now)

    display = frame.copy()

    if recording:
        elapsed = frame_count / TARGET_FPS
        mins, secs = divmod(int(elapsed), 60)
        cv2.putText(display, f"  REC  {mins:02d}:{secs:02d}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.circle(display, (15, 28), 10, (0, 0, 255), -1)
        cv2.putText(display, "S: stop  |  Q/ESC: quit", (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(display, "S: start recording  |  Q/ESC: quit", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Camera Preview", display)

    key = cv2.waitKey(1) & 0xFF

    if key in (ord('s'), ord('S')):
        if not recording:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(OUTPUT_DIR, f"video_{stamp}.mp4")
            csv_path   = os.path.join(OUTPUT_DIR, f"data_{stamp}.csv")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, TARGET_FPS, (width, height))
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['frame', 'timestamp'])
            csv_file.flush()
            frame_count = 0
            next_frame_time = time.monotonic()
            recording = True
            print(f"Recording started: {video_path}")
            print(f"Data file:         {csv_path}")
        else:
            recording = False
            writer.release()
            writer = None
            csv_file.close()
            csv_file = None
            csv_writer = None
            print(f"Recording stopped. Saved to: {video_path}")

    elif key in (ord('q'), ord('Q'), 27):  # 27 = ESC
        break

if recording and writer:
    writer.release()
    if csv_file:
        csv_file.close()
    print(f"Recording saved to: {video_path}")

cap.release()
cv2.destroyAllWindows()
print("Done.")
