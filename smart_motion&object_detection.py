import cv2
import time
import os
import math
from datetime import datetime
from ultralytics import YOLO

model = YOLO("yolov5s.pt")


CATEGORY_MAP = {
    "stationary": ["book", "scissors"],
    "electrical": ["tvmonitor", "cell phone"],
    "plastic": ["bottle", "cup"]
}


SAVE_DIR = "detected_objects"
os.makedirs(SAVE_DIR, exist_ok=True)


object_history = {}
cooldown_seconds = 5
last_snapshot_time = 0
movement_threshold = 20  

def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def detect_and_group(frame):
    results = model.predict(source=frame, save=False, verbose=False)[0]
    current_objects = {}
    grouped = {"stationary": [], "electrical": [], "plastic": []}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0]
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        current_objects[label] = center

        
        for group, items in CATEGORY_MAP.items():
            if label in items:
                grouped[group].append(label)

    return current_objects, grouped

def should_save_snapshot(current_objects):
    global object_history, last_snapshot_time

    now = time.time()
    if now - last_snapshot_time < cooldown_seconds:
        return False, []

    changed_objects = []

    for label, center in current_objects.items():
        if label not in object_history:
            changed_objects.append(label)
        else:
            dist = calculate_distance(object_history[label], center)
            if dist > movement_threshold:
                changed_objects.append(label)

    if changed_objects:
        last_snapshot_time = now
        return True, changed_objects

    return False, []

def save_snapshot(frame, labels):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "_".join(sorted(labels))
    filename = f"{SAVE_DIR}/Detected_{tag}_{timestamp}.jpeg"
    cv2.imwrite(filename, frame)
    print(f"📸 Snapshot saved: {filename}")

def main():
    global object_history
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Could not access webcam.")
        return

    print("🎥 Object Detection + Category Grouping Started (press 'q' to quit)")
    time.sleep(2)

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            continue

        frame = cv2.resize(frame, (640, 480))
        detected_objects, grouped = detect_and_group(frame)
        should_save, changed = should_save_snapshot(detected_objects)

        if should_save:
            print(f"\n⚠️ Object Change Detected: {', '.join(changed)}")

            for group, items in grouped.items():
                if items:
                    print(f"🧠 {group.capitalize()}: {', '.join(set(items))}")
            save_snapshot(frame, detected_objects.keys())

        object_history = detected_objects

        cv2.imshow("Grouped Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
