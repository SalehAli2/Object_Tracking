import cv2
import time
from collections import defaultdict
from src.detector import preprocess, extract_detections
from src.tracker import draw_tracks, build_tracker
import onnxruntime

MODEL_PATH = "models/yolo26n.onnx"
confidence = 0.4
class_id   = 0

def main():
    session = onnxruntime.InferenceSession(
    MODEL_PATH, 
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    cap     = cv2.VideoCapture(0)
    tracker = build_tracker(fps=30)
    trails  = defaultdict(list)
    prev_time = time.time()

    while cap.isOpened():
        ret, frame =cap.read()
        if not ret:
            break
        current_time = time.time()
        FPS = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS Counter:{int(FPS)}",(10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        inp    = preprocess(frame)
        output = session.run(None, {session.get_inputs()[0].name: inp})
        dets   = extract_detections(output, confidence, class_id)
        tracks = tracker.update(dets, None)
        for track in tracks:
            track_id = int(track[4])
            x1, y1, x2, y2 = track[0], track[1], track[2], track[3]
            cy = (y1 + y2) / 2
            cx = (x1 + x2) / 2
            trails[track_id].append((cx, cy))
            trails[track_id] = trails[track_id][-30:]
        if len(tracks) == 0:
            cv2.imshow("Tracking", frame)
        else:
            result = draw_tracks(frame, tracks, trails)
            cv2.imshow("Tracking", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()