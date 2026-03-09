import cv2
import csv
import time
from collections import defaultdict
from src.detector import preprocess, extract_detections
from src.utils import read_video
from bytetracker import BYTETracker

def draw_tracks(frame, tracks, trails):
    h, w    = frame.shape[:2]
    x_scale = w / 640
    y_scale = h / 640
    for i in tracks:
        x1 = int(i[0] * x_scale)
        y1 = int(i[1] * y_scale)
        x2 = int(i[2] * x_scale)
        y2 = int(i[3] * y_scale)
        points = trails.get(int(i[4]), [])
        for j in range(1, len(points)):
            cv2.line(frame,
                     (int(points[j-1][0] * x_scale), int(points[j-1][1] * y_scale)),
                     (int(points[j][0]   * x_scale), int(points[j][1]   * y_scale)),
                     (0, 255, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{int(i[4])}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


def onnx_tracking_video(video_path, output_path, session,
                        confidence=0.5, class_id=0,
                        track_thresh=0.5, match_thresh=0.7, track_buffer=60):
    cap, fps = read_video(video_path)
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    tracker  = BYTETracker(track_thresh=track_thresh,
                           match_thresh=match_thresh,
                           track_buffer=track_buffer,
                           frame_rate=fps)
    prev_time = time.time()
    seen_ids = set()
    frame_id = 0
    csv_file = open("tracking_log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_id", "track_id", "x1", "y1", "x2", "y2", "score"])
    trails = defaultdict(list)
    while cap.isOpened():
        ret, frame = cap.read()
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
            seen_ids.add(int(track[4]))
            track_id = int(track[4])
            x1, y1, x2, y2 = track[0], track[1], track[2], track[3]
            cy = (y1 + y2) / 2
            cx = (x1 + x2) / 2
            trails[track_id].append((cx, cy))
            trails[track_id] = trails[track_id][-30:]
            csv_writer.writerow([
                          frame_id,
                          int(track[4]),
                          round(track[0], 2),
                          round(track[1], 2),
                          round(track[2], 2),
                          round(track[3], 2),
                          round(track[6], 2)
                          ])
        frame_id += 1
        if len(tracks) == 0:
            writer.write(frame)
            continue
        result = draw_tracks(frame, tracks, trails)
        writer.write(result)
    cap.release()
    writer.release()
    csv_file.close()
    print(f"Total unique people: {len(seen_ids)}")
    print(f"Video saved to '{output_path}'")


def build_tracker(fps, track_thresh=0.5, match_thresh=0.7, track_buffer=60):
    return BYTETracker(track_thresh=track_thresh, match_thresh=match_thresh,
                        track_buffer=track_buffer, frame_rate=fps)
