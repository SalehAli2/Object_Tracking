import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, fps

def extract_frames(output_dir, cap, fps, frames_per_second=1):
    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0
    saved_count = 0
    interval    = max(1, int(fps / frames_per_second))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f"Saved {saved_count} frames to '{output_dir}/'")

def frames_to_video(frames_dir, output_path, fps):
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for file_name in frame_files:
        frame = cv2.imread(os.path.join(frames_dir, file_name))
        writer.write(frame)
    writer.release()
    print(f"Video saved to '{output_path}'")