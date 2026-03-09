import cv2
import numpy as np
import torch


def preprocess(frame):
    resized    = cv2.resize(frame, (640, 640))
    rgb        = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    norm       = rgb / 255.0
    transposed = np.transpose(norm, (2, 0, 1))
    batch      = np.expand_dims(transposed, axis=0)
    return batch.astype(np.float32)

def extract_detections(output, confidence=0.5, class_id=0):
    dets     = output[0][0]
    mask     = (dets[:, 4] > confidence) & (dets[:, 5] == class_id)
    filtered = dets[mask]
    return torch.tensor(filtered)