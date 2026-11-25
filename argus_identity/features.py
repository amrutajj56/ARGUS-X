import cv2
import numpy as np

class AppearanceExtractor:
    # Handcrafted CPU baseline; not a trained deep Re-ID model.
    def __init__(self, bins=(16, 8, 8)):
        self.bins = tuple(int(x) for x in bins)

    def extract(self, frame, bbox):
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or len(bbox) != 4:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            return None
        crop = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        parts = []
        for region in (hsv[:64, :], hsv[64:, :]):
            hist = cv2.calcHist([region], [0,1,2], None, list(self.bins), [0,180,0,256,0,256])
            hist = cv2.normalize(hist, None, alpha=1.0, norm_type=cv2.NORM_L1).flatten()
            parts.append(hist.astype(np.float32))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        texture = np.array([gray.mean()/255.0, gray.std()/128.0,
                            np.mean(np.abs(gx))/255.0, np.mean(np.abs(gy))/255.0],
                           dtype=np.float32)
        descriptor = np.concatenate(parts + [texture]).astype(np.float32)
        norm = float(np.linalg.norm(descriptor))
        return descriptor / norm if norm > 1e-12 else descriptor

def cosine_similarity(a, b):
    if a is None or b is None or a.shape != b.shape:
        return -1.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return -1.0
    return float(np.clip(np.dot(a,b)/denom, -1.0, 1.0))
