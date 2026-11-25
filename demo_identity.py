import cv2
import numpy as np
from argus_identity import IdentityEngine

def main():
    engine = IdentityEngine()
    frame = np.zeros((360,640,3), dtype=np.uint8)
    def person():
        img = frame.copy()
        cv2.rectangle(img,(220,50),(330,340),(40,90,210),-1)
        return img
    print("--- ARGUS-X Synthetic Recovery Demo ---")
    for n in range(3):
        r = engine.update(person(), [{"track_id":1,"bbox":[220,50,330,340],"confidence":0.95,"class_id":0}])[0]
        print(f"Frame={n:02d} Tracker={r['track_id']} Global={r['global_id']} Status={r['identity_status']}")
    for _ in range(3,12):
        engine.update(np.zeros_like(frame), [])
    r = engine.update(person(), [{"track_id":9,"bbox":[220,50,330,340],"confidence":0.95,"class_id":0}])[0]
    print(f"Frame=12 Tracker={r['track_id']} Global={r['global_id']} Status={r['identity_status']} Confidence={r['identity_confidence']}")
    print("Diagnostics:", engine.diagnostics())

if __name__ == "__main__":
    main()
