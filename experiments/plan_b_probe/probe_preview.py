"""Scratch: dump a downscaled preview of an arbitrary F: image so it can be looked at."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import common as C

if __name__ == "__main__":
    src, name = sys.argv[1], sys.argv[2]
    side = int(sys.argv[3]) if len(sys.argv) > 3 else 900
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    print(src, "->", None if img is None else img.shape)
    if img is not None:
        print(C.save_preview(img, name, side))
