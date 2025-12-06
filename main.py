import os
import cv2
path = r"C:\Users\sande\Downloads\archive (7)\SOCOFing\Altered\Altered-Easy\1__M_Left_index_finger_CR.BMP"
if not os.path.exists(path):
    print(f"❌ Error: File not found at: {path}")
    print("Check specific spelling, especially underscores and file extension case (.BMP vs .bmp)")
else:
    sample = cv2.imread(path)
    
    # Double check if OpenCV loaded it (sometimes path exists but image is corrupt)
    if sample is None:
        print("❌ Error: Path exists, but OpenCV could not read the image.")
    else:
        print("✅ Image loaded successfully!")
        sample = cv2.resize(sample, None, fx=2.5, fy=2.5)
        cv2.imshow("Sample", sample)
        cv2.waitKey(0)
        cv2.destroyAllWindows()