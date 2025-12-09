import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from datetime import datetime

# ============================================================================
#                    LOAD TRAINED CNN MODEL
# ============================================================================

try:
   
    model = load_model(r'C:\Users\sande\OneDrive\drive\OneDrive\Desktop\Fingerprint-detection-main (2)\Fingerprint-detection-main\best_fingerprint_model.keras')
    print("\n✅ CNN Model loaded successfully!")
    print(f"   Model Accuracy: 84.50%")
    print(f"   Trained on: 1000 fingerprints (500 Real + 500 Altered)\n")
except:
    try:
        model = load_model(r'C:\Users\sande\OneDrive\drive\OneDrive\Desktop\Fingerprint-detection-main (2)\Fingerprint-detection-main\fingerprint_model_final.keras')
        print("\n✅ CNN Model loaded successfully!\n")
    except:
        print("\n❌ Model not found! Please run training script first.\n")
        print("Looking for models in:")
        print(r" C:\Users\sande\OneDrive\drive\OneDrive\Desktop\Fingerprint-detection-main (2)\Fingerprint-detection-main\fingerprint_model.h5")
        exit()


# ============================================================================
#                    DETECTION FUNCTION
# ============================================================================
def detect_fingerprint_alteration(image_path):
    
    # ===== STEP 1: VALIDATE FILE =====
    if not os.path.exists(image_path):
        print(f" ERROR: File not found")
        print(f"   Path: {image_path}\n")
        return None, None
    
    print(f" Input Image: {os.path.basename(image_path)}")
    print("-"*80)
    
    # ===== STEP 2: LOAD IMAGE =====
    img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img_original is None:
        print(" ERROR: Cannot read image. Check file format.\n")
        return None, None
    
    print(f"   ✓ Image loaded: {img_original.shape[0]}x{img_original.shape[1]} pixels")
    
    # ===== STEP 3: PREPROCESS IMAGE =====
 
    img_resized = cv2.resize(img_original, (96, 96))
    
    img_normalized = img_resized.reshape(1, 96, 96, 1) / 255.0
    
    # ===== STEP 4: CNN PREDICTION =====
   
    prediction = model.predict(img_normalized, verbose=0)
    raw_score = prediction[0][0]
    
    print(f"   ✓ Raw CNN output: {raw_score:.4f}")
    
    # ===== STEP 5: INTERPRET RESULTS =====
    
    if raw_score > 0.5:
        result = "ALTERED"
        confidence = raw_score * 100
        status = " FAKE/MODIFIED FINGERPRINT DETECTED"
        interpretation = "Signs of manipulation or forgery detected"
    else:
        result = "REAL"
        confidence = (1 - raw_score) * 100
        status = " AUTHENTIC FINGERPRINT VERIFIED"
        interpretation = "Natural fingerprint patterns confirmed"
    
    # ===== STEP 6: DISPLAY RESULTS =====
   
    print(f"  {status}")
    print("="*80)
    print(f"Classification:  {result}")
    print(f"Confidence:      {confidence:.2f}%")
    print(f"Interpretation:  {interpretation}")
    
    # Confidence level
    if confidence >= 90:
        certainty = "VERY HIGH - Highly reliable result"
    elif confidence >= 75:
        certainty = "HIGH - Reliable result"
    elif confidence >= 60:
        certainty = "MODERATE - Acceptable confidence"
    else:
        certainty = "LOW - Manual verification recommended"
    
    print(f"Certainty:       {certainty}")
    print(f"Timestamp:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    
    # ===== STEP 7: EXPLAIN WHAT THIS MEANS =====
    print("💡 WHAT THIS MEANS:")
    if result == "REAL":
        print("   → This fingerprint is AUTHENTIC/GENUINE")
        print("   → Safe to use for biometric authentication")
    else:
        print("   → This fingerprint is ALTERED/FAKE")
        print("   → Should NOT be used for authentication")
    print()
    
    return result, confidence


# ============================================================================
#                           MAIN PROGRAM
# ============================================================================
if __name__ == "__main__":
    # Show project info
    
    # Get image path from use
    print("SINGLE FINGERPRINT DETECTION")
    
    image_path = input(" Enter fingerprint image path: ").strip().strip('"')
    
    if image_path:
        # Run detection
        result, confidence = detect_fingerprint_alteration(image_path)
        
        if result:
            print(" Detection completed successfully!")
        else:
            print(" Detection failed. Check the error messages above.")
    else:
        print("\n No image path provided!\n")