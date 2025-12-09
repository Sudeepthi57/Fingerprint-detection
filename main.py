import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime

print("="*80)
print("  AI-BASED FINGERPRINT ALTERATION DETECTION SYSTEM")
print("  Tools: Python, OpenCV, TensorFlow, Keras")
print("  Dataset: SOCOFing Fingerprint Database")
print("="*80 + "\n")

# Load trained CNN model
print(" Loading trained CNN model...")
try:
    model = load_model('best_fingerprint_model.keras')
    print(" CNN Model loaded successfully!")
    print(f"  Model accuracy: 84.50% (from evaluation)")
    print(f"   Trained on: 1000 fingerprints (500 Real + 500 Altered)\n")
except:
    try:
        model = load_model('fingerprint_model_final.keras')
        print(" CNN Model loaded successfully!\n")
    except:
        print(" Model not found! Please run training script first.\n")
        exit()


def detect_fingerprint_alteration(image_path, show_visualization=True):
    """
    Detects if a fingerprint is REAL or ALTERED using CNN
    
    This function:
    1. Preprocesses the input image (resize, normalize)
    2. Uses CNN to extract features and classify
    3. Returns REAL or ALTERED with confidence score
    
    Args:
        image_path: Path to fingerprint image
        show_visualization: Whether to create result visualization
    
    Returns:
        result: "REAL" or "ALTERED"
        confidence: Confidence percentage (0-100)
    """
    
    print("="*80)
    print("FINGERPRINT ALTERATION DETECTION")
    print("="*80 + "\n")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f" ERROR: File not found")
        print(f"   Path: {image_path}\n")
        return None, None
    
    print(f" Input Image: {os.path.basename(image_path)}")
    print("-"*80)
    
    # STEP 1: Load and preprocess image
    print("\n STEP 1: Image Preprocessing")
    print("   • Reading image in grayscale...")
    img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img_original is None:
        print(" ERROR: Cannot read image. Check file format.\n")
        return None, None
    
    print(f"    Image loaded: {img_original.shape[0]}x{img_original.shape[1]} pixels")
    
    print("   • Resizing to 96x96 pixels (CNN input size)...")
    img_resized = cv2.resize(img_original, (96, 96))
    print("    Resized successfully")
    
    print("   • Normalizing pixel values (0-255 → 0-1)...")
    img_normalized = img_resized.reshape(1, 96, 96, 1) / 255.0
    print("    Normalization complete")
    
    # STEP 2: CNN Classification
    print("\n STEP 2: CNN-Based Classification")
    print("   • Passing image through convolutional layers...")
    print("     - Layer 1: Detecting basic features (edges, lines)")
    print("     - Layer 2: Detecting ridge patterns")
    print("     - Layer 3: Detecting complex patterns")
    print("   • Processing through dense layers...")
    
    # Get prediction from CNN
    prediction = model.predict(img_normalized, verbose=0)
    raw_score = prediction[0][0]
    
    print(f"    Classification complete")
    print(f"   • Raw CNN output: {raw_score:.4f}")
    
    # STEP 3: Interpret results
    print("\n STEP 3: Result Interpretation")
    print(f"   • Classification threshold: 0.5")
    print(f"   • Score > 0.5 → ALTERED (fake/modified)")
    print(f"   • Score < 0.5 → REAL (authentic)")
    
    if raw_score > 0.5:
        result = "ALTERED"
        confidence = raw_score * 100
        status = "🚨 FAKE/MODIFIED FINGERPRINT DETECTED"
        color = "red"
        interpretation = "Signs of manipulation or forgery detected"
    else:
        result = "REAL"
        confidence = (1 - raw_score) * 100
        status = " AUTHENTIC FINGERPRINT VERIFIED"
        color = "green"
        interpretation = "Natural fingerprint patterns confirmed"
    
    # Display results
    print("\n" + "="*80)
    print(f"  {status}")
    print("="*80)
    print(f"Classification:  {result}")
    print(f"Confidence:      {confidence:.2f}%")
    print(f"Interpretation:  {interpretation}")
    
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
    print("="*80 + "\n")
    
    # Create visualization
    if show_visualization:
        create_detection_visualization(
            img_original, img_resized, result, confidence, 
            os.path.basename(image_path), color
        )
    
    return result, confidence


def create_detection_visualization(img_original, img_processed, result, 
                                   confidence, filename, color):
    """
    Creates a visual representation of the detection result
    """
    
    fig = plt.figure(figsize=(14, 6))
    
    # Original image
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img_original, cmap='gray')
    ax1.set_title('Original Fingerprint Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Processed image with result
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(img_processed, cmap='gray')
    
    if result == "REAL":
        title_text = f"✅ REAL FINGERPRINT\nAuthentic/Genuine\nConfidence: {confidence:.2f}%"
        title_color = 'green'
        border_color = 'lightgreen'
    else:
        title_text = f"🚨 ALTERED FINGERPRINT\nFake/Modified\nConfidence: {confidence:.2f}%"
        title_color = 'red'
        border_color = 'lightcoral'
    
    ax2.set_title(title_text, fontsize=12, fontweight='bold', 
                  color=title_color, pad=20)
    ax2.axis('off')
    
    # Add colored border
    for spine in ax2.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(5)
    
    # Overall title
    fig.suptitle(f'CNN-Based Fingerprint Alteration Detection\nFile: {filename}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"detection_result_{timestamp}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📷 Visualization saved: {output_filename}\n")


def batch_detection(folder_path):
    """
    Detect multiple fingerprints in a folder
    """
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}\n")
        return
    
    # Find all image files
    extensions = ['.BMP', '.bmp', '.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG']
    files = [f for f in os.listdir(folder_path) if any(f.endswith(ext) for ext in extensions)]
    
    if not files:
        print(f"❌ No image files found in folder\n")
        return
    
    print(f"\n🔍 Batch Detection: Processing {len(files)} fingerprints...\n")
    print("="*80 + "\n")
    
    results = []
    real_count = 0
    altered_count = 0
    
    for idx, filename in enumerate(files, 1):
        filepath = os.path.join(folder_path, filename)
        print(f"[{idx}/{len(files)}] Processing: {filename}")
        
        result, conf = detect_fingerprint_alteration(filepath, show_visualization=False)
        
        if result:
            results.append({
                'filename': filename,
                'result': result,
                'confidence': conf
            })
            
            if result == "REAL":
                real_count += 1
            else:
                altered_count += 1
        
        print()
    
    # Summary report
    print("\n" + "="*80)
    print("BATCH DETECTION SUMMARY REPORT")
    print("="*80)
    print(f"Total Images Processed:     {len(results)}")
    print(f"✅ Real Fingerprints:          {real_count}")
    print(f"🚨 Altered Fingerprints:       {altered_count}")
    print(f"Average Confidence:         {np.mean([r['confidence'] for r in results]):.2f}%")
    print("="*80 + "\n")
    
    # Save report
    report_name = f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_name, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FINGERPRINT ALTERATION DETECTION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Folder: {folder_path}\n")
        f.write(f"Total Images: {len(results)}\n")
        f.write("="*80 + "\n\n")
        
        for r in results:
            f.write(f"File: {r['filename']}\n")
            f.write(f"  Result: {r['result']}\n")
            f.write(f"  Confidence: {r['confidence']:.2f}%\n\n")
    
    print(f"📄 Report saved: {report_name}\n")


# ============================================================================
#                           MAIN PROGRAM
# ============================================================================

def main():
    """Main detection program"""
    
    print("\n" + "="*80)
    print("                         DETECTION MODE")
    print("="*80)
    print("1. Detect Single Fingerprint")
    print("2. Detect Multiple Fingerprints (Batch Mode)")
    print("3. Quick Test with Dataset Sample")
    print("4. Exit")
    print("="*80 + "\n")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == '1':
        print("\n" + "-"*80)
        image_path = input("Enter fingerprint image path: ").strip().strip('"')
        
        if image_path:
            result, confidence = detect_fingerprint_alteration(image_path)
            
            if result:
                print("\n💡 WHAT THIS MEANS:")
                if result == "REAL":
                    print("   → This fingerprint is AUTHENTIC/GENUINE")
                    print("   → No signs of manipulation detected")
                    print("   → Safe to use for biometric authentication")
                else:
                    print("   → This fingerprint is ALTERED/FAKE")
                    print("   → Signs of manipulation or forgery detected")
                    print("   → Should NOT be used for authentication")
                print()
    
    elif choice == '2':
        print("\n" + "-"*80)
        folder_path = input("Enter folder path: ").strip().strip('"')
        
        if folder_path:
            batch_detection(folder_path)
    
    elif choice == '3':
        print("\n" + "-"*80)
        print("Quick Test: Testing with sample images from dataset\n")
        
        # Test real fingerprint
        real_sample = r"C:\Users\sande\Downloads\archive (7)\SOCOFing\Real\1_M_Left_index_finger.BMP"
        if os.path.exists(real_sample):
            print("TEST 1: Authentic Fingerprint")
            detect_fingerprint_alteration(real_sample)
        
        # Test altered fingerprint
        altered_sample = r"C:\Users\sande\Downloads\archive (7)\SOCOFing\Altered\Altered-Hard\1_M_Left_index_finger_Obl.BMP"
        if os.path.exists(altered_sample):
            print("\nTEST 2: Altered Fingerprint")
            detect_fingerprint_alteration(altered_sample)
    
    elif choice == '4':
        print("\n Thank you for using the detection system!\n")
    
    else:
        print("\n❌ Invalid choice\n")


if __name__ == "__main__":
    # Show project objectives
    print("\n📋 PROJECT OBJECTIVES:")
    print("-"*80)
    print(" Built a deep learning-based biometric authentication system")
    print(" Used SOCOFing fingerprint dataset (1000 images)")
    print(" Implemented image preprocessing (resize, normalize)")
    print("Built CNN-based classification (3 Conv layers + Dense layers)")
    print(" Achieved 84.50% accuracy on test data")
    print(" Evaluated using confusion matrix and accuracy metrics")
    print(" Can test predictions on unseen/new fingerprint images")
    print("-"*80 + "\n")
    
    main()