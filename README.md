📌 Project Overview

This project uses a trained Convolutional Neural Network (CNN) to classify fingerprints as:

✅ REAL (Authentic / Genuine)

🚨 ALTERED (Fake / Manipulated)

It uses the SOCOFing dataset, containing real and synthetically altered fingerprints.

🧠 Key Features

✔ Deep Learning model (CNN with 3 convolutional layers)
✔ Preprocessing: grayscale → resize → normalization
✔ Single-image detection
✔ Batch detection for folders
✔ Automatic visualization generation
✔ Streamlit web interface
✔ Model accuracy: 84.50%
✔ Supports BMP / JPG / PNG images

🗂 Tech Stack

Python

TensorFlow/Keras

OpenCV

NumPy

Matplotlib

Streamlit

📦 Dataset Used

SOCOFing – Sokoto Coventry Fingerprint Dataset

1000 images used

500 REAL + 500 ALTERED

Includes "Easy", "Medium" & "Hard" alterations

🚀 How to Run
1️⃣ Clone Repository
git clone https://github.com/yourusername/fingerprint-alteration-detector.git
cd fingerprint-alteration-detector

2️⃣ Install Requirements
pip install -r requirements.txt

3️⃣ Run the Python Detection Program (CLI)
python detect_fingerprint.py


You will see:

1. Detect Single Fingerprint
2. Detect Multiple Fingerprints
3. Quick Test with Dataset Sample
4. Exit

4️⃣ Run the Streamlit Web App
streamlit run app.py


The browser UI will appear:

Upload fingerprint

View prediction + confidence score

📁 Project Structure
📦 Fingerprint-Detection
│── best_fingerprint_model.keras     # Trained CNN model
│── detect_fingerprint.py            # CLI detection system
│── app.py                           # Streamlit interface
│── requirements.txt
│── README.md
│── /output_visualizations/          # Auto-saved result images
│── /reports/                        # Batch detection reports

🧪 Model Architecture

Your CNN contains:

3 Convolutional layers

MaxPooling layers

Dense classification layers

Sigmoid output (binary classification)

Input size: 96×96 grayscale

📸 Visualization Output

For each prediction, the program generates:

Original image

Processed image

Result title (REAL / ALTERED)

Confidence score

Colored border (green/red)

Saved automatically as:

detection_result_YYYYMMDD_HHMMSS.png

📊 Batch Detection

Provide any folder path, and the program will:

Scan for images

Predict every file

Save text report

Count REAL + ALTERED

Display average confidence

Example output:

Real Fingerprints: 312  
Altered Fingerprints: 188  
Average Confidence: 81.92%

🌐 Streamlit Web App

Simple web UI:

Upload → Detect → Display

Automatic preprocessing

Displays uploaded image

Shows classification result and confidence

Usage:

streamlit run app.py

🏆 Accuracy

Your reported model accuracy:

✔ 84.50% (Test Set)

Trained on:

1000 fingerprints

500 REAL

500 ALTERED

⚠️ Notes

Works best with 96×96 grayscale BMP images

Confidence below 60% → manual verification recommended

Not intended as a production biometric system

📘 Future Improvements

Improve CNN accuracy

Add noise removal / enhancement module

Add Ridge Feature Extraction (Gabor filters)

Deploy as a cloud API

Add mobile-friendly UI

❤️ Acknowledgements

SOCOFing Dataset

TensorFlow & OpenCV Community

Contributors & testers
