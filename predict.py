import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import pickle

# --- KONFIGURASI ---
OUTPUT_DIR = "output"
IMG_SIZE = 64

def real_time_prediction():
    model_path_load = os.path.join(OUTPUT_DIR, 'sign_language_model.keras')
    class_names_path_load = os.path.join(OUTPUT_DIR, 'class_names.pkl')

    try:
        print("Memuat model...")
        model = load_model(model_path_load)
        with open(class_names_path_load, 'rb') as f:
            class_names = pickle.load(f)
        print("Model berhasil dimuat.")
    except Exception as e:
        print(f"Error: Tidak bisa memuat model atau file class_names. Pastikan Anda sudah menjalankan training.\nDetail: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak bisa membuka webcam.")
        return

    print("\nMemulai prediksi real-time... Tekan 'q' untuk keluar.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (IMG_SIZE, IMG_SIZE))
        input_frame = np.expand_dims(resized_frame, axis=0) / 255.0
        
        # Predict
        prediction = model.predict(input_frame, verbose=0)
        predicted_class_index = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Display prediction
        label = f"{class_names[predicted_class_index]}: {confidence:.2f}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Sign Language Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    real_time_prediction()
