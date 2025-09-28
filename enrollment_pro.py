import cv2
import numpy as np
import tensorflow as tf
import os
import time

# --- Configuration ---
TFLITE_MODEL_PATH = 'output_model.tflite'
INPUT_SIZE = (112, 112)
DATABASE_FILE = 'face_database.npy'
NUM_PICS_TO_ENROLL = 3 # ## NEW ## Number of pictures to take for enrollment

# --- Load Models (unchanged) ---
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
detector_path = "face_detector"
weights_path = detector_path + "/res10_300x300_ssd_iter_140000.caffemodel"
config_path = detector_path + "/deploy.prototxt"
face_detector = cv2.dnn.readNet(config_path, weights_path)

# --- Load Database (unchanged) ---
if os.path.exists(DATABASE_FILE):
    database = np.load(DATABASE_FILE, allow_pickle=True).item()
else:
    database = {}

def get_embedding(face_image):
    # This function is unchanged
    if face_image is None: return None
    img = cv2.resize(face_image, INPUT_SIZE).astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# --- ## NEW LOGIC: Guided Enrollment Process ---
cap = cv2.VideoCapture(0)
pics_taken = 0
embeddings_list = []
prompts = ["Please look STRAIGHT at the camera", "Please turn your head SLIGHTLY LEFT", "Please turn your head SLIGHTLY RIGHT"]

while pics_taken < NUM_PICS_TO_ENROLL:
    ret, frame = cap.read()
    if not ret: break

    # Display the current prompt for the user
    prompt_text = f"({pics_taken + 1}/{NUM_PICS_TO_ENROLL}) {prompts[pics_taken]}"
    cv2.putText(frame, prompt_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Align face in box and press 'C' to capture.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()
    
    current_face_crop = None
    if detections.shape[2] > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        if confidence > 0.8:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)
            if (endX - startX) > 0 and (endY - startY) > 0:
                current_face_crop = frame[startY:endY, startX:endX]
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow('Enrollment - Press C to Capture, Q to Quit', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    
    elif key == ord('c'):
        if current_face_crop is not None:
            embedding = get_embedding(current_face_crop)
            embeddings_list.append(embedding)
            pics_taken += 1
            print(f"Captured picture {pics_taken}/{NUM_PICS_TO_ENROLL}")
            # Simple feedback to the user
            cv2.putText(frame, "Captured!", (w // 2 - 50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.imshow('Enrollment - Press C to Capture, Q to Quit', frame)
            cv2.waitKey(500) # Pause for half a second
        else:
            print("No face detected! Please try again.")

cap.release()
cv2.destroyAllWindows()

# --- ## NEW LOGIC: Unify and Save the Embeddings ---
if len(embeddings_list) == NUM_PICS_TO_ENROLL:
    # Calculate the average of all captured embeddings
    unified_embedding = np.mean(embeddings_list, axis=0)
    
    name = input(f"All {NUM_PICS_TO_ENROLL} pictures captured. Please enter your name: ")
    if name:
        database[name] = unified_embedding
        np.save(DATABASE_FILE, database)
        print(f"Success! Unified face embedding for '{name}' saved.")
    else:
        print("Invalid name. Enrollment cancelled.")
else:
    print("Enrollment process was not completed. Nothing saved.")