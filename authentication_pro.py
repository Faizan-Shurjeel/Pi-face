import cv2
import numpy as np
import tensorflow as tf
import os
import time # ## NEW ## For timing the inference interval
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
TFLITE_MODEL_PATH = 'output_model.tflite'
INPUT_SIZE = (112, 112)
DATABASE_FILE = 'face_database.npy'
AUTH_THRESHOLD = 0.70
INFERENCE_INTERVAL = 1.0  # ## NEW ## Run inference every 1.0 seconds

# --- Load Models and Database (unchanged) ---
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
detector_path = "face_detector"
weights_path = detector_path + "/res10_300x300_ssd_iter_140000.caffemodel"
config_path = detector_path + "/deploy.prototxt"
face_detector = cv2.dnn.readNet(config_path, weights_path)
if not os.path.exists(DATABASE_FILE):
    print(f"Error: Database file '{DATABASE_FILE}' not found.")
    exit()
database = np.load(DATABASE_FILE, allow_pickle=True).item()
known_face_names = list(database.keys())
known_face_embeddings = np.array(list(database.values()))
print(f"Database loaded with {len(known_face_names)} enrolled faces.")

def get_embedding(face_image):
    # This function is unchanged
    if face_image is None: return None
    img = cv2.resize(face_image, INPUT_SIZE).astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# --- ## NEW LOGIC: Timed Inference Loop ---
cap = cv2.VideoCapture(0)
last_inference_time = 0
display_name = "Unknown"
display_status = "Auth Failed"
display_color = (0, 0, 255) # Red
box_coords = None

while True:
    ret, frame = cap.read()
    if not ret: break

    current_time = time.time()
    
    # --- Only run inference on the set interval ---
    if (current_time - last_inference_time) > INFERENCE_INTERVAL:
        last_inference_time = current_time # Reset the timer

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_detector.setInput(blob)
        detections = face_detector.forward()

        # Assume no face is found until one is
        best_match_found = False
        
        if detections.shape[2] > 0:
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                box_coords = (startX, startY, endX, endY) # Store coords

                face = frame[startY:endY, startX:endX]
                if face.shape[0] > 0 and face.shape[1] > 0:
                    live_embedding = get_embedding(face)
                    similarities = cosine_similarity(live_embedding.reshape(1, -1), known_face_embeddings)
                    best_match_index = np.argmax(similarities)
                    best_match_score = similarities[0][best_match_index]

                    if best_match_score > AUTH_THRESHOLD:
                        display_name = f"{known_face_names[best_match_index]}: {best_match_score:.2f}"
                        display_status = "Auth Passed"
                        display_color = (0, 255, 0)
                    else:
                        display_name = f"Unknown: {best_match_score:.2f}"
                        display_status = "Auth Failed"
                        display_color = (0, 0, 255)
                    best_match_found = True
        
        # If the detector ran but found no confident face, reset display
        if not best_match_found:
            box_coords = None # No box to draw

    # --- This part runs every frame, drawing the LAST known result ---
    if box_coords:
        cv2.putText(frame, display_name, (box_coords[0], box_coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)
        cv2.rectangle(frame, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), display_color, 2)
    
    # Display the persistent auth status
    cv2.putText(frame, display_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 2)
    
    cv2.imshow('Optimized Authentication - Press Q to Quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()