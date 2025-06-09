import sys
import time
import cv2 
import imutils
from yoloDet import YoloTRT

# Charger le modèle
model = YoloTRT(
    library="tensorrtx/build/libmyplugins.so",
    engine_path="tensorrtx/build/best.engine",
    conf=0.5,
    yolo_ver="v5"
)

# Charger la vidéo (ou caméra)
video_path = "video_101_flip.avi"  # Remplace par un fichier fonctionnel
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f" Erreur : Impossible d’ouvrir la vidéo '{video_path}'")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Fin de la vidéo ou erreur de lecture.")
        break

    frame = imutils.resize(frame, width=640)

    #  Mesurer le temps d'inférence
    start = time.time()
    detections, t = model.Inference(frame)
    end = time.time()
    inference_time_ms = (end - start) * 1000  # en millisecondes

    # Afficher le temps d'inférence sur l'image
    cv2.putText(frame, f"Inference: {inference_time_ms:.2f} ms", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)

    cv2.imshow("Output", frame)

    key = cv2.waitKey(10)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

