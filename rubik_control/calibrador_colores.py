# calibrador_colores.py
import cv2
import numpy as np

def nothing(x):
    pass

print("[*] Iniciando Calibrador de HSV (Versión Anti-Bugs WSL)...")

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Ventana más ancha para intentar forzar a WSL a pintar bien
cv2.namedWindow('Calibrador', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Calibrador', 600, 400)

# El orden de las barras de arriba a abajo es este:
cv2.createTrackbar('H Min', 'Calibrador', 0, 179, nothing)
cv2.createTrackbar('S Min', 'Calibrador', 0, 255, nothing)
cv2.createTrackbar('V Min', 'Calibrador', 0, 255, nothing)
cv2.createTrackbar('H Max', 'Calibrador', 179, 179, nothing)
cv2.createTrackbar('S Max', 'Calibrador', 255, 255, nothing)
cv2.createTrackbar('V Max', 'Calibrador', 255, 255, nothing)

print("[+] Calibrador listo. Mueve las barras a ciegas, verás los números en el vídeo.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    
    alto, ancho, _ = frame.shape
    cx, cy = ancho // 2, alto // 2
    cv2.rectangle(frame, (cx-40, cy-40), (cx+40, cy+40), (255, 255, 255), 2)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Leer las posiciones
    h_min = cv2.getTrackbarPos('H Min', 'Calibrador')
    s_min = cv2.getTrackbarPos('S Min', 'Calibrador')
    v_min = cv2.getTrackbarPos('V Min', 'Calibrador')
    h_max = cv2.getTrackbarPos('H Max', 'Calibrador')
    s_max = cv2.getTrackbarPos('S Max', 'Calibrador')
    v_max = cv2.getTrackbarPos('V Max', 'Calibrador')

    # --- EL TRUCO: ESCRIBIR LOS VALORES EN EL VÍDEO ---
    # Tono (H) en Rojo
    cv2.putText(frame, f"H (Color): MIN {h_min} - MAX {h_max}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # Saturacion (S) en Verde
    cv2.putText(frame, f"S (Intensidad): MIN {s_min} - MAX {s_max}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Brillo/Valor (V) en Azul
    cv2.putText(frame, f"V (Brillo): MIN {v_min} - MAX {v_max}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower, upper)
    resultado = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Original', frame)
    cv2.imshow('Mascara (Blanco = Detectado)', mask)
    cv2.imshow('Resultado', resultado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
