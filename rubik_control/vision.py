# vision.py
import cv2
import numpy as np
import time
import os
import subprocess

# 1. RANGOS FÍSICOS
RANGOS_HSV_REAL = {
    'blanco':   ([0, 0, 140],      [179, 65, 255],  'U'),
    'amarillo': ([29, 100, 100],   [41, 255, 255],  'D'),
    'verde':    ([53, 100, 100],   [87, 255, 250],  'F'),
    'azul':     ([91, 130, 113],   [127, 255, 205], 'B'),
    'naranja':  ([10, 123, 125],   [23, 255, 255],  'L'),
    'rojo_1':   ([0, 117, 119],    [11, 255, 255],  'R'), 
    'rojo_2':   ([165, 117, 119],  [179, 255, 255], 'R')  
}

# 2. RANGOS VIRTUALES
RANGOS_HSV_VIRTUAL = {
    'blanco':   ([0, 0, 180],      [179, 40, 255],  'U'), 
    'amarillo': ([25, 120, 120],   [35, 255, 255],  'D'),
    'verde':    ([45, 50, 30],     [85, 255, 255],  'F'), 
    'azul':     ([100, 120, 100],  [130, 255, 255], 'B'),
    'naranja':  ([10, 140, 120],   [22, 255, 255],  'L'), 
    'rojo_1':   ([0, 140, 80],     [9, 255, 255],   'R'), 
    'rojo_2':   ([165, 140, 80],   [179, 255, 255], 'R')  
}

def obtener_ruta_virtual_universal():
    try:
        comando = 'cmd.exe /c "echo %TEMP%"'
        temp_windows = subprocess.check_output(comando, shell=True, text=True).strip()
        temp_wsl = temp_windows.replace("C:\\", "/mnt/c/").replace("c:\\", "/mnt/c/").replace("\\", "/")
        ruta_final = f"{temp_wsl}/RobotStudio_Vision_TFG/snapshot_yolo.png"
        return ruta_final
    except Exception as e:
        print(f"[-] Error fatal obteniendo ruta universal de Windows: {e}")
        return ""

def obtener_color(hsv_pixel, diccionario_rangos):
    h, s, v = hsv_pixel
    for color, (bajo, alto, letra) in diccionario_rangos.items():
        bajo = np.array(bajo)
        alto = np.array(alto)
        if (bajo[0] <= h <= alto[0]) and (bajo[1] <= s <= alto[1]) and (bajo[2] <= v <= alto[2]):
            return letra
    return '?'

# Radio de muestreo (en px) alrededor de cada centro de sticker. Lo fija
# dibujar_cuadricula() en funcion del tamano detectado del cubo.
_radio_muestreo = 3

def muestrear_hsv(frame_hsv, px, py):
    """Mediana HSV de un parche alrededor del punto: robusto frente a
    ruido y bordes, a cualquier distancia del cubo."""
    r = max(1, _radio_muestreo)
    alto, ancho = frame_hsv.shape[:2]
    x0, x1 = max(0, px - r), min(ancho, px + r + 1)
    y0, y1 = max(0, py - r), min(alto, py + r + 1)
    parche = frame_hsv[y0:y1, x0:x1].reshape(-1, 3)
    return np.median(parche, axis=0).astype(int)

def dibujar_cuadricula(frame):
    """DETECCION AUTOMATICA DEL CUBO: localiza la cara visible por color
    (stickers saturados + marco negro del cubo) y ajusta la rejilla 3x3 a
    su posicion y tamano reales, independientemente de la distancia a la
    camara. El fondo gris y la pinza blanca (baja saturacion, brillo
    medio/alto) quedan excluidos de la mascara.
    Si no se detecta nada, usa la ultima posicion valida y, en su
    defecto, la rejilla fija clasica en el centro del frame."""
    global _radio_muestreo
    alto, ancho, _ = frame.shape

    # --- 1. Mascara "esto es cubo": colores saturados O plastico negro ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask_color = ((s >= 90) & (v >= 60)).astype(np.uint8) * 255
    mask_negro = (v <= 70).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask_color, mask_negro)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # --- 2. Contorno mayor con pinta de cara de cubo (cuadrado-ish) ---
    deteccion = None
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        c = max(contornos, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        area_rel = (w * h) / float(ancho * alto)
        ratio = w / float(h) if h > 0 else 0
        if 0.01 < area_rel < 0.9 and 0.6 < ratio < 1.6:
            deteccion = (x + w / 2.0, y + h / 2.0, min(w, h))

    # --- 3. Suavizado temporal (EMA) y memoria de ultima deteccion ---
    if deteccion is not None:
        if getattr(dibujar_cuadricula, 'caja', None) is None:
            dibujar_cuadricula.caja = deteccion
        else:
            a = 0.4  # peso de la medida nueva
            pcx, pcy, pl = dibujar_cuadricula.caja
            dibujar_cuadricula.caja = (pcx + a * (deteccion[0] - pcx),
                                       pcy + a * (deteccion[1] - pcy),
                                       pl + a * (deteccion[2] - pl))

    if getattr(dibujar_cuadricula, 'caja', None) is not None:
        ccx, ccy, lado_cubo = dibujar_cuadricula.caja
        celda = lado_cubo / 3.0
        estado = 'CUBO DETECTADO' if deteccion is not None else 'CUBO (ultima pos.)'
        color_estado = (0, 255, 0) if deteccion is not None else (0, 200, 255)
        cv2.rectangle(frame, (int(ccx - lado_cubo / 2), int(ccy - lado_cubo / 2)),
                      (int(ccx + lado_cubo / 2), int(ccy + lado_cubo / 2)), (0, 255, 255), 1)
    else:
        # Fallback: rejilla fija clasica en el centro
        ccx, ccy, celda = ancho / 2.0, alto / 2.0, 100.0
        estado, color_estado = 'BUSCANDO CUBO...', (0, 0, 255)

    cv2.putText(frame, estado, (20, alto - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_estado, 2)

    # --- 4. Rejilla 3x3 adaptada (mismo orden de lectura que siempre:
    #        filas de arriba a abajo, columnas de derecha a izquierda,
    #        coherente con el cv2.flip del bucle principal) ---
    _radio_muestreo = max(2, int(celda * 0.12))
    lado_caja = int(celda * 0.6)
    puntos_centrales = []
    for fila in (-1, 0, 1):
        for col in (1, 0, -1):
            px = int(ccx + col * celda)
            py = int(ccy + fila * celda)
            cv2.rectangle(frame, (px - lado_caja // 2, py - lado_caja // 2),
                          (px + lado_caja // 2, py + lado_caja // 2), (0, 255, 0), 2)
            cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)
            puntos_centrales.append((px, py))
    return frame, puntos_centrales

def escanear_cubo(fn_is_robot_listo, fn_set_comando_robot, fn_get_telemetria, modo="real"):
    if modo == "real":
        print("[*] Encendiendo cámara externa (Webcam)...")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened(): return ""
    else:
        ruta_virtual = obtener_ruta_virtual_universal()
        print(f"[*] Modo VIRTUAL activo. Leyendo a alta velocidad: {ruta_virtual}")
        
        if not ruta_virtual or not os.path.exists(os.path.dirname(ruta_virtual)):
            print("[-] ADVERTENCIA: La carpeta temporal de RobotStudio no existe todavía.")
            print("[-] Por favor, abre la pestaña de la Cámara Virtual en RobotStudio antes de continuar.")
            return ""

    print("\n=======================================================")
    print("[+] CAMARA OK. ESPERANDO SINCRONIZACIÓN CON ROS 2...")
    print("=======================================================\n")

    # --- ORDEN DE ESCANEO (rutina optimizada: 3 viajes x 2 caras) ---
    # Debe coincidir con Rutina_Escanear_Cubo en RAPID:
    #   Viaje 1 (agarre 0, dedos F/B):  L (Rz=180), R (Rz=0)
    #   Viaje 2 (agarre 90, dedos L/R): F (Rz=180), B (Rz=0)
    #   Viaje 3 (tras volteo F->U):     D (Rz=180), U (Rz=0)
    # Si en la calibracion dos caras de un par salen intercambiadas,
    # permuta aqui sus letras (o los Rz en RAPID, pero no ambos).
    secuencia = ['L', 'R', 'F', 'B', 'D', 'U']

    # --- CORRECCION DE ORIENTACION POR CARA (calibrable) ---
    # Numero de rotaciones de 90 grados HORARIAS a aplicar a la rejilla
    # 3x3 capturada de cada cara antes de guardarla. Con la presentacion
    # nueva, el giro de muneca que selecciona la cara tambien fija como
    # aparece rotada en la imagen; este mapa corrige ese residuo.
    # CALIBRACION: escanea en virtual con un scramble conocido y ajusta
    # cada valor (0,1,2,3) hasta que el mapa 2D coincida con el cubo.
    ROTACION_CARA = {'F': 0, 'B': 0, 'L': 0, 'R': 0, 'D': 0, 'U': 0}

    def rotar_cara(cara_str, veces):
        """Rota una cara 3x3 (string de 9 stickers, orden lectura) 90 grados
        horarios 'veces' veces."""
        s = cara_str
        for _ in range(veces % 4):
            s = s[6] + s[3] + s[0] + s[7] + s[4] + s[1] + s[8] + s[5] + s[2]
        return s
    datos_caras = {}
    paso = 0
    esperando_bajada_robot = False
    tiempo_ultima_foto = 0
    cara_estable_previa = ""
    tiempo_inicio_estabilidad = 0.0
    comando_actual = 0 

    rangos_activos = RANGOS_HSV_VIRTUAL if modo == "virtual" else RANGOS_HSV_REAL

    while paso < 6:
        if modo == "real":
            ret, frame = cap.read()
            if not ret: continue
        else:
            try:
                if not os.path.exists(ruta_virtual) or os.path.getsize(ruta_virtual) == 0: continue
                with open(ruta_virtual, 'rb') as f:
                    file_bytes = f.read()
                img_array = np.frombuffer(file_bytes, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is None: continue
            except Exception:
                continue

        frame = cv2.flip(frame, 1)

        frame_viz, puntos = dibujar_cuadricula(frame.copy())
        cara_actual = secuencia[paso]
        t_ahora = time.monotonic()

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        letras_detectadas = [obtener_color(muestrear_hsv(frame_hsv, px, py), rangos_activos) for px, py in puntos]
        cara_str = "".join(letras_detectadas)

        for i, (px, py) in enumerate(puntos):
            letra = letras_detectadas[i]
            color_texto = (0, 255, 0) if letra != '?' else (0, 0, 255)
            cv2.putText(frame_viz, letra, (px - 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_texto, 2)

        tecla = cv2.waitKey(1) & 0xFF
        forzar_foto = (tecla == ord(' '))
        robot_listo = fn_is_robot_listo()

        # ---- LÓGICA DE ESTABILIDAD CORREGIDA ----
        if t_ahora - tiempo_ultima_foto < 1.0:
            # Mantenemos el "1" solo mientras el robot NO haya confirmado la foto.
            # En cuanto baja su flag (robot_listo = False) está esperando nuestro "0",
            # así que se lo damos al instante: ahorra ~1s muerto por cara.
            if not robot_listo:
                esperando_bajada_robot = False
                comando_actual = 0
            else:
                comando_actual = 1
            fn_set_comando_robot(comando_actual)
            cv2.putText(frame_viz, f"[OK] Cara {cara_actual} guardada", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            if esperando_bajada_robot:
                comando_actual = 1
                fn_set_comando_robot(comando_actual)
                if not robot_listo:
                    esperando_bajada_robot = False
                    comando_actual = 0
                    fn_set_comando_robot(comando_actual)
            else:
                if robot_listo:
                    if forzar_foto:
                        datos_caras[cara_actual] = rotar_cara(cara_str, ROTACION_CARA.get(cara_actual, 0))
                        paso += 1
                        esperando_bajada_robot = True
                        tiempo_ultima_foto = t_ahora 
                        comando_actual = 1
                        fn_set_comando_robot(comando_actual)
                        cara_estable_previa = "" 
                    else:
                        if '?' not in cara_str:
                            if cara_str == cara_estable_previa:
                                tiempo_transcurrido = t_ahora - tiempo_inicio_estabilidad
                                if tiempo_transcurrido >= 1.0: 
                                    print(f"\033[92m[+] Cara {cara_actual} capturada (Estable 1s): {cara_str}\033[0m")
                                    datos_caras[cara_actual] = rotar_cara(cara_str, ROTACION_CARA.get(cara_actual, 0))
                                    paso += 1
                                    esperando_bajada_robot = True
                                    tiempo_ultima_foto = t_ahora 
                                    comando_actual = 1
                                    fn_set_comando_robot(comando_actual)
                                    cara_estable_previa = "" 
                                else:
                                    comando_actual = 0
                                    fn_set_comando_robot(comando_actual) 
                                    cv2.putText(frame_viz, f"Fijando... {tiempo_transcurrido:.1f}s / 1.0s", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            else:
                                cara_estable_previa = cara_str
                                tiempo_inicio_estabilidad = t_ahora
                                comando_actual = 0
                                fn_set_comando_robot(comando_actual) 
                                cv2.putText(frame_viz, "Fijando... 0.0s / 1.0s", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    cara_estable_previa = ""
                    comando_actual = 0
                    fn_set_comando_robot(comando_actual) 
                    cv2.putText(frame_viz, "Analizando... (O pulsa ESPACIO para forzar)", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        texto_estado = f"Robot: PIDIENDO FOTO {cara_actual}" if robot_listo else "Robot: Moviendose..."
        color_estado = (0, 255, 255) if robot_listo else (0, 255, 0)
        cv2.putText(frame_viz, texto_estado, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)
        cv2.putText(frame_viz, f"Recibe del Robot: {fn_get_telemetria():.1f}", (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame_viz, f"Envia al Robot:   {comando_actual:.1f}", (20, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Escaner TFG", frame_viz)
        if tecla == ord('q'): break
        
        # --- EL FRENO QUE EVITA QUE SE CUELGUE ROBOTSTUDIO ---
        time.sleep(0.05)

    if modo == "real": cap.release()
    cv2.destroyAllWindows()

    if paso == 6:
        # --- CIERRE LIMPIO DEL HANDSHAKE FINAL ---
        # Tras la 6a cara el RAPID queda en "WaitUntil EntradaRubik{3} = 0".
        # Si salimos dejando el flag en 1, el robot se queda con el brazo
        # extendido delante de la camara. Enviamos el 0 de forma sostenida
        # para que deposite el cubo y pase a esperar la solucion.
        print("[*] Cerrando handshake final con el robot...")
        t_fin = time.monotonic() + 3.0
        while time.monotonic() < t_fin:
            fn_set_comando_robot(0)
            time.sleep(0.05)

        string_final = datos_caras['U'] + datos_caras['R'] + datos_caras['F'] + datos_caras['D'] + datos_caras['L'] + datos_caras['B']
        print(f"\n[*] STRING FINAL KOCIEMBA: {string_final}")
        return string_final
    return ""

COLORES_BGR = {
    'U': (255, 255, 255), 'R': (0, 0, 255), 'F': (0, 255, 0),
    'D': (0, 255, 255),   'L': (0, 165, 255), 'B': (255, 0, 0)
}

def validar_cubo(string_final):
    errores = [f"{c}: {string_final.count(c)}/9" for c in ['U', 'R', 'F', 'D', 'L', 'B'] if string_final.count(c) != 9]
    if errores:
        print(f"\n[!] ERROR DE INTEGRIDAD: El cubo detectado es matemáticamente imposible.")
        print(f"[*] Detalle de errores: {', '.join(errores)}")
        return False
    return True

def mostrar_mapa_2d(string_final):
    datos_caras = {'U': string_final[0:9], 'R': string_final[9:18], 'F': string_final[18:27],
                   'D': string_final[27:36], 'L': string_final[36:45], 'B': string_final[45:54]}
    
    size_sticker, gap = 40, 5
    face_size = (size_sticker * 3) + (gap * 4)
    img = np.zeros((face_size * 3, face_size * 4, 3), dtype=np.uint8)
    offsets = {'U': (0, 1), 'L': (1, 0), 'F': (1, 1), 'R': (1, 2), 'B': (1, 3), 'D': (2, 1)}
    
    for cara, (fila, col) in offsets.items():
        for i in range(9):
            color_bgr = COLORES_BGR.get(datos_caras[cara][i], (128, 128, 128))
            x1 = col * face_size + (i % 3) * (size_sticker + gap) + gap
            y1 = fila * face_size + (i // 3) * (size_sticker + gap) + gap
            cv2.rectangle(img, (x1, y1), (x1 + size_sticker, y1 + size_sticker), color_bgr, -1)
            cv2.rectangle(img, (x1, y1), (x1 + size_sticker, y1 + size_sticker), (50, 50, 50), 1)

    cv2.imshow("Mapa 2D - Cubo Detectado", img)
    print("\n[*] Mostrando mapa 2D...\n[*] El programa continuará en 3 segundos...")
    cv2.waitKey(3000) 
    cv2.destroyWindow("Mapa 2D - Cubo Detectado")
    return img