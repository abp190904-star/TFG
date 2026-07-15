#!/usr/bin/env python3
import sys
import os

# --- SILENCIADOR NUCLEAR DE WARNINGS ---
os.environ["QT_LOGGING_RULES"] = "*=false"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
# ---------------------------------------

import time
import cv2
import psutil
import GPUtil  # <--- NUEVO: Librería para monitorizar la GPU
import shutil
import json
import random
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64


# --- IMPORTS ACTUALIZADOS A ROS 2 ---
from rubik_control.vision import escanear_cubo, validar_cubo, mostrar_mapa_2d
from rubik_control.solver import resolver_cubo

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class RubikOrchestratorNode(Node):
    def __init__(self, modo_ejecucion):
        super().__init__('rubik_orchestrator')
        self.modo_ejecucion = modo_ejecucion
        self.get_logger().info(f'Orquestador del TFG iniciado (Modo {modo_ejecucion.upper()})')

        self.robot_telemetria = 0.0
        self.robot_terminado = False
        self.mezclado_terminado = False

        self.proceso = psutil.Process(os.getpid())
        psutil.cpu_percent(interval=None)

        # --- PERFIL QoS DEL CANAL DE VISIÓN ---
        # IMPORTANTE: debe ser RELIABLE en AMBOS extremos (main.py y robot.py).
        # Un publisher BEST_EFFORT con un subscriber RELIABLE es INCOMPATIBLE
        # en DDS y los mensajes NO se entregan (era la causa del bloqueo del
        # escaneo). depth=10 da margen sin acumular retraso apreciable en
        # señales de nivel (0/1) que se republiican continuamente.
        qos_vision = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # QoS con memoria (TRANSIENT_LOCAL): si robot_node aun no ha elegido
        # su modo en el menu cuando se publica un comando, lo recibira al
        # suscribirse. Debe coincidir con el qos_comandos de robot.py.
        qos_comandos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                  history=HistoryPolicy.KEEP_LAST, depth=10,
                                  durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher_ = self.create_publisher(String, 'comando_movimiento', qos_comandos)
        self.pub_scramble = self.create_publisher(String, 'comando_scramble', qos_comandos)

        self.scan_cmd_pub = self.create_publisher(Float64, 'scan_cmd', qos_vision)
        self.scan_state_sub = self.create_subscription(Float64, 'scan_state', self.scan_state_callback, qos_vision)

    def generar_scramble(self, n=15):
        caras = ['U', 'D', 'F', 'B', 'R', 'L']
        sufijos = ['', "'", '2']
        scramble = []
        ultima_cara = ''
        while len(scramble) < n:
            cara = random.choice(caras)
            if cara != ultima_cara:
                sufijo = random.choice(sufijos)
                scramble.append(cara + sufijo)
                ultima_cara = cara
        return " ".join(scramble)

    def scan_state_callback(self, msg):
        self.robot_telemetria = msg.data

        if int(msg.data) == 98:
            self.mezclado_terminado = True

        if msg.data == -1.0:
            self.robot_terminado = True

    # --- CALLBACKS VISION ---
    def cb_is_robot_listo(self):
        # Bajamos el tiempo de espera para que la CPU respire
        rclpy.spin_once(self, timeout_sec=0.002)
        return self.robot_telemetria >= 0.9

    def cb_set_comando(self, valor):
        # VERSIÓN LIMPIA Y DIRECTA: Publica siempre la orden sin filtros
        msg = Float64()
        msg.data = float(valor)
        self.scan_cmd_pub.publish(msg)
        rclpy.spin_once(self, timeout_sec=0.002)

    def cb_get_telemetria(self):
        rclpy.spin_once(self, timeout_sec=0.002)
        return self.robot_telemetria
    # ------------------------

    # --- ACTUALIZADO: Recibe variables de GPU ---
    def guardar_en_historial(self, img_cubo, detectado, solucion, t_fraccionados, ram_uso, cpu_uso, gpu_uso, vram_uso):
        # --- RUTA BASE ACTUALIZADA ---
        ruta_base = os.path.expanduser('~/rubik_ros2_ws/historial')
        if not os.path.exists(ruta_base): os.makedirs(ruta_base)

        nombre_carpeta = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ruta_run = os.path.join(ruta_base, nombre_carpeta)
        os.makedirs(ruta_run)

        if img_cubo is not None: cv2.imwrite(os.path.join(ruta_run, 'mapa_2d.png'), img_cubo)

        ruta_csv_temp = os.path.join(ruta_base, 'latest_telemetria.csv')
        if os.path.exists(ruta_csv_temp): shutil.move(ruta_csv_temp, os.path.join(ruta_run, 'telemetria_ejes.csv'))

        fecha_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        num_movimientos = len(solucion.split()) if solucion else 1
        cadencia = t_fraccionados['fisico'] / num_movimientos if num_movimientos > 0 else 0.0
        vel_solver = num_movimientos / t_fraccionados['calculo'] if t_fraccionados['calculo'] > 0 else 0.0

        # --- REPORTE CON MONITORIZACIÓN GRÁFICA ---
        texto_reporte = f"""========================================================
           REPORTE TÉCNICO DE RENDIMIENTO (TFG)
========================================================
Fecha de ejecución: {fecha_str}

--- 1. ANÁLISIS DEL SOLVER ---
String Identificado por Visión:  {detectado}
Solución Cinemática Generada:    {solucion}
Longitud de la Solución (N):     {num_movimientos} movimientos

--- 2. DESGLOSE TIMING PERFORMANCE ---
Tiempo Escaneo (Percepción Visual):   {t_fraccionados['vision']:.2f} s
Tiempo Cómputo (Algorítmica Solver):  {t_fraccionados['calculo']:.4f} s
Tiempo de Ejecución Mecánica (Robot): {t_fraccionados['fisico']:.2f} s
Tiempo Total de Operación del Sistema: {t_fraccionados['total']:.2f} s

--- 3. MÉTRICAS EFICIENCIA MECÁNICA ---
Cadencia Media de Giro:               {cadencia:.2f} seg/movimiento
Velocidad de Respuesta del Solver:    {vel_solver:.2f} movimientos/seg

--- 4. DIAGNÓSTICO DE CARGA COMPUTACIONAL (PC) ---
Consumo Máximo de Memoria RAM:        {ram_uso:.2f} MB
Carga Media de la CPU:                {cpu_uso:.1f} %
Carga Media de GPU (NVIDIA RTX 3060): {gpu_uso:.1f} %
Consumo de VRAM (Memoria de Vídeo):   {vram_uso:.2f} MB
========================================================
"""
        ruta_txt = os.path.join(ruta_run, 'datos_resolucion.txt')
        with open(ruta_txt, 'w', encoding='utf-8') as f:
            f.write(texto_reporte)

        # --- ANEXO OPCIONAL: METRICAS DEL NODO OBSERVADOR ---
        # Totalmente aditivo: si el nodo de metricas no esta lanzado o no ha
        # volcado aun su reporte, se omite sin afectar al historial base.
        try:
            self._anexar_metricas(ruta_run, ruta_txt)
        except Exception as e:
            self.get_logger().warn(f'[Metricas] No se pudo anexar el reporte: {e}')

        self.get_logger().info(f"[+] HISTORIAL GUARDADO EN: {ruta_run}")

    def _anexar_metricas(self, ruta_run, ruta_txt):
        """Espera (max 10s) al reporte del nodo de metricas, anexa su resumen
        al datos_resolucion.txt y copia sus CSVs a la carpeta de la ejecucion."""
        ruta_json = os.path.expanduser('~/rubik_ros2_ws/historial/metricas/latest_metricas.json')

        t_limite = time.monotonic() + 10.0
        while time.monotonic() < t_limite:
            # Solo aceptamos un reporte RECIENTE (evita arrastrar uno viejo)
            if os.path.exists(ruta_json) and (datetime.now().timestamp() - os.path.getmtime(ruta_json)) < 120:
                break
            time.sleep(0.5)
        else:
            self.get_logger().info('[Metricas] Nodo observador no detectado; historial sin anexo de metricas.')
            return

        with open(ruta_json, 'r') as f:
            m = json.load(f)

        lat = m.get('latencia_ms', {})
        dist = m.get('distancia_tcp_m', {})
        lineas = [
            '',
            '--- 5. MÉTRICAS DE COMUNICACIÓN Y CINEMÁTICA (nodo observador) ---',
            f"Latencia Handshake EGM (media):       {lat.get('media')} ms",
            f"Latencia Handshake EGM (min / max):   {lat.get('minima')} / {lat.get('maxima')} ms",
            f"Comandos EGM medidos:                 {lat.get('n_comandos')}",
            f"Distancia TCP total de la sesión:     {dist.get('total_sesion')} m",
            f"Distancia TCP durante la resolución:  {dist.get('solo_resolucion')} m",
        ]
        for eje in m.get('cinematica_por_eje') or []:
            lineas.append(f"Eje {eje['eje']}: v_max={eje['vel_max_deg_s']} deg/s | "
                          f"a_max={eje['acel_max_deg_s2']} deg/s2 | jerk_max={eje['jerk_max_deg_s3']} deg/s3")
        lineas.append('(Detalle completo en la subcarpeta metricas/)')
        lineas.append('========================================================')

        with open(ruta_txt, 'a', encoding='utf-8') as f:
            f.write('\n'.join(lineas) + '\n')

        # Copiamos los CSVs del reporte de metricas a la carpeta del run
        carpeta_metricas = m.get('carpeta')
        if carpeta_metricas and os.path.isdir(carpeta_metricas):
            destino = os.path.join(ruta_run, 'metricas')
            shutil.copytree(carpeta_metricas, destino)
        self.get_logger().info('[Metricas] Resumen anexado al reporte y CSVs copiados al historial.')

    def resolver_cubo(self):
        t_inicio_total = time.monotonic()
        tiempos = {'vision': 0.0, 'calculo': 0.0, 'fisico': 0.0, 'total': 0.0}

        try:
            # --- PASO 0: MEZCLADO ---
            if self.modo_ejecucion == "virtual":
                scramble_str = self.generar_scramble(15)
                self.get_logger().info(f'--- PASO 0: Mezclado Aleatorio ({scramble_str}) ---')
            else:
                scramble_str = ""
                self.get_logger().info('--- PASO 0: Cubo Físico (Mezclado Manual) ---')
                self.get_logger().info('Asegúrate de haber mezclado el cubo a mano en la base.')

            self.mezclado_terminado = False
            msg = String()
            msg.data = scramble_str

            time.sleep(1.0)

            while self.pub_scramble.get_subscription_count() == 0:
                self.get_logger().info('Esperando enlace de red con robot.py...')
                time.sleep(0.5)

            self.pub_scramble.publish(msg)

            self.get_logger().info('Sincronizando con el robot para iniciar escaneo...')
            while not self.mezclado_terminado:
                rclpy.spin_once(self, timeout_sec=0.1)

            self.get_logger().info('Sincronización completada. Abriendo visión...')
            time.sleep(1.0)

            # Limpiamos la telemetría residual del mezclado (el 98.0 retenido).
            # Si no, cb_is_robot_listo() devolvería True durante los primeros
            # segundos del escaneo cuando el robot aún está recogiendo el cubo.
            self.robot_telemetria = 0.0

            # --- PASO 1: VISIÓN ---
            self.get_logger().info('--- PASO 1: Visión ---')
            t_0 = time.monotonic()
            estado_actual = escanear_cubo(self.cb_is_robot_listo, self.cb_set_comando, self.cb_get_telemetria, modo=self.modo_ejecucion)
            t_1 = time.monotonic()
            tiempos['vision'] = t_1 - t_0

            if not estado_actual: return
            img_cubo = mostrar_mapa_2d(estado_actual)

            if not validar_cubo(estado_actual):
                self.enviar_comando_robot("ABORTAR")
                return

            # --- PASO 2: SOLVER ---
            self.get_logger().info('--- PASO 2: Solver ---')
            t_2 = time.monotonic()
            solucion_movimientos = resolver_cubo(estado_actual)
            t_3 = time.monotonic()
            tiempos['calculo'] = t_3 - t_2

            if not solucion_movimientos:
                 self.enviar_comando_robot("ABORTAR")
                 return

            self.get_logger().info(f'Solución encontrada: {solucion_movimientos}')

            # --- PASO 3: EJECUCIÓN ---
            self.get_logger().info('--- PASO 3: Ejecución Física ---')

            # --- RUTA CSV ACTUALIZADA ---
            ruta_csv_temp = os.path.expanduser('~/rubik_ros2_ws/historial/latest_telemetria.csv')
            if os.path.exists(ruta_csv_temp): os.remove(ruta_csv_temp)

            t_inicio_fisico = time.monotonic()

            self.enviar_comando_robot(solucion_movimientos)

            self.get_logger().info('Esperando a que el robot termine la resolución física...')
            while not os.path.exists(ruta_csv_temp):
                rclpy.spin_once(self, timeout_sec=0.5)
            time.sleep(1.0)

            tiempos['fisico'] = time.monotonic() - t_inicio_fisico
            tiempos['total'] = time.monotonic() - t_inicio_total

            # --- CÁLCULOS FINALES DE RENDIMIENTO ---
            ram_final = self.proceso.memory_info().rss / (1024 * 1024)
            cpu_final = psutil.cpu_percent(interval=None)

            try:
                gpus = GPUtil.getGPUs()
                gpu_load = gpus[0].load * 100 if gpus else 0.0
                vram_used = gpus[0].memoryUsed if gpus else 0.0
            except Exception:
                gpu_load = 0.0
                vram_used = 0.0
            # ---------------------------------------

            self.guardar_en_historial(img_cubo, estado_actual, solucion_movimientos, tiempos, ram_final, cpu_final, gpu_load, vram_used)
            self.get_logger().info('Operación completada limpiamente.')

        except Exception as e:
            self.get_logger().error(f'Error general: {e}')
            self.enviar_comando_robot("ABORTAR")

    def enviar_comando_robot(self, comando):
        msg = String()
        msg.data = comando
        self.publisher_.publish(msg)


def _modo_desde_argv():
    """Permite arrancar sin menu interactivo (para el launch file):
       ros2 run ... --modo v   |   --modo r
    Si no se pasa el argumento, devuelve None y se muestra el menu de siempre."""
    import sys
    for i, a in enumerate(sys.argv):
        if a == '--modo' and i + 1 < len(sys.argv):
            v = sys.argv[i + 1].strip().upper()
            if v.startswith('V'):
                return 'virtual'
            if v.startswith('R'):
                return 'real'
    return None

def main(args=None):
    print("\n=======================================================")
    print("         ORQUESTADOR TFG - RESOLUCIÓN RUBIK")
    print("=======================================================")
    print("Selecciona el modo de ejecución:")
    print("  [V] Virtual (Simulación RobotStudio + Add-in Visión)")
    print("  [R] Real (Brazo Omnicore Físico + Webcam Física)")
    print("=======================================================")
    modo_ejecucion = _modo_desde_argv()
    if modo_ejecucion is None:
        seleccion = input("Elige una opción (V/R): ").strip().upper()
        modo_ejecucion = "virtual" if seleccion == 'V' else "real"
    else:
        print(f">> Modo seleccionado por argumento: {modo_ejecucion.upper()}")

    rclpy.init(args=args)
    orquestador = RubikOrchestratorNode(modo_ejecucion)
    try:
        orquestador.resolver_cubo()
    except KeyboardInterrupt:
        pass
    finally:
        orquestador.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()