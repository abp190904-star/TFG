#!/usr/bin/env python3
# =============================================================================
#  METRICAS.PY - NODO OBSERVADOR DE METRICAS DEL TFG
# =============================================================================
#  Nodo 100% PASIVO: solo se suscribe a los topics que ya existen en el
#  sistema y NO publica nada ni interfiere en el funcionamiento base.
#  Se puede quitar del launch o cerrar en cualquier momento sin efecto
#  alguno sobre el pipeline principal.
#
#  Metricas que registra:
#    [1] Latencia del handshake EGM por comando (indice enviado -> ACK)
#    [2] Cinematica derivada de los ejes (velocidad, aceleracion, jerk)
#    [3] Distancia recorrida por el TCP (integracion de state/pose)
#    [5] Salud del sistema (frecuencia real de cada topic)
#
#  Salida: ~/rubik_ros2_ws/historial/metricas/metricas_<fecha>/
#    - resumen.txt        (informe legible)
#    - latencias.csv      (una fila por comando con su latencia)
#    - cinematica.csv     (t, posiciones, velocidades por eje)
#    - salud_topics.csv   (frecuencias medidas por ventana de 5s)
#    - metricas.json      (todo lo anterior en formato maquina)
# =============================================================================

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.clock import Clock, ClockType
from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose

import time
import os
import csv
import json
import math
from datetime import datetime

# Nombres legibles de los codigos de comando RAPID (ver robot.py)
NOMBRES_CODIGO = {98: 'FIN_MEZCLADO', 99: 'INICIO_RESOLUCION',
                  998: 'ABORTAR', 999: 'CUBO_RESUELTO'}

MAX_MUESTRAS_CINEMATICA = 300000   # tope de seguridad (~50 min a 100 Hz)


class NodoMetricas(Node):

    def __init__(self):
        super().__init__('rubik_metricas')

        qos_fiable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                history=HistoryPolicy.KEEP_LAST, depth=10)

        # ---------------- Estado interno ----------------
        # [1] Latencias de handshake
        self.t_envio_indice = {}      # indice -> t_monotonic del primer envio
        self.codigo_indice = {}       # indice -> codigo RAPID del comando
        self.ultimo_indice_visto = 0
        self.ultimo_ack_visto = 0
        self.latencias = []           # [(indice, codigo, fase, latencia_s)]
        self.fase = 'mezclado'        # mezclado -> escaneo -> resolucion

        # [2] Cinematica (buffer de articulaciones)
        self.buffer_joints = []       # [(t_mono, [j1..j6] en rad)]

        # [3] Distancia TCP
        self.pos_tcp_anterior = None
        self.distancia_total_m = 0.0
        self.distancia_resolucion_m = 0.0
        self.en_resolucion = False

        # [5] Salud de topics (contadores por ventana)
        self.contadores = {'command/data': 0, 'state/data': 0,
                           'state/joint': 0, 'state/pose': 0, 'scan_state': 0}
        self.historial_salud = []     # [(t_rel, {topic: hz})]
        self.t_inicio = time.monotonic()
        self.t_ventana = time.monotonic()

        self.reporte_generado = False

        # ---------------- Suscripciones (solo escucha) ----------------
        self.create_subscription(Float64MultiArray, 'command/data',
                                 self.cb_command_data, qos_fiable)
        self.create_subscription(Float64MultiArray, 'state/data',
                                 self.cb_state_data, qos_fiable)
        self.create_subscription(JointState, 'state/joint',
                                 self.cb_joint, 10)
        self.create_subscription(Pose, 'state/pose',
                                 self.cb_pose, 10)
        self.create_subscription(Float64, 'scan_state',
                                 self.cb_scan_state, qos_fiable)

        # Timer de salud con reloj monotonico (inmune a saltos de reloj)
        self.create_timer(5.0, self.cb_ventana_salud,
                          clock=Clock(clock_type=ClockType.STEADY_TIME))

        self.get_logger().info('Nodo de METRICAS iniciado (observador pasivo).')
        self.get_logger().info('Registrando: latencias EGM, cinematica, distancia TCP y salud de topics.')

    # =========================================================================
    #  [1] LATENCIA DE HANDSHAKE: command/data (envio) vs state/data (ACK)
    # =========================================================================
    def cb_command_data(self, msg):
        self.contadores['command/data'] += 1
        if len(msg.data) < 2:
            return
        indice = int(msg.data[0])
        codigo = int(msg.data[1])

        # Primer envio de un indice nuevo -> arrancamos su cronometro
        if indice > 0 and indice != self.ultimo_indice_visto:
            self.ultimo_indice_visto = indice
            self.t_envio_indice[indice] = time.monotonic()
            self.codigo_indice[indice] = codigo

    def cb_state_data(self, msg):
        self.contadores['state/data'] += 1
        if len(msg.data) < 1:
            return
        ack = int(msg.data[0])

        if ack != self.ultimo_ack_visto:
            self.ultimo_ack_visto = ack

            # ACK de fin de resolucion (999) no coincide con ningun indice
            if ack == 999:
                self.en_resolucion = False
                self.get_logger().info('[999] Cubo resuelto. Cerrando ventana de resolucion.')
                return

            if ack in self.t_envio_indice:
                lat = time.monotonic() - self.t_envio_indice.pop(ack)
                codigo = self.codigo_indice.pop(ack, -1)
                self.latencias.append((ack, codigo, self.fase, lat))
                nombre = NOMBRES_CODIGO.get(codigo, f'giro_{codigo}')
                self.get_logger().info(
                    f'[LAT] idx={ack:<3} cmd={nombre:<18} fase={self.fase:<10} '
                    f'latencia={lat*1000:7.1f} ms')

                # Transiciones de fase segun el codigo confirmado
                if codigo == 98:
                    self.fase = 'escaneo'
                    self.get_logger().info('--- Fase: ESCANEO ---')
                elif codigo == 99:
                    self.fase = 'resolucion'
                    self.en_resolucion = True
                    self.get_logger().info('--- Fase: RESOLUCION ---')

    # =========================================================================
    #  [2] CINEMATICA: buffer de articulaciones (rad, tiempos monotonicos)
    # =========================================================================
    def cb_joint(self, msg):
        self.contadores['state/joint'] += 1
        if len(msg.position) >= 6 and len(self.buffer_joints) < MAX_MUESTRAS_CINEMATICA:
            self.buffer_joints.append((time.monotonic(), list(msg.position[:6])))

    # =========================================================================
    #  [3] DISTANCIA TCP: integracion de la trayectoria (state/pose en metros)
    # =========================================================================
    def cb_pose(self, msg):
        self.contadores['state/pose'] += 1
        p = (msg.position.x, msg.position.y, msg.position.z)
        if self.pos_tcp_anterior is not None:
            d = math.dist(p, self.pos_tcp_anterior)
            # Filtro anti-ruido: ignoramos micro-variaciones < 0.1 mm
            if d > 1e-4:
                self.distancia_total_m += d
                if self.en_resolucion:
                    self.distancia_resolucion_m += d
        self.pos_tcp_anterior = p

    # =========================================================================
    #  Fin de sesion: robot.py publica -1.0 en scan_state al terminar
    # =========================================================================
    def cb_scan_state(self, msg):
        self.contadores['scan_state'] += 1
        if int(msg.data) == -1:
            self.get_logger().info('Robot notifica fin de sesion. Generando reporte...')
            self.generar_reporte('fin de resolucion')

    # =========================================================================
    #  [5] SALUD: frecuencia real de cada topic en ventanas de 5 s
    # =========================================================================
    def cb_ventana_salud(self):
        ahora = time.monotonic()
        dt = ahora - self.t_ventana
        if dt <= 0:
            return
        hz = {t: round(c / dt, 1) for t, c in self.contadores.items()}
        self.historial_salud.append((round(ahora - self.t_inicio, 1), hz))
        self.contadores = {t: 0 for t in self.contadores}
        self.t_ventana = ahora
        self.get_logger().info(
            f"[SALUD] cmd={hz['command/data']:>6} Hz | state={hz['state/data']:>6} Hz | "
            f"joint={hz['state/joint']:>6} Hz | pose={hz['state/pose']:>6} Hz")

    # =========================================================================
    #  CALCULO DE CINEMATICA DERIVADA (al generar el reporte)
    # =========================================================================
    def _calcular_cinematica(self):
        """Deriva velocidad, aceleracion y jerk por eje (en grados) a partir
        del buffer. Suaviza con media movil para atenuar el ruido numerico."""
        if len(self.buffer_joints) < 20:
            return None, []

        t = [m[0] - self.buffer_joints[0][0] for m in self.buffer_joints]
        ejes_deg = [[math.degrees(m[1][j]) for m in self.buffer_joints] for j in range(6)]

        def suavizar(serie, ventana=5):
            medio = ventana // 2
            return [sum(serie[max(0, i - medio):i + medio + 1]) /
                    len(serie[max(0, i - medio):i + medio + 1])
                    for i in range(len(serie))]

        def derivar(serie, tiempos):
            out = [0.0]
            for i in range(1, len(serie)):
                dt = tiempos[i] - tiempos[i - 1]
                out.append((serie[i] - serie[i - 1]) / dt if dt > 1e-6 else 0.0)
            return out

        filas_csv = []       # para cinematica.csv: t + pos + vel por eje
        resumen_ejes = []    # para el informe: maximos por eje
        vels, acels = [], []
        for j in range(6):
            pos_s = suavizar(ejes_deg[j])
            v = suavizar(derivar(pos_s, t))
            a = suavizar(derivar(v, t))
            jerk = derivar(a, t)
            vels.append(v)
            acels.append(a)
            resumen_ejes.append({
                'eje': j + 1,
                'vel_max_deg_s': round(max(abs(x) for x in v), 2),
                'acel_max_deg_s2': round(max(abs(x) for x in a), 2),
                'jerk_max_deg_s3': round(max(abs(x) for x in jerk), 2),
            })

        for i in range(len(t)):
            filas_csv.append([round(t[i], 4)]
                             + [round(ejes_deg[j][i], 3) for j in range(6)]
                             + [round(vels[j][i], 3) for j in range(6)])
        return resumen_ejes, filas_csv

    # =========================================================================
    #  GENERACION DEL REPORTE
    # =========================================================================
    def generar_reporte(self, razon=''):
        if self.reporte_generado:
            return
        self.reporte_generado = True

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        carpeta = os.path.expanduser(f'~/rubik_ros2_ws/historial/metricas/metricas_{ts}')
        os.makedirs(carpeta, exist_ok=True)

        # --- [1] Latencias ---
        lat_ms = [l[3] * 1000 for l in self.latencias]
        with open(os.path.join(carpeta, 'latencias.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['indice', 'codigo', 'comando', 'fase', 'latencia_ms'])
            for idx, cod, fase, lat in self.latencias:
                w.writerow([idx, cod, NOMBRES_CODIGO.get(cod, f'giro_{cod}'),
                            fase, round(lat * 1000, 2)])

        # --- [2] Cinematica ---
        resumen_ejes, filas_cin = self._calcular_cinematica()
        if filas_cin:
            with open(os.path.join(carpeta, 'cinematica.csv'), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['t_s'] + [f'pos_eje{j}_deg' for j in range(1, 7)]
                           + [f'vel_eje{j}_deg_s' for j in range(1, 7)])
                w.writerows(filas_cin)

        # --- [5] Salud ---
        with open(os.path.join(carpeta, 'salud_topics.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            topics = list(self.contadores.keys())
            w.writerow(['t_rel_s'] + [f'hz_{t.replace("/", "_")}' for t in topics])
            for t_rel, hz in self.historial_salud:
                w.writerow([t_rel] + [hz.get(t, 0) for t in topics])

        # --- JSON maquina ---
        datos = {
            'fecha': ts, 'razon_cierre': razon,
            'duracion_sesion_s': round(time.monotonic() - self.t_inicio, 1),
            'latencia_ms': {
                'n_comandos': len(lat_ms),
                'media': round(sum(lat_ms) / len(lat_ms), 2) if lat_ms else None,
                'minima': round(min(lat_ms), 2) if lat_ms else None,
                'maxima': round(max(lat_ms), 2) if lat_ms else None,
            },
            'distancia_tcp_m': {'total_sesion': round(self.distancia_total_m, 3),
                                'solo_resolucion': round(self.distancia_resolucion_m, 3)},
            'cinematica_por_eje': resumen_ejes,
            'muestras_cinematica': len(self.buffer_joints),
        }
        datos['carpeta'] = carpeta
        with open(os.path.join(carpeta, 'metricas.json'), 'w') as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)

        # Copia "latest" para que main.py pueda anexar estas metricas al
        # reporte del historial de la ejecucion en curso.
        with open(os.path.expanduser('~/rubik_ros2_ws/historial/metricas/latest_metricas.json'), 'w') as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)

        # --- Resumen legible ---
        with open(os.path.join(carpeta, 'resumen.txt'), 'w') as f:
            f.write('==========================================================\n')
            f.write('        REPORTE DE METRICAS - TFG RUBIK GOFA\n')
            f.write('==========================================================\n')
            f.write(f'Fecha: {ts}   |   Cierre: {razon}\n')
            f.write(f'Duracion de la sesion: {datos["duracion_sesion_s"]} s\n\n')
            f.write('--- [1] LATENCIA DE HANDSHAKE EGM ---\n')
            if lat_ms:
                f.write(f'  Comandos medidos: {len(lat_ms)}\n')
                f.write(f'  Media: {datos["latencia_ms"]["media"]} ms | '
                        f'Min: {datos["latencia_ms"]["minima"]} ms | '
                        f'Max: {datos["latencia_ms"]["maxima"]} ms\n')
            else:
                f.write('  (sin datos)\n')
            f.write('\n--- [3] DISTANCIA RECORRIDA POR EL TCP ---\n')
            f.write(f'  Total de la sesion: {datos["distancia_tcp_m"]["total_sesion"]} m\n')
            f.write(f'  Solo resolucion:    {datos["distancia_tcp_m"]["solo_resolucion"]} m\n')
            f.write('\n--- [2] CINEMATICA POR EJE (maximos) ---\n')
            if resumen_ejes:
                for e in resumen_ejes:
                    f.write(f'  Eje {e["eje"]}: v_max={e["vel_max_deg_s"]:>8} deg/s | '
                            f'a_max={e["acel_max_deg_s2"]:>9} deg/s2 | '
                            f'jerk_max={e["jerk_max_deg_s3"]:>10} deg/s3\n')
            else:
                f.write('  (muestras insuficientes)\n')
            f.write('\n--- [5] SALUD DE TOPICS ---\n')
            f.write('  Ver salud_topics.csv (frecuencias por ventana de 5 s)\n')

        self.get_logger().info(f'[OK] Reporte de metricas generado en: {carpeta}')


def main(args=None):
    rclpy.init(args=args)
    nodo = NodoMetricas()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        # Si la sesion se corta a mano, volcamos igualmente lo capturado
        nodo.generar_reporte('cierre manual')
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()