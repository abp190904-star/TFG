#!/usr/bin/env python3
import sys
import os
import time
import threading
import csv
import numpy as np
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64, Float64MultiArray
from sensor_msgs.msg import JointState

# --- NUEVOS IMPORTS PARA LATENCIA CERO ---
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.clock import Clock, ClockType

DICCIONARIO_RUBIK = {
    'U': 1,  "U'": 2,  'U2': 13,
    'F': 3,  "F'": 4,  'F2': 14,
    'D': 5,  "D'": 6,  'D2': 15,
    'B': 7,  "B'": 8,  'B2': 16,
    'R': 9,  "R'": 10, 'R2': 17,
    'L': 11, "L'": 12, 'L2': 18
}

def _invertir_movimiento(mov):
    if '2' in mov: return mov
    return mov[0] if "'" in mov else mov + "'"

def _preparar_array_scramble(sol_str):
    res = []
    if sol_str:
        for m in sol_str.split(): res.append(DICCIONARIO_RUBIK[m])
    res.append(98) 
    return res

def _preparar_array_solucion(sol_str, modo):
    res = []
    if modo == "real" and sol_str:
        scramble_inverso = " ".join([_invertir_movimiento(m) for m in reversed(sol_str.split())])
        for m in scramble_inverso.split():
            res.append(DICCIONARIO_RUBIK[m])

    res.append(99)
    if sol_str:
        for m in sol_str.split():
            res.append(DICCIONARIO_RUBIK[m])

    res.append(0)
    return res

class GofaControllerNode(Node):
    def __init__(self, modo_ejecucion):
        super().__init__('gofa_controller_node')
        self.modo_ejecucion = modo_ejecucion

        # Pausa entre giros virtuales del mezclado (en segundos). Da tiempo a
        # que los motores del Smart Component del cubo terminen su animacion
        # antes del siguiente pulso. Ajustar aqui si hace falta mas margen.
        self.PAUSA_GIRO_VIRTUAL = 0.5
        self.get_logger().info(f'Iniciando Nodo GOFA (Cliente Driver) - Modo {modo_ejecucion.upper()}')

        # --- PERFIL QoS DE ALTO RENDIMIENTO (CONEXIÓN DRIVER) ---
        qos_tiempo_real = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- PERFIL QoS DEL CANAL DE VISIÓN ---
        # RELIABLE en AMBOS extremos (debe coincidir con el qos_vision de main.py).
        # Regla DDS: la fiabilidad ofrecida por el publisher debe ser >= a la
        # pedida por el subscriber. RELIABLE<->RELIABLE garantiza el match.
        qos_vision = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # --- CONEXIÓN CON EL DRIVER DEL PROFESOR (Aplicando QoS Tiempo Real) ---
        self.egm_data_pub = self.create_publisher(Float64MultiArray, 'command/data', qos_tiempo_real)
        self.egm_data_sub = self.create_subscription(Float64MultiArray, 'state/data', self.egm_data_callback, qos_tiempo_real)
        self.egm_joint_sub = self.create_subscription(JointState, 'state/joint', self.egm_joint_callback, qos_tiempo_real)

        self.latest_data_in = np.zeros(40, dtype=np.float64)
        self.latest_joints = [0.0] * 6
        self.conectado = False

        # --- CONEXIÓN CON EL ORQUESTADOR ---
        # QoS con memoria (TRANSIENT_LOCAL): aunque este nodo se suscriba
        # DESPUES de que el orquestador publique (p.ej. si respondes su menu
        # V/R mas tarde), el ultimo comando publicado se recibe igualmente.
        # Hace irrelevante el orden en que se responden los menus del launch.
        qos_comandos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                  history=HistoryPolicy.KEEP_LAST, depth=10,
                                  durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.sub_movimiento = self.create_subscription(String, 'comando_movimiento', self.movimiento_callback, qos_comandos)
        self.sub_scramble = self.create_subscription(String, 'comando_scramble', self.scramble_callback, qos_comandos)
        
        # APLICAMOS EL QoS DE VISIÓN AQUÍ:
        self.scan_cmd_sub = self.create_subscription(Float64, 'scan_cmd', self.scan_cmd_callback, qos_vision)
        self.scan_state_pub = self.create_publisher(Float64, 'scan_state', qos_vision)

        self.modo_escaneo = True
        self.payload_escaneo = np.zeros(40, dtype=np.float64)

        # Timer con reloj MONOTONICO (STEADY_TIME). El reloj de sistema de WSL2
        # puede pegar saltos hacia atras al resincronizarse; un timer basado en
        # reloj de sistema queda CONGELADO hasta que el reloj "alcanza" de nuevo
        # su proximo instante de disparo. Con STEADY_TIME es inmune a eso.
        self.timer_escaneo = self.create_timer(0.02, self.bucle_escaneo_ros,
                                               clock=Clock(clock_type=ClockType.STEADY_TIME))

    def egm_data_callback(self, msg):
        self.latest_data_in = np.array(msg.data)
        self.conectado = True

    def egm_joint_callback(self, msg):
        self.latest_joints = list(map(math.degrees, msg.position))

    def scan_cmd_callback(self, msg):
        self.payload_escaneo[2] = msg.data

    def bucle_escaneo_ros(self):
        if self.modo_escaneo:
            if len(self.latest_data_in) > 1:
                msg = Float64()
                msg.data = float(self.latest_data_in[1])
                self.scan_state_pub.publish(msg)

            msg_out = Float64MultiArray()
            msg_out.data = self.payload_escaneo.tolist()
            self.egm_data_pub.publish(msg_out)

    def scramble_callback(self, msg):
        threading.Thread(target=self._hilo_scramble, args=(msg.data,)).start()

    def _hilo_scramble(self, solucion_str):
        if not solucion_str:
            self.get_logger().info('Mezclado omitido (Modo Real). Saltando directo al escaneo.')
        else:
            self.get_logger().info(f'Recibida secuencia de mezclado: {solucion_str}')

        self.modo_escaneo = False

        try:
            self.enviar_al_robot(solucion_str, es_scramble=True)

            msg_fin = Float64()
            msg_fin.data = 98.0
            for _ in range(10):
                self.scan_state_pub.publish(msg_fin)
                time.sleep(0.05)

            self.get_logger().info('[*] Reteniendo telemetría 1.5s para asegurar lectura de main.py...')
            time.sleep(1.5)

            self.modo_escaneo = True
        except Exception as e:
            self.get_logger().error(f'Error en mezclado: {e}')

    def movimiento_callback(self, msg):
        threading.Thread(target=self._hilo_movimiento, args=(msg.data,)).start()

    def _hilo_movimiento(self, solucion_str):
        if solucion_str == "ABORTAR":
            self.abortar_robot()
            return

        self.get_logger().info(f'Recibida solución oficial: {solucion_str}')
        self.modo_escaneo = False

        try:
            tiempo, telemetria = self.enviar_al_robot(solucion_str, es_scramble=False)
            if tiempo:
                ruta_csv_temp = os.path.expanduser('~/rubik_ros2_ws/historial/latest_telemetria.csv')
                os.makedirs(os.path.dirname(ruta_csv_temp), exist_ok=True)
                with open(ruta_csv_temp, 'w', newline='') as f_csv:
                    writer = csv.writer(f_csv)
                    writer.writerow(['Tiempo_Relativo_s', 'Eje_1', 'Eje_2', 'Eje_3', 'Eje_4', 'Eje_5', 'Eje_6'])
                    writer.writerows(telemetria)
                self.get_logger().info('[+] Matriz cinemática volcada a CSV temporal con éxito.')

                msg_fin = Float64()
                msg_fin.data = -1.0
                self.scan_state_pub.publish(msg_fin)
                time.sleep(1.0)

                self.get_logger().info('Apagando Nodo limpiamente.')
                os._exit(0)
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

    def enviar_al_robot(self, solucion_str, es_scramble=False):
        array_comandos = _preparar_array_scramble(solucion_str) if es_scramble else _preparar_array_solucion(solucion_str, self.modo_ejecucion)
        self.get_logger().info(f"[*] Total de comandos RAPID a ejecutar: {len(array_comandos)}")

        # Posicion del marcador 99 (inicio de resolucion FISICA). En modo
        # real, todos los comandos ANTERIORES a el son giros VIRTUALES del
        # gemelo digital (reconstruccion del estado escaneado) y necesitan
        # la misma pausa de animacion que el mezclado virtual.
        indice_marcador_99 = (array_comandos.index(99) + 1) if 99 in array_comandos else 0

        handshake_done = False
        indice_actual = 1
        tiempo_inicio_fisico = 0
        historial_cinematico = []
        t_crono_movimiento = None

        while True:
            if not self.conectado:
                time.sleep(0.01)
                continue

            if len(self.latest_data_in) > 0:
                ack_robot = self.latest_data_in[0]

                if not es_scramble and t_crono_movimiento is not None:
                    t_relativo = time.monotonic() - t_crono_movimiento
                    historial_cinematico.append([t_relativo] + self.latest_joints)

                if ack_robot == 999 and not es_scramble:
                    tiempo_fisico = time.monotonic() - tiempo_inicio_fisico
                    self.get_logger().info("\n[OK] ¡El robot confirma que el cubo está resuelto!")
                    return tiempo_fisico, historial_cinematico

                if ack_robot == indice_actual:
                    if es_scramble and array_comandos[indice_actual - 1] == 98:
                        self.get_logger().info("\n[OK] ¡Robot confirma fin de mezclado / salto al escaneo!")
                        return None, None

                    if not es_scramble and array_comandos[indice_actual - 1] == 99:
                        self.get_logger().info("[*] Preparando resolución (Pausa de 3s)...")
                        time.sleep(3.0)
                        tiempo_inicio_fisico = time.monotonic()
                        t_crono_movimiento = time.monotonic()

                    # Pausa para GIROS VIRTUALES del Smart Component:
                    #   - mezclado virtual (es_scramble), y
                    #   - reconstruccion virtual del estado escaneado en modo
                    #     real (comandos previos al marcador 99).
                    # Los movimientos fisicos (post-99) no la necesitan: el
                    # propio brazo marca el ritmo.
                    es_giro_virtual = es_scramble or \
                        (indice_marcador_99 > 0 and indice_actual < indice_marcador_99)
                    if es_giro_virtual:
                        time.sleep(self.PAUSA_GIRO_VIRTUAL)

                    indice_actual += 1

            if not handshake_done:
                self.get_logger().info("[!] Conectado. Handshake con driver EGM establecido...")
                handshake_done = True

            msg_hb = Float64()
            msg_hb.data = -50.0
            self.scan_state_pub.publish(msg_hb)

            payload = np.zeros(40, dtype=np.float64)
            if indice_actual <= len(array_comandos):
                payload[0] = indice_actual
                payload[1] = array_comandos[indice_actual - 1]

            msg_out = Float64MultiArray()
            msg_out.data = payload.tolist()
            self.egm_data_pub.publish(msg_out)

            time.sleep(0.005)

    def abortar_robot(self):
        self.get_logger().warn("[!] Enviando señal de ABORTO al robot...")
        payload = np.zeros(40, dtype=np.float64)
        payload[0] = 1
        payload[1] = 998

        msg_out = Float64MultiArray()
        msg_out.data = payload.tolist()

        tiempo_fin = time.monotonic() + 1.0
        while time.monotonic() < tiempo_fin:
            self.egm_data_pub.publish(msg_out)
            time.sleep(0.02)
        self.get_logger().warn("[!] Señal enviada.")
        os._exit(1)


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
    print("         NODO CONTROLADOR EGM - BRAZO GOFA")
    print("=======================================================")
    print("Selecciona el modo de ejecución:")
    print("  [V] Virtual (RobotStudio Virtual Controller)")
    print("  [R] Real (Controlador ABB Omnicore Físico)")
    print("=======================================================")
    modo_ejecucion = _modo_desde_argv()
    if modo_ejecucion is None:
        seleccion = input("Elige una opción (V/R): ").strip().upper()
        modo_ejecucion = "virtual" if seleccion == 'V' else "real"
    else:
        print(f">> Modo seleccionado por argumento: {modo_ejecucion.upper()}")

    rclpy.init(args=args)
    nodo = GofaControllerNode(modo_ejecucion)
    try:
        rclpy.spin(nodo)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()