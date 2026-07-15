# =============================================================================
#  RUBIK.LAUNCH.PY - ARRANQUE COMPLETO DEL SISTEMA CON UN SOLO COMANDO
# =============================================================================
#  Uso:
#    ros2 launch rubik_control rubik.launch.py
#        -> abre las ventanas y el modo V/R se elige ESCRIBIENDO en los menus
#           de las ventanas T2 (robot) y T3 (orquestador), como siempre.
#           Gracias al QoS TRANSIENT_LOCAL de los comandos, da igual el orden
#           en que respondas los dos menus.
#    ros2 launch rubik_control rubik.launch.py modo:=v   (atajo sin menus)
#    ros2 launch rubik_control rubik.launch.py metricas:=false  (sin nodo de metricas)
#    ros2 launch rubik_control rubik.launch.py terminales:=false (todo en una terminal;
#           en este caso es obligatorio modo:=v/r porque no hay menus interactivos)
#
#  Con terminales:=true (por defecto) cada nodo se abre en su propia ventana
#  xterm, manteniendo la vista por modulo de siempre. Requiere tener xterm:
#    sudo apt install -y xterm
#
#  Orden de arranque (escalonado con TimerAction):
#    t=0s  driver EGM      (debe estar escuchando antes que nadie)
#    t=2s  robot_node + metricas
#    t=6s  orquestador     (publica el scramble ~1s tras arrancar; el robot
#                           ya lleva 4s suscrito y no se pierde el mensaje)
# =============================================================================

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, OpaqueFunction
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generar_nodos(context, *args, **kwargs):
    modo = LaunchConfiguration('modo').perform(context).strip().lower()
    usar_terminales = LaunchConfiguration('terminales').perform(context).strip().lower() == 'true'
    con_metricas = LaunchConfiguration('metricas').perform(context).strip().lower() == 'true'

    # Ruta al yaml de parametros del driver (instalado en el share del paquete)
    params_driver = os.path.join(
        get_package_share_directory('abb_egm_driver'), 'crb-15000-5.yaml')

    def prefijo(titulo):
        """Cada nodo en su propia ventana xterm (se conserva la vista por
        modulo). -hold mantiene la ventana abierta al terminar el nodo."""
        if not usar_terminales:
            return None
        return f'xterm -T "{titulo}" -bg black -fg white -fa Monospace -fs 11 -geometry 110x30 -hold -e'

    nodo_driver = Node(
        package='abb_egm_driver',
        executable='egm_driver',
        name='abb_egm_driver',
        parameters=[params_driver],
        prefix=prefijo('T1 - DRIVER EGM'),
        output='screen',
    )

    # Si no se pasa modo:=v/r, NO se pasa argumento y cada nodo muestra su
    # menu interactivo V/R en su propia ventana (comportamiento clasico).
    args_modo = ['--modo', modo] if modo in ('v', 'r', 'virtual', 'real') else []

    nodo_robot = Node(
        package='rubik_control',
        executable='robot_node',
        name='gofa_controller_node',
        arguments=args_modo,
        prefix=prefijo('T2 - ROBOT GOFA'),
        output='screen',
    )

    nodo_orquestador = Node(
        package='rubik_control',
        executable='orquestador',
        name='rubik_orchestrator',
        arguments=args_modo,
        prefix=prefijo('T3 - ORQUESTADOR'),
        output='screen',
    )

    acciones = [
        nodo_driver,
        TimerAction(period=2.0, actions=[nodo_robot]),
        TimerAction(period=6.0, actions=[nodo_orquestador]),
    ]

    if con_metricas:
        nodo_metricas = Node(
            package='rubik_control',
            executable='metricas',
            name='rubik_metricas',
            prefix=prefijo('T4 - METRICAS'),
            output='screen',
        )
        acciones.append(TimerAction(period=2.0, actions=[nodo_metricas]))

    return acciones


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('modo', default_value='',
                              description='Vacio = menu interactivo V/R en cada ventana. v/r = modo fijado sin menus'),
        DeclareLaunchArgument('terminales', default_value='true',
                              description='true = una ventana xterm por nodo; false = todo en esta terminal'),
        DeclareLaunchArgument('metricas', default_value='true',
                              description='true = lanzar tambien el nodo observador de metricas'),
        OpaqueFunction(function=generar_nodos),
    ])