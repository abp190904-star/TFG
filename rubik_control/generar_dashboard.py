import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def obtener_ultima_carpeta():
    """Busca la carpeta de historial más reciente."""
    carpetas = glob.glob('historial/*/')
    if not carpetas:
        return None
    # Ordenamos por fecha de modificación y devolvemos la más nueva
    return max(carpetas, key=os.path.getmtime)

def generar_reporte():
    carpeta = obtener_ultima_carpeta()
    if not carpeta:
        print("[!] Error: No se encontró ninguna subcarpeta en 'historial'.")
        return

    ruta_csv = os.path.join(carpeta, 'telemetria_ejes.csv')
    if not os.path.exists(ruta_csv):
        print(f"[!] Error: No hay archivo 'telemetria_ejes.csv' en {carpeta}")
        return

    print(f"[*] Analizando telemetría del último ensayo: {carpeta}")

    # 1. Leer los datos registrados por EGM
    df = pd.read_csv(ruta_csv)
    t = df['tiempo_relativo']

    # 2. Configurar el lienzo del Dashboard (2 gráficas apiladas)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # --- GRÁFICA SUPERIOR: Posición de los 6 Ejes ---
    ejes = ['eje1', 'eje2', 'eje3', 'eje4', 'eje5', 'eje6']
    colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for eje, color in zip(ejes, colores):
        ax1.plot(t, df[eje], label=eje.upper(), color=color, linewidth=1.5)

    ax1.set_title('Cinemática EGM: Trayectoria Articular (Position Profiling)', fontweight='bold')
    ax1.set_ylabel('Grados (º)')
    ax1.legend(loc='upper right', ncol=6)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- GRÁFICA INFERIOR: Perfil de Velocidad del Eje 6 (Muñeca) ---
    # Calculamos la velocidad derivando la posición respecto al tiempo (Velocidad = dPosición / dTiempo)
    dt = np.diff(t)
    dt[dt == 0] = 0.001  # Pequeño filtro de seguridad para evitar divisiones por cero
    
    velocidad_eje6 = np.diff(df['eje6']) / dt
    t_vel = t[:-1] # Ajustamos la matriz de tiempo para que cuadre con la derivada

    ax2.plot(t_vel, velocidad_eje6, color='#8c564b', linewidth=1.5)
    ax2.fill_between(t_vel, velocidad_eje6, alpha=0.3, color='#8c564b')
    ax2.set_title('Perfil de Velocidad: Eje 6 (Rotación del Cubo)', fontweight='bold')
    ax2.set_xlabel('Tiempo Total de Movimiento Físico (Segundos)')
    ax2.set_ylabel('Velocidad (º/s)')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 3. Empaquetar y exportar la imagen
    plt.tight_layout()
    ruta_guardado = os.path.join(carpeta, 'Dashboard_Cinematico.png')
    
    # Guardamos la gráfica en Alta Resolución (300 dpi) ideal para imprimir en la memoria del TFG
    plt.savefig(ruta_guardado, dpi=300)
    print(f"\n[OK] Dashboard generado con éxito.")
    print(f"[+] Archivo guardado en: {ruta_guardado}\n")

if __name__ == '__main__':
    generar_reporte()
