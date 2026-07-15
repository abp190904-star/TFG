# solver.py
import kociemba

def resolver_cubo(estado_cubo):
    """
    Recibe el string de 54 caracteres y devuelve la solución óptima
    usando el algoritmo de Kociemba (máximo 20 movimientos).
    """
    if not estado_cubo:
        print("[-] Error: No se ha recibido ningún estado del cubo.")
        return None

    print(f"[*] Calculando solución óptima para: {estado_cubo[:15]}...")
    
    try:
        # La magia de Kociemba ocurre aquí
        solucion = kociemba.solve(estado_cubo)
        print(f"[+] ¡Solución encontrada!: {solucion}")
        return solucion
        
    except ValueError as e:
        # Si la cámara (en el futuro) lee un cubo imposible, Kociemba lanzará este error
        print(f"\n[-] ERROR DE KOCIEMBA: El cubo no es válido.")
        print(f"[-] Detalle del error: {e}")
        print("[-] Revisa que los colores se hayan escaneado correctamente.")
        return None
