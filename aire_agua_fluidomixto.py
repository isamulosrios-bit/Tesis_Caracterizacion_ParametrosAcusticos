import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# CÓDIGO ANALÍTICO - SISTEMA BICAPA (AIRE - AGUA) EN TUBO DE IMPEDANCIA
# =========================================================================
# Este script modela la solución analítica exacta de un tubo de impedancia 
# con fluido mixto:
# - Medio 1 (Aire): x en [0.0, 0.5]
# - Medio 2 (Agua): x en (0.5, 1.0]
# 
# EXPRESIONES GENERALES DE LOS CAMPOS DE PRESIÓN ANALÍTICOS:
# # P1(x) = A1 * exp(-j * k1 * x) + B1 * exp(j * k1 * x)
# # P2(x) = A2 * exp(-j * k2 * x) + B2 * exp(j * k2 * x)
#
# El script calcula las 4 constantes (A1, B1, A2, B2) resolviendo el sistema 
# matricial de contorno e interfaz, evaluando ambos escenarios de frontera 
# en x = 1.0 (Pared Rígida vs. Condición de Sommerfeld).
# =========================================================================

def resolver_sistema_bicapa_mixto():
    # 1. Parámetros físicos del sistema
    f = 500.0                      
    omega = 2.0 * np.pi * f

    # Medio 1 (Aire)
    rho1 = 1.225
    c1 = 343.0
    k1 = omega / c1

    # Medio 2 (Agua)
    rho2 = 1000.0
    c2 = 1480.0
    k2 = omega / c2

    pin = 1.0 + 0.0j      
    x_interface = 0.5
    L_val = 1.0
    x_coords = np.linspace(0.0, L_val, 500)

    # -------------------------------------------------------------------------
    # ESPACIO RESERVADO PARA EL SCRIPT DE RESOLUCIÓN NUMÉRICA (FEniCSx)
    # -------------------------------------------------------------------------
    # # [INICIO BLOQUE NUMÉRICO - FENICS / BICAPA FLUIDO MIXTO]
    # # domain = mesh.create_interval(mpi4py.MPI.COMM_WORLD, nx=400, points=[0.0, L_val])
    # # V = fem.functionspace(domain, ("Lagrange", 1))
    # # cells_air = mesh.locate_entities(domain, domain.topology.dim, lambda x: x[0] <= x_interface)
    # # cells_water = mesh.locate_entities(domain, domain.topology.dim, lambda x: x[0] >= x_interface)
    # # Definición de la forma débil variacional con coeficientes variables por tramos:
    # # ∫_air (1/rho1 * ∇p·∇q - k1^2/rho1 * p*q) dx + ∫_water (1/rho2 * ∇p·∇q - k2^2/rho2 * p*q) dx = 0
    # # [FIN BLOQUE NUMÉRICO]
    # -------------------------------------------------------------------------

    resultados = {}

    # Iterar sobre las dos condiciones de contorno requeridas en x = 1.0
    for usar_sommerfeld in [False, True]:
        M = np.zeros((4, 4), dtype=complex)
        b_vec = np.zeros(4, dtype=complex)

        # Ecuación 1: Condición de Dirichlet en x = 0 (P(0) = pin -> A1 + B1 = pin)
        M[0, 0] = 1.0
        M[0, 1] = 1.0
        b_vec[0] = pin

        # Ecuación 2: Continuidad de Presión en la Interfaz (x = 0.5)
        # A1*exp(-j*k1*0.5) + B1*exp(j*k1*0.5) - A2*exp(-j*k2*0.5) - B2*exp(j*k2*0.5) = 0
        M[1, 0] = np.exp(-1j * k1 * x_interface)
        M[1, 1] = np.exp(1j * k1 * x_interface)
        M[1, 2] = -np.exp(-1j * k2 * x_interface)
        M[1, 3] = -np.exp(1j * k2 * x_interface)

        # Ecuación 3: Continuidad de Velocidad Normal / Flujo en la Interfaz (x = 0.5)
        # (-j*k1/rho1)*(A1*exp - B1*exp) = (-j*k2/rho2)*A2*exp - (j*k2/rho2)*B2*exp
        M[2, 0] = (-1j * k1 / rho1) * np.exp(-1j * k1 * x_interface)
        M[2, 1] = (1j * k1 / rho1) * np.exp(1j * k1 * x_interface)
        M[2, 2] = (1j * k2 / rho2) * np.exp(-1j * k2 * x_interface)
        M[2, 3] = (-1j * k2 / rho2) * np.exp(1j * k2 * x_interface)

        # Ecuación 4: Condición de Contorno en el Extremo Derecho (x = 1.0)
        if not usar_sommerfeld:
            # Caso 1: Pared Rígida (dP/dx = 0 en x = 1.0)
            M[3, 2] = -1j * k2 * np.exp(-1j * k2 * L_val)
            M[3, 3] = 1j * k2 * np.exp(1j * k2 * L_val)
            nombre_caso = "Pared Rígida"
        else:
            # Caso 2: Condición de Sommerfeld (dP/dx - j*k2*P = 0 en x = 1.0)
            M[3, 2] = (-1j * k2 - 1j * k2) * np.exp(-1j * k2 * L_val)
            M[3, 3] = (1j * k2 - 1j * k2) * np.exp(1j * k2 * L_val)
            nombre_caso = "Sommerfeld"

        # Resolver el sistema lineal 4x4 para obtener las 4 constantes analíticas
        A1, B1, A2, B2 = np.linalg.solve(M, b_vec)

        # Comentarios detallados de las constantes calculadas para el caso activo:
        # # Coeficientes analíticos para {nombre_caso}:
        # # A1 = {A1}
        # # B1 = {B1}
        # # A2 = {A2}
        # # B2 = {B2}

        # Evaluación espacial de la curva analítica P(x)
        p_analitica = np.zeros_like(x_coords, dtype=complex)
        for i, x in enumerate(x_coords):
            if x <= x_interface:
                # Tramo 1: Aire
                # # P1(x) = A1 * exp(-j * k1 * x) + B1 * exp(j * k1 * x)
                p_analitica[i] = A1 * np.exp(-1j * k1 * x) + B1 * np.exp(1j * k1 * x)
            else:
                # Tramo 2: Agua
                # # P2(x) = A2 * exp(-j * k2 * x) + B2 * exp(j * k2 * x)
                p_analitica[i] = A2 * np.exp(-1j * k2 * x) + B2 * np.exp(1j * k2 * x)

        resultados[nombre_caso] = p_analitica

    # 2. Generar gráfica comparativa de ambas soluciones analíticas
    plt.figure(figsize=(10, 5))
    plt.plot(x_coords, resultados["Pared Rígida"].real, 'r-', linewidth=2, label="Analítica - Pared Rígida")
    plt.plot(x_coords, resultados["Sommerfeld"].real, 'b--', linewidth=2, label="Analítica - Sommerfeld")
    plt.axvline(x=x_interface, color='k', linestyle=':', alpha=0.6, label="Interfaz Aire-Agua ($x = 0.5$ m)")
    
    plt.title(f"Solución Analítica - Sistema Bicapa Fluido Mixto ($f = {f}$ Hz)", fontsize=13)
    plt.xlabel("Posición en el tubo $x$ [m]", fontsize=12)
    plt.ylabel("Presión Acústica Real Re{$P(x)$} [Pa]", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()

    nombre_archivo = "curvas_analiticas_bicapa_mixto.png"
    plt.savefig(nombre_archivo, dpi=300)
    print(f"¡Curvas analíticas generadas y guardadas exitosamente como '{nombre_archivo}'!")

if __name__ == "__main__":
    resolver_sistema_bicapa_mixto()