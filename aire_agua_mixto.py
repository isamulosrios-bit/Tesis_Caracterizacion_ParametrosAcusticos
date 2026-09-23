import shutil #Librería para mover, copiar, borrar carpetas y archivos potentes.
from pathlib import Path # Librería stándar de Python para manipular rutas de archivos y directorios de manera independiente del sistema operativo.

# 1. LIMPIAR CACHE DE FENICSX PARA EVITAR CONFLICTOS DE VERSIONES.
cache_dir = Path.home() / ".cache" / "fenics" # Construye una ruta directa hacia el directorio de caché de FEniCSX en el sistema del usuario.
if cache_dir.exists():
    shutil.rmtree(cache_dir) # Elimina por completo el directorio de caché para evitar conflictos de versiones y asegurar que se utilicen las últimas configuraciones y compilaciones.

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem

def resolver_sistema_completo(): # Se define una función donde todos los parámetros físicos y de malla se encapsulan para su uso en la resolución del sistema completo.

#2. PARÁMETROS FÍSICOS Y CONFIGURACIÓN. 
    f = 500.0                      
    omega = 2.0 * np.pi * f

    # Medio 1: Aire (0.0 a 0.5 metros)
    rho1 = 1.225
    c1 = 343.0
    k1 = omega / c1

    # Medio 2: Agua (0.5 a 1.0 metros)
    rho2 = 1000.0
    c2 = 1480.0
    k2 = omega / c2

    pin = 1.0 + 0.0j      
    x_interface = 0.5
    L_val = 1.0
    nx_mesh = 400  # Malla numérica (nx elementos)
    
    resultados_analiticos = {} # Diccionario para almacenar resultados analíticos exactos para cada caso de frontera.

#3. CÁLCULO DE LA SOLUCIÓN ANALÍTICA EXACTA (Simbólica Matricial).
    for usar_sommerfeld in [False, True]: # Se usa el for para calcular los valores en base al if y al else de la frontera derecha, ya que se tienen dos casos: Pared rígida y Sommerfeld. Itera dos casos de frontera para obtener soluciones analíticas exactas para cada uno.
        M = np.zeros((4, 4), dtype=complex) # Matriz 4x4 llena de ceros complejos para el sistema de ecuaciones lineales que representa las condiciones de frontera y continuidad en la interfaz.
        b_vec = np.zeros(4, dtype=complex) # Vector del lado derecho del sistema de ecuaciones lineales, inicializado con ceros complejos.

        # Frontera izquierda x = 0 (Presión fija)
        M[0, 0] = 1.0 # Fila 0, columna 0 es uno.
        M[0, 1] = 1.0 # Fila 0, columna 1 es uno.
        b_vec[0] = pin # Resultado conocido al que se iguala la presión en la frontera izquierda. A1+B1 = pin
        # Esto es la primera fila de la matriz M y el primer elemento del vector b_vec.
        
        # Continuidad de la presión en la interfaz x = 0.5 (p1=p2)
        M[1, 0] = np.exp(-1j * k1 * x_interface)
        M[1, 1] = np.exp(1j * k1 * x_interface)
        M[1, 2] = -np.exp(-1j * k2 * x_interface)
        M[1, 3] = -np.exp(1j * k2 * x_interface)
        # Esto forma parte de la segunda fila de la matriz M.

        # Continuidad del flujo (1/rho * dp/dx) en la interfaz x = 0.5
        M[2, 0] = (-1j * k1 / rho1) * np.exp(-1j * k1 * x_interface)
        M[2, 1] = (1j * k1 / rho1) * np.exp(1j * k1 * x_interface)
        M[2, 2] = (1j * k2 / rho2) * np.exp(-1j * k2 * x_interface)
        M[2, 3] = (-1j * k2 / rho2) * np.exp(1j * k2 * x_interface)
        # Esto forma parte de la tercera fila de la matriz M.

        # Frontera derecha x = L (Condiciones Homogéneas estándar)
        if not usar_sommerfeld:
            # Pared rígida (dp/dx = 0)
            M[3, 2] = -1j * k2 * np.exp(-1j * k2 * L_val)
            M[3, 3] = 1j * k2 * np.exp(1j * k2 * L_val)
            nombre_caso = "Pared Rígida"
        else:
            # Sommerfeld: dp/dx + 1j*k2*p = 0
            M[3, 2] = 0.0
            M[3, 3] = 1.0 #(-2j * k2 * np.exp(-1j * k2 * L_val)
            nombre_caso = "Sommerfeld"
            
        # Con condición o sin condición sommerfeld, se completa la cuarta fila de la matriz M y se define el nombre del caso para su posterior uso en el diccionario de resultados.

        # NumPy resuelve el sistema de ecuaciones para despejar amplitudes de onda.
        A1, B1, A2, B2 = np.linalg.solve(M, b_vec)

        # Malla analítica punto a punto idéntica para validación estricta
        x_coords_malla = np.sort(np.linspace(0.0, L_val, nx_mesh + 1)) # Se crean los puntos del dominio de 0 a L_val, con nx_mesh + 1 puntos para incluir ambos extremos.
        # .sort hace que ordene los puntos de menor a mayor, aunque en este caso ya están ordenados por la forma en que se generan.
        
        p_analitica = np.zeros_like(x_coords_malla, dtype=complex) # Se crea un vector de ceros complejos del mismo tamaño que x_coords_malla para almacenar la presión analítica en cada punto de la malla.
        
        for i, x in enumerate(x_coords_malla):
            if x <= x_interface:
                p_analitica[i] = A1 * np.exp(-1j * k1 * x) + B1 * np.exp(1j * k1 * x)
            else:
                p_analitica[i] = A2 * np.exp(-1j * k2 * x) + B2 * np.exp(1j * k2 * x)
        # Bucle que recorre cada punto de la malla y calcula la presión analítica según la región (aire o agua) usando las amplitudes de onda resueltas previamente.

        resultados_analiticos[nombre_caso] = p_analitica

    #3. CREACIÓN DE MALLA Y SUBDOMINIOS EN DOLFINx.
    domain = mesh.create_interval(MPI.COMM_WORLD, nx=nx_mesh, points=[0.0, L_val])

    # Identificar celdas de ambos fluidos
    cells_air = mesh.locate_entities(domain, domain.topology.dim, lambda x: x[0] <= x_interface)
    cells_water = mesh.locate_entities(domain, domain.topology.dim, lambda x: x[0] >= x_interface)
    # el domain.topology.dim devuelve la dimensión del dominio (1D en este caso), busca las celdas (segmento de rectas).
    
    subdomain_tags = mesh.meshtags(
        domain, domain.topology.dim, 
        np.hstack([cells_air, cells_water]), 
        np.hstack([np.full_like(cells_air, 1), np.full_like(cells_water, 2)])
    )
    # Numpy con el full llena de 1 y 2 para etiquetar las celdas de aire y agua respectivamente, y hstack concatena los arrays de celdas y etiquetas (las pega una al lado de otra)
    # El subdomain a la larga es el mismo domain pero etiquetado e identificado para poder aplicar condiciones de frontera y propiedades físicas diferentes en cada subdominio.
    dx = ufl.Measure("dx", domain=domain, subdomain_data=subdomain_tags)
    # Se crea un objeto de medida dx que permite integrar sobre los subdominios etiquetados, facilitando la formulación de las ecuaciones débiles en cada región del dominio.

    # Identificación geométrica de los bordes extremos
    facets_x0 = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, lambda x: np.isclose(x[0], 0.0))
    facets_x1 = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, lambda x: np.isclose(x[0], L_val))
    # Numpy con el isclose identifica los bordes usando una tolerancia para evitar problemas de precisión numérica, y locate_entities_boundary encuentra las entidades de borde (facetas) en los extremos del dominio.
    
    boundary_tags = mesh.meshtags(
        domain, domain.topology.dim - 1, 
        np.hstack([facets_x0, facets_x1]), 
        np.hstack([np.full_like(facets_x0, 1), np.full_like(facets_x1, 2)])
    )
    ds = ufl.Measure("ds", domain=domain, subdomain_data=boundary_tags)
    # Se crea un objeto de medida ds que permite integrar sobre los bordes etiquetados, facilitando la aplicación de condiciones de frontera en cada extremo del dominio.


    #4. CONFIGURACIÓN MATEMÁTICA EN ELEMENTOS FINITOS.
    V = fem.functionspace(domain, ("Lagrange", 1)) # Espacio de funciones.
    # Lagrange de primer orden (lineal), las funciones serán polinomios continuos.
    # La función solo cambia en linea recta entre los nodos, y es continua en todo el dominio.
    dofs_x0 = fem.locate_dofs_topological(V, domain.topology.dim - 1, facets_x0)
    # Se localizan los grados de libertad (dofs) en el borde izquierdo (x=0) en el espacio de funciones v. Gracias a facets_x0, se sabe dónde están los nodos en el borde izquierdo y se obtienen los índices de los dofs correspondientes.
    bc = fem.dirichletbc(PETSc.ScalarType(pin), dofs_x0, V)
    # Teniendo estos grados de libertad localizados, se aplica la condición de frontera de Dirichlet (presión fija) en el borde izquierdo del dominio, estableciendo la presión en pin.
    p = ufl.TrialFunction(V) # Función incognita (presión) que se va a resolver en el espacio de funciones V.
    q = ufl.TestFunction(V) # Función de prueba para multiplicar la ecuación diferencial e integrarla por partes, también en el espacio de funciones V.

    # Definición de constantes físicas en FEM (disfrazadas de constantes para que PETSc las maneje eficientemente).
    inv_rho1 = fem.Constant(domain, PETSc.ScalarType(1.0 / rho1))
    inv_rho2 = fem.Constant(domain, PETSc.ScalarType(1.0 / rho2))
    k1_sq_rho1 = fem.Constant(domain, PETSc.ScalarType(k1**2 / rho1))
    k2_sq_rho2 = fem.Constant(domain, PETSc.ScalarType(k2**2 / rho2))

    a_base = (
        inv_rho1 * ufl.inner(ufl.grad(p), ufl.grad(q)) * dx(1) +
        inv_rho2 * ufl.inner(ufl.grad(p), ufl.grad(q)) * dx(2) -
        k1_sq_rho1 * ufl.inner(p, q) * dx(1) -
        k2_sq_rho2 * ufl.inner(p, q) * dx(2)
    )
    
    # Lado derecho limpio igual a cero
    zero_val = fem.Constant(domain, PETSc.ScalarType(0.0)) # Se define una constante cero para el lado derecho de la ecuación débil, representando la ausencia de fuentes externas en el dominio (disfraza el vacío).
    L_form = ufl.inner(zero_val, q) * dx # integral del lado derecho de la ecuación débil, que es cero en todo el dominio, lo que significa que no hay contribuciones externas a la presión.

    coef_sommerfeld = fem.Constant(domain, PETSc.ScalarType(1j * k2 / rho2)) # Se define la constante compleja para la condición de Sommerfeld en el borde derecho, que se usará en la formulación débil para imponer la condición de radiación.
    
    formas = {
        "Pared Rígida": a_base,
        "Sommerfeld": a_base + coef_sommerfeld * ufl.inner(p, q) * ds(2)
    }

    resultados_numericos = {}
    errores_relativos = {}

    # Mapeo espacial continuo para extracción directa segura ordenada por eje X
    x_coor_num = domain.geometry.x[:, 0] # Extrae las coordenadas X de los nodos de la malla numérica, que se usarán para ordenar los resultados de presión obtenidos del solver.
    indices_ordenados = np.argsort(x_coor_num) #Se ordenan los índices de los nodos según sus coordenadas X para asegurar que los resultados numéricos se comparen correctamente con la solución analítica exacta, que también está ordenada por X (de menor a mayor).

    #5. BUCLE DE RESOLUCIÓN AUTOMÁTICA CON LINEARPROBLEM
    for nombre_caso, a_form in formas.items(): # El nombre del caso dependerá de la definición del a_form, el cual, dependerá del .items, pues hará las dos iteraciones, una para pared rígida y otra para sommerfeld.
        
        # 1. Crear el recipiente vacío para guardar la presión
        ph = fem.Function(V, name="presion_1d")
        
        # 2. Configurar y cocinar el problema en dos líneas gracias a LinearProblem
        prob = LinearProblem(
            a_form, L_form, bcs=[bc], u=ph, 
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
            petsc_options_prefix=f"sol_mixto_{nombre_caso.replace(' ', '_')}"
        )
        # El .replace transforma los espacios en guiones bajos para que PETSc pueda manejar el prefijo de las opciones sin problemas de sintaxis.
        prob.solve() # Se resuelve el sistema lineal usando PETSc, almacenando la solución de presión en la función ph.

        # Extracción directa ordenada geométricamente
        p_num = ph.x.array[indices_ordenados] # Se extrae el vector de presión numérica del objeto ph y se reordena según los índices previamente calculados para que coincida con la malla analítica exacta, asegurando una comparación correcta.
        resultados_numericos[nombre_caso] = p_num # Se guarda la presión numérica ordenada en el diccionario de resultados para cada caso de frontera, permitiendo su posterior análisis y comparación con la solución analítica.

        # Cálculo de métricas contrastado con NumPy analítico
        p_an = resultados_analiticos[nombre_caso]
        error_L2 = np.linalg.norm(p_num - p_an) / np.linalg.norm(p_an) * 100.0
        error_max = np.max(np.abs(p_num - p_an))
        errores_relativos[nombre_caso] = {"L2_pct": error_L2, "max_abs": error_max}

        print(f"\n--- Métricas de Error [LinearProblem - {nombre_caso} Mixto] ---")
        print(f"  Error Relativo L2 (%): {error_L2:.6f}%")
        print(f"  Error Absoluto Máximo:  {error_max:.6e} Pa")

#6. PLOTEO COMPARATIVO DE SUBDMINIOS CON MATPLOTLIB.
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    x_coords_plot = x_coor_num[indices_ordenados] 

    for idx, nombre_caso in enumerate(formas.keys()):
        ax = axs[idx]
        ax.plot(x_coords_plot, resultados_numericos[nombre_caso].real, 'bo-', label="Numérico (LinearProblem)", markersize=3, markevery=4)
        ax.plot(x_coords_plot, resultados_analiticos[nombre_caso].real, 'r--', label="Analítico Exacto (Docente)", linewidth=2)
        
        ax.axvline(x=x_interface, color='purple', linestyle=':', linewidth=2, label="Interfaz Aire/Agua (x=0.5)")
        ax.set_ylabel("Presión Real [Pa]", fontsize=11)
        
        # TÍTULO ACTUALIZADO CON PORCENTAJE RELATIVO Y PASCALES ABSOLUTOS MÁXIMOS:
        ax.set_title(f"Caso Acústico Mixto: {nombre_caso} (Err Rel: {errores_relativos[nombre_caso]['L2_pct']:.4f}% | Err Máx Abs: {errores_relativos[nombre_caso]['max_abs']:.3e} Pa)", fontsize=11)
        
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9, loc="upper right")
        
        ax.text(0.15, ax.get_ylim()[1]*0.4, "AIRE (Medio 1)", fontsize=11, fontweight='bold', color='grey', alpha=0.5)
        ax.text(0.65, ax.get_ylim()[1]*0.4, "AGUA (Medio 2)", fontsize=11, fontweight='bold', color='blue', alpha=0.3)

    axs[1].set_xlabel("Posición a lo largo del tubo x [m]", fontsize=11)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    resolver_sistema_completo()
