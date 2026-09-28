import mpi4py.MPI # Computación paralela, distribuye tareas pesadas en varios núcleos de CPU.
import petsc4py.PETSc # Principalmente para resolver sistemas lineales grandes y dispersos.
import ufl # Interfaz para definir formas débiles y problemas de elementos finitos.
from dolfinx import fem, mesh # Trae desde la biblioteca de código abierto dolfinx los módulos de elementos finitos y mallas.
from dolfinx.io import XDMFFile # Importa el módulo para leer y escribir archivos XDMF, que es un formato de archivo comúnmente utilizado para almacenar datos de simulación en mallas 3d y campos de elementos finitos.
import dolfinx.fem.petsc # Importa la herramienta puente entre dolfinx y PETSc para resolver problemas de elementos finitos.
import numpy as np # Librería fundamental para computación científica en Python, proporciona soporte para arreglos y matrices multidimensionales, así como funciones matemáticas de alto nivel.
import matplotlib.pyplot as plt # Librería para crear gráficos y visualizaciones en Python, especialmente útil para generar gráficos 2D y 3D de datos numéricos. Gráficos rápidos y de alta calidad.

# 1. PARAMETROS FISICOS.
f = 500 # Frecuencia de la onda acústica [Hz].                   
omega = 2.0 * np.pi * f # Frecuancia angular de la onda acústica [rad/s].
c0 = 343.0 # Velocidad del sonido en el aire [m/s].
rho0 = 1.225 # Densidad del aire [kg/m^3].
k = omega / c0 # Número de onda. Cuánto más grande es k, más rápido varía la onda en el espacio.
vin = 1.0 + 0.0j # Velocidad de partícula incidente compleja. Se calcula a partir de la presión incidente y las propiedades del medio (densidad y velocidad del sonido). Es un número complejo que representa la velocidad de oscilación de las partículas del aire debido a la onda acústica.
L_val = 1.0 # Longitud del dominio [m].
Zs = 150.0 + 50.0j # Impedancia superficial de prueba [Pa s / m] (ejemplo complejo)

# 2. MALLA 1D.
domain = mesh.create_interval(
    mpi4py.MPI.COMM_WORLD,
    nx=400, # Número de intervalos (elementos) en la malla 1D. A mayor número de intervalos, mayor resolución y precisión en la simulación. Ese metro de longitud se divide en nx partes iguales.               
    points=[0.0, L_val] # Rango del dominio en el eje x, desde 0 hasta L_val [metros].
)

# 3. ESPACIO DE FUNCIONES.
V = fem.functionspace(domain, ("Lagrange", 1)) # Espacio de funciones de elementos finitos de tipo Lagrange de primer orden (P1) en 1D. Esto significa que la solución se aproximará mediante funciones lineales dentro de cada elemento de la malla.
# Funciones dentro de cada fragmento de la malla (elemento) se aproximan mediante polinomios lineales.
# Dentro de cada uno de los nx elementos, la presión acústica solo puede cambiar en linea recta.

# 4. FUNCIONES DE PRUEBA Y ENSAYO.
p = ufl.TrialFunction(V) # Incognita de presión compleja P(x) que se busca resolver. Mapa de presiones acústicas a lo largo del dominio 1D. 
q = ufl.TestFunction(V) # Función de prueba.

# 5. MARCADO DE FRONTERAS.
fdim = domain.topology.dim - 1 # La dimensión de la frontera 1D es cero. La frontera consiste en los puntos finales del intervalo, y se resta 1 de la dimensión del dominio para obtener la dimensión de la frontera (0).

def boundary_left(x):
    return np.isclose(x[0], 0.0) # La primera coordenada x[0] (eje x), está cerca de 0.0, lo que indica que estamos en el extremo izquierdo del dominio.

def boundary_right(x):
    return np.isclose(x[0], L_val) # La primera coordenada x[0] (eje x), está cerca de L_val, lo que indica que estamos en el extremo derecho del dominio.

# Marcado de ambas fronteras (Izquierda = 1, Derecha = 2) para que ds(1) y ds(2) funcionen correctamente.
facets_left = mesh.locate_entities_boundary(domain, fdim, boundary_left)
facets_right = mesh.locate_entities_boundary(domain, fdim, boundary_right)
facets = np.concatenate([facets_left, facets_right])
facet_markers = np.concatenate([np.full_like(facets_left, 1), np.full_like(facets_right, 2)])
sorted_facets = np.argsort(facets)
tagged_facets = mesh.meshtags(domain, fdim, facets[sorted_facets], facet_markers[sorted_facets])

bcs_list = [] # Sin condiciones de Dirichlet en los extremos; todo se maneja mediante condiciones naturales (Neumann/Robin).

# 6. MEDIDAS DE INTEGRACIÓN.
dx = ufl.Measure("dx", domain=domain) # Define la integral sobre toda la longitud del dominio. 
# Calcula la integral sumando el aporte de cada elemento (nx) de la malla.
# UFL es una libreria de dolfinx que permite definir integrales y formas débiles de manera simbólica.

ds = ufl.Measure("ds", domain=domain, subdomain_data=tagged_facets) # Define la integral sobre la frontera del dominio.
# Se le da la etiqueta 2 a la frontera derecha (x=L) para poder aplicar la condición de Sommerfeld o Neumann allí.
# El Measure define la region geométrica sobre la cual se va a integrar, ya sea el dominio completo (dx) o la frontera (ds).

# 7. FORMA DEBIL SEGUN LA CONDICION ELEGIDA.
k_cuadrado = fem.Constant(domain, np.complex128(k**2))
# Con fem.constant se disfraza el número complejo como una constante que puede ser utilizada en el dominio de la simulación (fácil de leer por dolfinx).
robin_coef = 1j * (omega * rho0 / Zs)
neumann_coef = 1j * (omega * rho0 * vin) 

from dolfinx.fem.petsc import LinearProblem

# 8. SE CREA EL CONTENEDOR DE LA SOLUCION FINAL Y DICCIONARIOS
jk_const_aux = fem.Constant(domain, np.complex128(1j * k)) # Se crea la constante compleja j*k para la condición de Sommerfeld.
robin_const_aux = fem.Constant(domain, np.complex128(robin_coef))
neumann_const_aux = fem.Constant(domain, np.complex128(neumann_coef)) 

# Lado derecho corregido con ufl.inner para satisfacer los requisitos del modo complejo de FEniCSx en los bordes.
L_rhs = ufl.inner(neumann_const_aux, q) * ds(1)

# Definimos las variantes físicas del lado izquierdo de la ecuación débil según la condición de frontera elegida en la derecha.
formas = {
    "Neumann - Neumann (pared rígida)": (ufl.inner(ufl.grad(p), ufl.grad(q)) - k_cuadrado * ufl.inner(p, q)) * dx,
    "Neumann - Sommerfeld": (ufl.inner(ufl.grad(p), ufl.grad(q)) - k_cuadrado * ufl.inner(p, q)) * dx + jk_const_aux * ufl.inner(p, q) * ds(2),
    "Neumann - Robin": (ufl.inner(ufl.grad(p), ufl.grad(q)) - k_cuadrado * ufl.inner(p, q)) * dx + robin_const_aux * ufl.inner(p, q) * ds(2)
}

resultados_numericos = {}
resultados_analiticos = {}
errores_l2 = {}

# Mapeo espacial continuo para extracción directa, segura y ordenada por eje X (Evita cruces en mallas paralelas).
x_coor_num = domain.geometry.x[:, 0] # El domain.geometry extrae la matriz de coordenadas de los nodos de la malla 1D. La notación [:, 0] selecciona todas las filas (nodos) y la primera columna (coordenada x) de esa matriz, que representa la posición de cada nodo a lo largo del dominio 1D.
indices_ordenados = np.argsort(x_coor_num)
x_coords_ordenadas = x_coor_num[indices_ordenados]

# 9. BUCLE DE RESOLUCIÓN AUTOMÁTICA
for nombre_caso, a_form in formas.items():
    
    # SE CREA EL CONTENEDOR DE LA SOLUCION FINAL
    ph = fem.Function(V, name="presion_1d")
    
    # SOLUCION DEL PROBLEMA LINEAL
    prob = LinearProblem(
        a_form, L_rhs, bcs=bcs_list, u=ph, 
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix=f"sol_aire_{nombre_caso.replace(' ', '_')}"
    )
    prob.solve()
    
    # Extraemos y ordenamos los resultados numéricos para que Matplotlib no dibuje en zig-zag
    p_num = ph.x.array[indices_ordenados]
    resultados_numericos[nombre_caso] = p_num

# 10. SOLUCION ANALITICA DE REFERENCIA CORRESPONDIENTE
    if nombre_caso == "Neumann - Neumann (pared rígida)":
        # P(x) = Vin*w*rho0/k *[(e^-jk(L-x) + e^-jk(L-x))/(e^jkL - e^-jkL)] 
        p_an = vin*omega*rho0/k * (np.exp(1j * k * (L_val-x_coords_ordenadas)) + np.exp(-1j * k * (L_val - x_coords_ordenadas))) / (np.exp(1j * k * L_val) - np.exp(-1j * k * L_val))
        
    elif nombre_caso == "Neumann - Sommerfeld": 
        # P(x)=vin*w*rho0/k * (e^-jkx)
        p_an = vin*omega*rho0/k * (np.exp(-1j * k * x_coords_ordenadas))
        
    elif nombre_caso == "Neumann - Robin":
        # P(x) = vin*w*rho0 * [(e^-jk(x-L_val)*(k+w*rho0/Zs) + e^jk(L_val-x)*(k-w*rho0/Zs)) / k*[(e^jkL_val*(k+w*rho0/Zs) - e^-jkL_val*(k-w*rho0/Zs))]
        p_an = vin*omega*rho0 * ((np.exp(-1j * k * (x_coords_ordenadas - L_val)) * (k + omega * rho0 / Zs) + np.exp(1j * k * (x_coords_ordenadas-L_val)) * (k - omega * rho0 / Zs)) / (k * (np.exp(1j * k * L_val) * (k + omega * rho0 / Zs) - np.exp(-1j * k * L_val) * (k - omega * rho0 / Zs))))
        
    resultados_analiticos[nombre_caso] = p_an

    # Cálculo del error cuantitativo L2
    error_l2_caso = (np.linalg.norm(p_num - p_an) / np.linalg.norm(p_an)) * 100.0 # Calcula el error relativo porcentual en magnitud de todas las diferencias de presión entre la solución numérica (ph.x.array) y la solución analítica (p_analitica) en todos los nodos de la malla. 
    error_max_abs = np.max(np.abs(p_num - p_an)) # Calcula la máxima desviación física absoluta en Pascales en cualquier nodo del dominio.
    # El ph ya contiene la solución numérica de presión acústica obtenida con DOLFINx gracias al resolver y al v=v.
    # np.linalg.norm calcula el error cuadratico total (L2) de todas las diferencias de presión en todos los nodos de la malla.
    errores_l2[nombre_caso] = {"rel_pct": error_l2_caso, "max_abs": error_max_abs}
    print(f"Error Relativo L2 [{nombre_caso} Aire]: {error_l2_caso:.6f}%")
    print(f"Error Absoluto Máximo [{nombre_caso} Aire]: {error_max_abs:.5e} Pa")

print("¡Simulación 1D resuelta con éxito en DOLFINx (modo complejo y limpio)! ")

# 11. GRAFICA COMPARATIVA TRIPLE SIMULTANEA.
fig, axs = plt.subplots(3, 1, figsize=(9, 8)) # Crea el lienzo o ventana donde se dibujará la gráfica. El tamaño de la ventana es de 9 pulgadas de ancho y 8 pulgadas de alto.

for idx, nombre_caso in enumerate(formas.keys()):
    ax = axs[idx]
    
    # Dibuja la curva numérica (azul con círculos)
    ax.plot(x_coords_ordenadas, resultados_numericos[nombre_caso].real, 'bo-', label="Numérico (FEniCSx)", markersize=4, markevery=4)
    
    # Dibuja la curva analítica (roja segmentada) encima
    ax.plot(x_coords_ordenadas, resultados_analiticos[nombre_caso].real, 'r--', label="Analítico Exacto", linewidth=2)
    
    # Decoraciones rápidas
    ax.set_ylabel("Presión Real [Pa]")
    ax.set_title(f"Caso: {nombre_caso} (Err Rel: {errores_l2[nombre_caso]['rel_pct']:.4f}% | Err Máx Abs: {errores_l2[nombre_caso]['max_abs']:.3e} Pa)", fontsize=9)
    ax.legend(loc="upper right")
    ax.grid(True)

# Decoraciones rápidas para entender el gráfico
axs[2].set_xlabel("Posición en el tubo x [m]")
plt.tight_layout()

# ¡Muestra la ventana interactiva en la pantalla!
plt.show()