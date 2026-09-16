import mpi4py.MPI # Computación paralela, distribuye tareas pesadas en varios núcleos de CPU.
import petsc4py.PETSc # Principalmente para resolver sistemas lineales grandes y dispersos.
import ufl # Interfaz para definir formas débiles y problemas de elementos finitos.
from dolfinx import fem, mesh # Trae desde la biblioteca de código abierto dolfinx los módulos de elementos finitos y mallas.
from dolfinx.io import XDMFFile # Importa el módulo para leer y escribir archivos XDMF, que es un formato de archivo comúnmente utilizado para almacenar datos de simulación en mallas 3d y campos de elementos finitos.
import dolfinx.fem.petsc # Importa la herramienta puente entre dolfinx y PETSc para resolver problemas de elementos finitos.
import numpy as np # Librería fundamental para computación científica en Python, proporciona soporte para arreglos y matrices multidimensionales, así como funciones matemáticas de alto nivel.
import matplotlib.pyplot as plt # Librería para crear gráficos y visualizaciones en Python, especialmente útil para generar gráficos 2D y 3D de datos numéricos. Gráficos rápidos y de alta calidad.

# 1. PARAMETROS FISICOS.
f = 500.0 # Frecuencia de la onda acústica en Hz.                    
omega = 2.0 * np.pi * f # Frecuancia angular de la onda acústica en rad/s.
c0 = 343.0 # Velocidad del sonido en el aire [m/s].
rho0 = 1.225 # Densidad del aire [kg/m^3].
k = omega / c0 # Número de onda. Cuánto más grande es k, más rápido varía la onda en el espacio.
pin = 1.0 + 0.0j # Presión incidente compleja. Es un número complejo, donde el número condensa tanto la amplitud como la fase de la onda mediante la notación de fasores (un formato que simplifica las ecuaciones de onda temporales a espaciales).
L_val = 1.0 # Longitud del dominio [m].

# 2. MALLA 1D.
domain = mesh.create_interval(
    mpi4py.MPI.COMM_WORLD,
    nx=400, # Número de intervalos (elementos) en la malla 1D. A mayor número de intervalos, mayor resolución y precisión en la simulación. Ese metro de longitud se divide en nx partes iguales.               
    points=[0.0, L_val] # Rango del dominio en el eje x, desde 0 hasta L_val metros.
)

# 3. ESPACIO DE FUNCIONES.
V = fem.functionspace(domain, ("Lagrange", 1)) # Espacio de funciones de elementos finitos de tipo Lagrange de primer orden (P1) en 1D. Esto significa que la solución se aproximará mediante funciones lineales dentro de cada elemento de la malla.
# Funciones dentro de cada fragmento de la malla (elemento) se aproximan mediante polinomios lineales.
# Dentro de cada uno de los nx elementos, la presión acústica solo puede cambiar en linea recta.

# 4. FUNCIONES DE PRUEBA Y ENSAYO.
p = ufl.TrialFunction(V) # Incognita de presión compleja p(x) que se busca resolver. Mapa de presiones acústicas a lo largo del dominio 1D. 
q = ufl.TestFunction(V) # Función de prueba.

# 5. MARCADO DE FRONTERAS.
fdim = domain.topology.dim - 1 # La dimensión de la frontera 1D es cero. La frontera consiste en los puntos finales del intervalo, y se resta 1 de la dimensión del dominio para obtener la dimensión de la frontera (0).

def boundary_left(x):
    return np.isclose(x[0], 0.0) # La primera coordenada x[0] (eje x), está cerca de 0.0, lo que indica que estamos en el extremo izquierdo del dominio.

def boundary_right(x):
    return np.isclose(x[0], L_val) # La primera coordenada x[0] (eje x), está cerca de L_val, lo que indica que estamos en el extremo derecho del dominio.

dof_left = fem.locate_dofs_geometrical(V, boundary_left) # Busca en el espacio v todos los puntos (grados de libertad) que cumplen la condición de estar en el extremo izquierdo del dominio (x=0). Devuelve los índices de esos puntos.

# CONDICION DE DIRICHLET EN x = 0: p(0) = pin.
pin_val = fem.Constant(domain, np.complex128(pin)) # Toma el número complejo pin y lo convierte en una constante que puede ser utilizada en el dominio de la simulación (fácil de leer por dolfinx). Esto asegura que la presión en el extremo izquierdo del dominio sea igual a la presión incidente especificada.
bc_left = fem.dirichletbc(value=pin_val, dofs=dof_left, V=V) # Se aplica el valor (constante) únicamente en el extremo izquierdo del dominio (x=0). La presión en ese punto se fija a la presión incidente pin. 
# v=v es el espacio de funciones donde se aplica la condición de frontera.
bcs_list = [bc_left] # Lista de python donde se almacenan todas las condiciones de frontera. 

# MARCAR EL EXTREMO DERECHO x = L PARA LA INTEGRAL DE CONTORNO (ds)
facets_right = mesh.locate_entities_boundary(domain, fdim, boundary_right) # El programa busca en el dominio (malla) usando la dimensión de la frontera (fdim) todos los puntos que cumplen la condición de estar en el extremo derecho del dominio (x=L). Devuelve los índices de esos puntos.
facet_markers = np.full_like(facets_right, 2) # Numpy crea una lista del mismo tamaño que facets_right, donde todos los elementos se inicializan con el valor 2. 
tagged_facets = mesh.meshtags(domain, fdim, facets_right, facet_markers) # Toma el dominio (malla), va a la dimension de la frontera (fdim), busca los puntos que cumplen la condición de estar en el extremo derecho del dominio (facets_right) y les asigna el valor 2 (facet_markers).  


# 6. MEDIDAS DE INTEGRACIÓN.
dx = ufl.Measure("dx", domain=domain) # Define la integral sobre toda la longitud del dominio. 
# Calcula la integral sumando el aporte de cada elemento (nx) de la malla.
# UFL es una libreria de dolfinx que permite definir integrales y formas débiles de manera simbólica.

ds = ufl.Measure("ds", domain=domain, subdomain_data=tagged_facets) # Define la integral sobre la frontera del dominio.
# Se le da la etiqueta 2 a la frontera derecha (x=L) para poder aplicar la condición de Sommerfeld allí.
# El Measure define la region geométrica sobre la cual se va a integrar, ya sea el dominio completo (dx) o la frontera (ds).

# 7. FORMA DEBIL SEGUN LA CONDICION ELEGIDA.
k_cuadrado = fem.Constant(domain, np.complex128(k**2))
cero_complejo = fem.Constant(domain, np.complex128(0.0 + 0.0j))
# Con fem.constant se disfraza el número complejo como una constante que puede ser utilizada en el dominio de la simulación (fácil de leer por dolfinx).
L_rhs = ufl.inner(cero_complejo, q) * dx #El lado derecho de la ecuación débil es cero, ya que no hay fuentes internas en el dominio. La integral de prueba q se multiplica por cero, lo que significa que no hay contribución de fuentes externas a la presión acústica en el dominio.

from dolfinx.fem.petsc import LinearProblem

#7. SE CREA EL CONTENEDOR DE LA SOLUCION FINAL Y DICCIONARIOS
jk_const_aux = fem.Constant(domain, np.complex128(1j * k)) # Se crea la constante compleja j*k para la condición de Sommerfeld.

# Definimos las dos variantes físicas del lado izquierdo (la ecuación débil).
formas = {
    "Pared Rígida": (ufl.inner(ufl.grad(p), ufl.grad(q)) - k_cuadrado * ufl.inner(p, q)) * dx,
    "Sommerfeld": (ufl.inner(ufl.grad(p), ufl.grad(q)) - k_cuadrado * ufl.inner(p, q)) * dx - jk_const_aux * ufl.inner(p, q) * ds(2)
}

resultados_numericos = {}
resultados_analiticos = {}
errores_l2 = {}

# Mapeo espacial continuo para extracción directa, segura y ordenada por eje X (Evita cruces en mallas paralelas).
x_coor_num = domain.geometry.x[:, 0] # El domain.geometry extrae la matriz de coordenadas de los nodos de la malla 1D. La notación [:, 0] selecciona todas las filas (nodos) y la primera columna (coordenada x) de esa matriz, que representa la posición de cada nodo a lo largo del dominio 1D.
indices_ordenados = np.argsort(x_coor_num)
x_coords_ordenadas = x_coor_num[indices_ordenados]

#8. BUCLE DE RESOLUCIÓN AUTOMÁTICA
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

# 9. SOLUCION ANALITICA DE REFERENCIA CORRESPONDIENTE
    if nombre_caso == "Sommerfeld":
        # P(x) = Po * e^(j*k*x)
        p_an = pin * np.exp(1j * k * x_coords_ordenadas)
    else:
        # P(x) = Po*e^(-jkx) + [Po*e^(-jkL) / (e^(jkL) + e^(-jkL))] * (e^(jkx) - e^(-jkx))
        #termino_1 = pin * np.exp(-1j * k * x_coords)
        #numerador = pin * np.exp(-1j * k * L_val)
        #denominador = np.exp(1j * k * L_val) + np.exp(-1j * k * L_val)
        #corchete = np.exp(1j * k * x_coords) - np.exp(-1j * k * x_coords)
        #p_analitica = termino_1 + (numerador / denominador) * corchete
        p_an = pin * (np.cos(k * (L_val - x_coords_ordenadas)) / np.cos(k * L_val))
    resultados_analiticos[nombre_caso] = p_an

    # Cálculo del error cuantitativo L2
    error_l2_caso = (np.linalg.norm(p_num - p_an) / np.linalg.norm(p_an)) * 100.0 # Calcula el error relativo porcentual en magnitud de todas las diferencias de presión entre la solución numérica (ph.x.array) y la solución analítica (p_analitica) en todos los nodos de la malla. 
    error_max_abs = np.max(np.abs(p_num - p_an)) # Calcula la máxima desviación física absoluta en Pascales en cualquier nodo del dominio.
    # El ph ya contiene la solución numérica de presión acústica obtenida con DOLFINx gracias al resolver y al u=v.
    # np.linalg.norm calcula el error cuadratico total (L2) de todas las diferencias de presión en todos los nodos de la malla.
    errores_l2[nombre_caso] = {"rel_pct": error_l2_caso, "max_abs": error_max_abs}
    print(f"Error Relativo L2 [{nombre_caso} Aire]: {error_l2_caso:.6f}%")
    print(f"Error Absoluto Máximo [{nombre_caso} Aire]: {error_max_abs:.5e} Pa")

print("¡Simulación 1D resuelta con éxito en DOLFINx (modo complejo y limpio)! ")

#10. GRAFICA COMPARATIVA DOBLE SIMULTANEA.
fig, axs = plt.subplots(2, 1, figsize=(9, 5)) # Crea el lienzo o ventana donde se dibujará la gráfica. El tamaño de la ventana es de 9 pulgadas de ancho y 5 pulgadas de alto.

for idx, nombre_caso in enumerate(formas.keys()):
    ax = axs[idx]
    
    # Dibuja la curva numérica (azul con círculos)
    ax.plot(x_coords_ordenadas, resultados_numericos[nombre_caso].real, 'bo-', label="Numérico (FEniCSx)", markersize=4, markevery=4)
    
    # Dibuja la curva analítica (roja segmentada) encima
    ax.plot(x_coords_ordenadas, resultados_analiticos[nombre_caso].real, 'r--', label="Analítico Exacto", linewidth=2)
    
    # Decoraciones rápidas para entender el gráfico
    ax.set_ylabel("Presión Real [Pa]")
    
    # TÍTULO MODIFICADO CON PORCENTAJE Y PASCALES ABSOLUTOS:
    ax.set_title(f"Caso Aire-Aire: {nombre_caso} (Err Rel: {errores_l2[nombre_caso]['rel_pct']:.4f}% | Err Máx Abs: {errores_l2[nombre_caso]['max_abs']:.3e} Pa)", fontsize=10)
    
    ax.legend()
    ax.grid(True)

# Decoraciones rápidas para entender el gráfico
axs[1].set_xlabel("Posición en el tubo x [m]")
plt.tight_layout()

# ¡Muestra la ventana interactiva en la pantalla!
plt.show()

