# Caracterización de Parámetros Acústicos mediante Elementos Finitos (FEniCSx)

Repositorio oficial para el desarrollo de la tesis de pregrado en ingeniería, enfocado en la simulación numérica y validación analítica de ondas acústicas en medios homogéneos y multicapa utilizando el método de elementos finitos con **FEniCSx (DOLFINx)** en Python.

---

## 📂 Estructura del Repositorio

* `fluido_homogeneo.py` — Simulación acústica unidimensional/multidimensional considerando un único medio (Aire).
* `fluido_mixto.py` — Simulación acústica avanzada en medios multicapa (Aire - Agua) con definición de subdominios e interfaz de acoplamiento.
* `solver_matrices.py` — Módulo de ensamblaje y resolución de matrices para los sistemas algebraicos lineales.
* `Malla/` — Archivos de discretización espacial y geometría de los dominios analizados.

---

## 🔬 Descripción de Modelos y Solvers

### 1. Modelo de Medio Único (`fluido_homogeneo.py`)
* **Propósito físico:** Modela la propagación de ondas acústicas puras a través de un medio homogéneo (Aire).
* **Formulación matemática:** Resuelve la ecuación de Helmholtz gobernante del campo de presiones.
* **Condiciones de contorno:** 
  * Se implementan condiciones de tipo **Dirichlet** en la entrada ($x = 0$) para fijar la excitación o presión inicial ($p_{in}$).
  * Se evalúan condiciones de contorno de **Sommerfeld** (radiación) en el extremo derecho ($x = L$) para simular la salida libre de la onda hacia el exterior.

### 2. Modelo Acústico Mixto / Interfaz Aire-Agua (fluido_mixto.py`)
* **Propósito físico:** Simula el comportamiento de ondas acústicas viajando a través de medios con distintas propiedades físicas y velocidades de propagación (interfaz aire-agua).
* **Formulación matemática:** Se divide el dominio computacional en **subdominios específicos** separados por una **interfaz interna**, aplicando la ecuación de Helmholtz adaptada a las características de impedancia y densidad de cada medio (Aire $\rightarrow$ Medio 1, Agua $\rightarrow$ Medio 2).
* **Condiciones de contorno y acoplamiento:**
  * Condición de **Dirichlet** en la entrada ($x = p_{in}$) para la excitación acústica inicial.
  * Acoplamiento de continuidad de presiones y flujo en la interfaz de los medios.
  * Condición de contorno de **Sommerfeld** en el extremo final ($x = L$).
* **Casos de estudio visualizados:** Las rutinas gráficas permiten contrastar de forma directa el comportamiento del sistema bajo dos configuraciones de frontera en el extremo:
  1. **Neumann (pared rígida):** Reflexión total de la onda sin pérdida hacia el exterior.
  2. **Sin pared rígida (Sommerfeld Mixto):** Transmisión y salida de la onda al exterior, logrando una superposición y validación casi exacta frente a los modelos analíticos docentes (con errores relativos menores al $0.075\%$).

---

## 🛠️ Requisitos e Instalación

Las simulaciones están optimizadas para ejecutarse en un entorno de Python con soporte para cálculo científico avanzado:

* **Python** $\ge$ 3.10
* **FEniCSx (DOLFINx)** == 0.11.0
* NumPy, Matplotlib, SciPy, PETSc4py

Para clonar y configurar el entorno localmente:
```bash
git clone [https://github.com/isamulosrios-bit/Tesis_Caracterizacion_ParametrosAcusticos.git](https://github.com/isamulosrios-bit/Tesis_Caracterizacion_ParametrosAcusticos.git)
cd Tesis_Caracterizacion_ParametrosAcusticos
