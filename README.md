# Caracterización de Parámetros Acústicos mediante Elementos Finitos (FEniCSx)

Repositorio oficial para el desarrollo de la tesis de pregrado en ingeniería, enfocado en la simulación numérica y validación analítica de ondas acústicas en medios homogéneos y multicapa utilizando el método de elementos finitos con **FEniCSx (DOLFINx)** en Python.

---

## 📂 Estructura del Repositorio

- `fluido_homogeneo01.py` — Simulación en medio homogéneo (Aire) con excitación de presión en el extremo izquierdo ($p(0) = p_{in}$) y evaluación de condiciones de Neumann, Sommerfeld y Robin en el extremo derecho.
- `fluido_homogeneo02.py` — Simulación en medio homogéneo (Aire) con velocidad de partícula impuesta en el extremo izquierdo ($v(0) = v_{in}$) y evaluación de condiciones en el extremo derecho.
- `Fluido_mixto01.py` — Simulación acústica avanzada en medios multicapa (Aire - Agua) con condición de Dirichlet a la izquierda, acoplamiento de continuidad en la interfaz y condiciones en el extremo derecho.
- `fluido_mixto02.py` — Simulación acústica avanzada en medios multicapa con velocidad impuesta (Neumann no homogéneo) a la izquierda y condición de Robin a la derecha (**modelo clave para el método de medio equivalente**).
- `solver_npmatrices_Robin.py` / `solver_npmatrices_sommerfeld.py` — Módulos de ensamblaje y resolución de sistemas algebraicos lineales (incluyendo matrices de $4 \times 4$ para los modelos mixtos).
- `Malla/` — Archivos de discretización espacial y geometría de los dominios analizados.

---

## 🔬 Descripción de Modelos y Solvers

### 1. Modelos de Medio Único (`fluido_homogeneo01.py` y `fluido_homogeneo02.py`)
- **Propósito físico:** Modela la propagación de ondas acústicas puras a través de un único medio (Aire) bajo diferentes esquemas de excitación de frontera.
- **Formulación matemática:** Resolución de la ecuación de Helmholtz gobernante del campo de presiones.
- **Casos implementados:**
  * **`fluido_homogeneo01.py`:** Implementa una condición de contorno de tipo **Dirichlet** en la entrada izquierda ($p(0) = p_{in}$). En el extremo derecho se evalúan y visualizan de forma comparativa tres opciones: Neumann (pared rígida), Sommerfeld y Robin.
  * **`fluido_homogeneo02.py`:** Implementa una condición de contorno de **Neumann no homogéneo** en la entrada izquierda ($v(0) = v_{in}$ / velocidad de partícula). De igual forma, en el extremo derecho contrasta las respuestas bajo condiciones de Neumann, Sommerfeld y Robin.

### 2. Modelos Acústicos Mixtos / Interfaz Aire-Agua (`Fluido_mixto01.py` y `fluido_mixto02.py`)
- **Propósito físico:** Simulan el comportamiento de ondas acústicas viajando a través de medios con distintas propiedades físicas, impedancias y velocidades de propagación (Medio 1: Aire $\rightarrow$ Medio 2: Agua) separados por una interfaz interna.
- **Formulación matemática:** División del dominio computacional en subdominios específicos aplicando la ecuación de Helmholtz adaptada a cada medio, incorporando rutinas con matrices de $4 \times 4$ para el acoplamiento y resolución del sistema en los módulos `solver_npmatrices_*.py`.
- **Casos implementados:**
  * **`Fluido_mixto01.py`:** 
    * Entrada izquierda en el Medio 1 con condición de **Dirichlet** ($p(0) = p_{in}$).
    * Interfaz interna con acoplamiento estricto de continuidad de presiones y flujos ($p_1 = p_2$, $v_1 = v_2$).
    * Extremo derecho en el Medio 2 evaluando condiciones de Sommerfeld y Neumann.
  * **`fluido_mixto02.py` (Modelo Principal):** 
    * Entrada izquierda en el Medio 1 con condición de **Neumann no homogéneo** ($v(0) = v_{in}$).
    * Interfaz interna con continuidad de fluido entre Aire y Agua.
    * Extremo derecho en el Medio 2 con condición de **Robin**. 
    * *Nota:* Este modelo en particular constituye la base a priori para la transición e implementación del **método de medio equivalente**.

---

## 🛠️ Requisitos e Instalación

Las simulaciones están optimizadas para ejecutarse en un entorno de Python con soporte para cálculo científico avanzado:

- **Python** $\ge$ 3.10
- **FEniCSx (DOLFINx)** == 0.11.0
- NumPy, Matplotlib, SciPy, PETSc4py

Para clonar y configurar el entorno localmente:
```bash
git clone [https://github.com/isamulosrios-bit/Tesis_Caracterizacion_ParametrosAcusticos.git](https://github.com/isamulosrios-bit/Tesis_Caracterizacion_ParametrosAcusticos.git)
cd Tesis_Caracterizacion_ParametrosAcusticos