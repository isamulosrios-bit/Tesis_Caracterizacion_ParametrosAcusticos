# 🔬 Caracterización de Parámetros Acústicos de Materiales Porosos mediante Elementos Finitos (FEniCSx) y Ensayos Virtuales de Dos Cavidades (ISO 10534-2)

* **👤 Autor:** Isaac Flores M.  
* **🎓 Tesis:** Pregrado en Ingeniería en Construcción / Ingeniería Civil  
* **🏢 Institución:** Universidad Austral de Chile (UACh), Valdivia, Chile  

---

## 1. 📝 Resumen del Proyecto

Este proyecto implementa un **gemelo digital computacional** para la caracterización acústica de materiales porosos absorbentes utilizados en construcción sostenible y control de ruido. 

El flujo de trabajo integra:
* **📐 Modelos analíticos directos (Ground Truth):** Formulación constitutiva JCA y propagación multicapa mediante el Método de Matrices de Transferencia (TMM).
* **💻 Simulaciones numéricas variacionales (FEM):** Resolución de la ecuación de Helmholtz para medios disipativos complejos en FEniCSx con soporte sesquilineal en PETSc (modo complejo).
* **🧪 Emulación virtual de tubo de impedancia (Kundt):** Medición de presiones mediante micrófonos virtuales en posiciones exactas de la malla, cálculo de la función de transferencia acústica \(H_{12}\), transferencia de fase al plano frontal de la muestra (\(x = 0\)) e identificación de la impedancia superficial (\(Z_s\)) y el coeficiente de reflexión (\(R\)).
* **🔄 Algoritmo inverso de caracterización (Two-Cavity Method):** Reconstrucción robusta de las propiedades intrínsecas del material (impedancia característica \(Z_c\) y número de onda complejo \(k_c\)) a partir de dos ensayos con cavidades de aire traseras (\(L_1\) y \(L_2\)) según la formulación de Utsuno et al. (1989), incorporando desenvolvimiento de fase continuo (*unwrap*) y validación cuantitativa mediante errores relativos globales frente a TMM y JCA exacto.

---

## 2. 🧮 Formulación Variacional del Problema

El problema acústico acoplado (tubo vacío + muestra porosa JCA + cavidad trasera de aire con pared rígida) se resuelve mediante la siguiente forma débil variacional derivada analíticamente:

\[\int_{\Omega} \frac{1}{\rho_{eq}} \nabla p \cdot \nabla q \, d\Omega - \int_{\Omega} \frac{\omega^2}{K_{eq}} p \cdot q \, d\Omega + \int_{\partial \Omega_{\text{izq}}} \frac{j k_0}{\rho_0} p \cdot q \, ds - \int_{\partial \Omega_{\text{der}}} \frac{\omega \tan(k_0 L)}{\rho_0 c_0} p \cdot q \, ds = \int_{\partial \Omega_{\text{izq}}} \frac{2 k_0 P_{in}}{\rho_0} q \, ds\]

Donde \(\rho_{eq}\) y \(K_{eq}\) varían espacialmente celda por celda mediante una máscara booleana (`np.where`), asignando las propiedades efectivas del modelo JCA en la zona porosa y las del aire puro en el resto del dominio.

---

## 📂 3. Estructura del Repositorio

```text
├── README.md                            # Documentación oficial del proyecto de tesis
├── JCA_TMMutsuno_FEMUtsuno.py           # Script unificado: JCA + TMM + Solver FEM TuboFEM + Utsuno + Gráficas
├── validacion_utsuno_tmm_fem_dolfinx.png # Gráfica comparativa de validación (Re/Im de Zc y kc)
└── Malla/                               # Archivos de discretización espacial y geometrías complementarias
```

---

## ⚙️ 4. Requisitos e Instalación

Para ejecutar con éxito las simulaciones numéricas de elementos finitos es estrictamente necesario contar con un entorno de Python configurado con soporte para números complejos en DOLFINx y PETSc (`fenics-complex`).

### 🔹 Paso 1: Configurar y activar el entorno FEniCSx (modo complejo)
Asegúrate de activar el entorno virtual o conda configurado para cálculos en complejos:

```bash
conda activate fenics-complex
```

### 🔹 Paso 2: Instalación de dependencias científicas
El script requiere librerías estándar orientadas al cálculo numérico, álgebra lineal distribuida y generación de gráficos:

```bash
pip install numpy matplotlib mpi4py petsc4py ufl dolfinx
```

---

## 🚀 5. Instrucciones de Ejecución

Una vez configurado el entorno e instaladas las dependencias, ejecuta el script principal de validación mediante el siguiente comando en tu terminal:

```bash
python JCA_TMMutsuno_FEMUtsuno.py
```

### ⚙️ ¿Qué realiza el script al ejecutarse?
* **🔍 Verificación de restricciones físicas:** Calcula la frecuencia de corte ($f_c$) del tubo para garantizar que no aparezcan modos transversales no planos.
* **📊 Barrido en frecuencias:** Resuelve iterativamente el sistema FEM y TMM en el rango especificado ($200\text{ Hz} - 3500\text{ Hz}$).
* **🎯 Validación de Utsuno:** Combina las dos cavidades de aire ($L_1$ y $L_2$) para extraer numéricamente $Z_c$ y $k_c$.
* **📈 Métricas de salida:** Imprime en la terminal los errores relativos porcentuales de convergencia entre FEM, TMM y el modelo teórico JCA exacto.
* **🖼️ Exportación gráfica:** Genera y guarda automáticamente la imagen de validación `validacion_utsuno_tmm_fem_dolfinx.png`.
