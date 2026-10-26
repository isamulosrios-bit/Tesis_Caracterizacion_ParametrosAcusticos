# Caracterización de Parámetros Acústicos de Materiales Porosos mediante Elementos Finitos (FEniCSx) y Ensayos Virtuales de Dos Cavidades (ISO 10534-2)

Repositorio oficial para el desarrollo de la tesis de pregrado en Ingeniería, centrado en el modelado numérico, la simulación variacional por elementos finitos con **FEniCSx (DOLFINx)** y la validación analítica de la propagación de ondas acústicas en medios porosos mediante el modelo de fluido equivalente de **Johnson-Champoux-Allard (JCA)** y la técnica estandarizada de **Dos Cavidades (ISO 10534-2 / ASTM E1050)**.

---

## 📌 Resumen del Proyecto

Este proyecto implementa un gemelo digital computacional para la caracterización acústica de materiales porosos absorbentes utilizados en construcción sostenible y control de ruido. El flujo de trabajo integra:

1. **Modelos analíticos directos (Ground Truth):** Formulación constitutiva JCA y propagación multicapa mediante el Método de Matrices de Transferencia (**TMM**).
2. **Simulaciones numéricas variacionales (FEM):** Resolución de la ecuación de Helmholtz para medios disipativos complejos en **FEniCSx** con soporte sesquilineal en **PETSc** (modo complejo).
3. **Emulación virtual de tubo de impedancia (Kundt):** Medición de presiones mediante micrófonos virtuales en posiciones exactas de la malla, cálculo de la función de transferencia acústica $H_{12}$, transferencia de fase al plano frontal de la muestra ($x = 0$) e identificación de la impedancia superficial ($Z_s$) y el coeficiente de reflexión ($R$).
4. **Algoritmo inverso de caracterización (Two-Cavity Method):** Reconstrucción robusta de las propiedades intrínsecas del material (impedancia característica $Z_c$ y número de onda complejo $k_c$) a partir de dos ensayos con cavidades de aire traseras ($L_1$ y $L_2$) según la formulación de **Utsuno et al. (1989)**, incorporando desenvolvimiento de fase continuo (`unwrap`) y validación cuantitativa mediante errores relativos globales frente a TMM y JCA exacto.

---

## 🧮 Formulación Variacional del Problema

El problema acústico acoplado (tubo vacío + muestra porosa JCA + cavidad trasera de aire con pared rígida) se resuelve mediante la siguiente forma débil variacional derivada analíticamente:

$$\int_{\Omega} \frac{1}{\rho_{eq}} \nabla p \cdot \nabla q \, d\Omega - \int_{\Omega} \frac{\omega^2}{K_{eq}} p \cdot q \, d\Omega + \int_{\partial \Omega_{izq}} \frac{j k_0}{\rho_0} p \cdot q \, ds - \int_{\partial \Omega_{der}} \frac{\omega \tan(k_0 L)}{\rho_0 c_0} p \cdot q \, ds = \int_{\partial \Omega_{izq}} \frac{2 k_0 P_{in}}{\rho_0} q \, ds$$

Donde $\rho_{eq}$ y $K_{eq}$ varían espacialmente celda por celda mediante una máscara booleana (`np.where`), asignando las propiedades efectivas del modelo JCA en la zona porosa y las del aire puro en el resto del dominio.

---

## 📂 Estructura del Repositorio

```text
├── README.md                               # Documentación oficial del proyecto de tesis
├── tubo_utsuno_fem_dolfinx.py              # [OFICIAL] Script unificado: JCA + TMM + Solver FEM TuboFEM + Utsuno + Gráficas
├── validacion_utsuno_tmm_fem_dolfinx.png   # Gráfica comparativa de validación (Re/Im de Zc y kc)
└── Malla/                                  # Archivos de discretización espacial y geometrías complementarias