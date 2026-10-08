# Caracterización de Parámetros Acústicos de Materiales Porosos mediante Elementos Finitos (FEniCSx) y Ensayos Virtuales de Dos Cavidades (ISO 10534-2)

Repositorio oficial para el desarrollo de la tesis de pregrado en Ingeniería, centrado en el modelado numérico, la simulación variacional por elementos finitos con **FEniCSx (DOLFINx)** y la validación analítica de la propagación de ondas acústicas en medios porosos mediante el modelo de fluido equivalente **Johnson-Champoux-Allard (JCA)** y la técnica estandarizada de **Dos Cavidades (ISO 10534-2 / ASTM E1050)**.

---

## 📌 Resumen del Proyecto

Este proyecto implementa un gemelo digital computacional para la caracterización acústica de materiales porosos utilizados en construcción y control de ruido. Se integra:

1. **Modelos analíticos directos (Ground Truth):** Formulación constitutiva JCA y propagación multicapa mediante el Método de Matrices de Transferencia (**TMM**).
2. **Simulaciones numéricas variacionales (FEM):** Resolución de la ecuación de Helmholtz para medios disipativos complejos en **FEniCSx** con soporte sesquilineal en **PETSc**.
3. **Emulación virtual de tubo de impedancia (Kundt):** Medición de presiones mediante dos micrófonos virtuales (\\(x = \mu_1\\) y \\(x = \mu_2\\)), cálculo de la función de transferencia acústica \\(H_{12}\\), desacoplamiento de ondas planas e identificación de la impedancia de superficie (\\(Z_s\\)) y el coeficiente de absorción (\\(\alpha\\)).
4. **Algoritmo inverso de caracterización (Two-Cavity Method):** Reconstrucción de las propiedades intrínsecas del material (impedancia característica \\(Z_c\\) y número de onda complejo \\(k_c\\)) a partir de dos ensayos con cavidades de aire (\\(L_1\\) y \\(L_2\\)) según la formulación de **Utsuno et al. (1989)** y **Eser et al. (2025)**, incorporando desenvolvimiento de fase continuo (`unwrap`) y métricas cuantitativas de error relativo global.

---

## 📂 Estructura del Repositorio

```text
├── README.md                                 # Documentación general del repositorio
├── propiedades_JCAanalitico.py               # [OFICIAL] Validación analítica: JCA + TMM Bicapa + Dos Cavidades (Utsuno)
├── propiedades_JCA.py                        # [OFICIAL] Solver FEM FEniCSx: Tubo virtual con 2 micrófonos (H12) vs TMM
├── fluido_homogeneo01.py                     # Modelo base 1D: Aire con excitación de presión Dirichlet
├── fluido_homogeneo02.py                     # Modelo base 1D: Aire con excitación de velocidad Neumann
├── Fluido_mixto01.py                         # Modelo bicapa Aire-Agua con excitación Dirichlet y continuidad
├── fluido_mixto02.py                         # Modelo bicapa Aire-Agua con excitación Neumann y Robin
├── solver_npmatrices_Robin.py                # Ensamblador matricial complementario (Robin)
├── solver_npmatrices_sommerfeld.py           # Ensamblador matricial complementario (Sommerfeld)
└── Malla/                                    # Archivos de discretización espacial y geometrías