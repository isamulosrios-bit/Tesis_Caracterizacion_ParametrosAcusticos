import matplotlib.pyplot as plt
import numpy as np

# --- IMPORTACIONES DE DOLFINx Y UFL ---
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc


def simular_sistema_mixto_completo():
    print("=" * 65)
    print(" SIMULACIÓN ACÚSTICA BICAPA 1D: FEniCSx vs. SOLUCIÓN ANALÍTICA")
    print("=" * 65)

    # =========================================================================
    # 1. PARÁMETROS FÍSICOS Y GEOMÉTRICOS
    # =========================================================================
    freq = 500.0  # Frecuencia de trabajo [Hz]
    omega = 2.0 * np.pi * freq  # Frecuencia angular [rad/s]

    # Medio 1 (Aire: 0 <= x <= x_interface)
    rho1 = 1.21  # Densidad [kg/m^3]
    c1 = 343.0  # Velocidad del sonido [m/s]
    k1 = omega / c1  # Número de onda [1/m]

    # Medio 2 (Agua / Gas pesado: x_interface < x <= L_val)
    rho2 = 998.0  # Densidad [kg/m^3]
    c2 = 1480.0  # Velocidad del sonido [m/s]
    k2 = omega / c2  # Número de onda [1/m]

    # Geometría y Condiciones de Frontera
    L_val = 1.0  # Longitud total del tubo [m]
    x_interface = 0.5  # Posición de la interfaz [m]
    vin = 1.0  # Velocidad de entrada impuesta [m/s]

    # Impedancia de Robin en x = L_val (Impedancia compleja de carga superficial)
    zs = (2.0 + 0.5j) * (rho2 * c2)

    # Discretización espacial
    nx_mesh = 400  # Número de elementos finitos en la malla

    # =========================================================================
    # 2. SOLUCIÓN ANALÍTICA EXACTA (SISTEMA MATRICIAL 4x4 EN NUMPY)
    # =========================================================================
    M = np.zeros((4, 4), dtype=complex)
    b_vec = np.zeros(4, dtype=complex)

    # Fila 0: Entrada Neumann (x = 0) -> A1 - B1 = (w * rho1 * vin) / k1
    M[0, 0] = 1.0
    M[0, 1] = -1.0
    b_vec[0] = (omega * rho1 * vin) / k1

    # Fila 1: Continuidad de presión en la interfaz (x = x_interface)
    M[1, 0] = 0.0
    M[1, 1] = np.exp(1j * k1 * x_interface) + np.exp(-1j * k1 * x_interface)
    M[1, 2] = -np.exp(-1j * k2 * x_interface)
    M[1, 3] = -np.exp(1j * k2 * x_interface)
    b_vec[1] = -(omega * rho1 * vin * np.exp(-1j * k1 * x_interface)) / k1

    # Fila 2: Continuidad de flujo / velocidad normal en la interfaz
    M[2, 0] = 0.0
    M[2, 1] = (k1 / rho1) * (
        np.exp(1j * k1 * x_interface) - np.exp(-1j * k1 * x_interface)
    )
    M[2, 2] = (k2 / rho2) * np.exp(-1j * k2 * x_interface)
    M[2, 3] = (-k2 / rho2) * np.exp(1j * k2 * x_interface)
    b_vec[2] = omega * vin * np.exp(-1j * k1 * x_interface)

    # Fila 3: Condición de Robin en la salida (x = L_val)
    M[3, 0] = 0.0
    M[3, 1] = 0.0
    M[3, 2] = np.exp(-1j * k2 * L_val) * ((omega * rho2 / zs) - k2)
    M[3, 3] = np.exp(1j * k2 * L_val) * ((omega * rho2 / zs) + k2)
    b_vec[3] = 0.0

    # Resolver el sistema lineal de amplitudes A1, B1, A2, B2
    A1, B1, A2, B2 = np.linalg.solve(M, b_vec)

    # =========================================================================
    # 3. SOLUCIÓN NUMÉRICA MEDIANTE ELEMENTOS FINITOS (DOLFINx)
    # =========================================================================
    domain = mesh.create_interval(MPI.COMM_WORLD, nx=nx_mesh, points=[0.0, L_val])

    # Identificación de Subdominios por celdas
    cells_air = mesh.locate_entities(
        domain, domain.topology.dim, lambda x: x[0] <= x_interface + 1e-8
    )
    cells_water = mesh.locate_entities(
        domain, domain.topology.dim, lambda x: x[0] >= x_interface - 1e-8
    )

    subdomain_tags = mesh.meshtags(
        domain,
        domain.topology.dim,
        np.hstack([cells_air, cells_water]),
        np.hstack([np.full_like(cells_air, 1), np.full_like(cells_water, 2)]),
    )
    dx = ufl.Measure("dx", domain=domain, subdomain_data=subdomain_tags)

    # Identificación de Fronteras (x = 0 y x = L_val)
    facets_x0 = mesh.locate_entities_boundary(
        domain, domain.topology.dim - 1, lambda x: np.isclose(x[0], 0.0)
    )
    facets_x1 = mesh.locate_entities_boundary(
        domain, domain.topology.dim - 1, lambda x: np.isclose(x[0], L_val)
    )

    boundary_tags = mesh.meshtags(
        domain,
        domain.topology.dim - 1,
        np.hstack([facets_x0, facets_x1]),
        np.hstack([np.full_like(facets_x0, 1), np.full_like(facets_x1, 2)]),
    )
    ds = ufl.Measure("ds", domain=domain, subdomain_data=boundary_tags)

    # Espacio de Funciones Lagrange P1
    V = fem.functionspace(domain, ("Lagrange", 1))
    p = ufl.TrialFunction(V)
    q = ufl.TestFunction(V)

    # Coeficientes del dominio
    inv_rho1 = fem.Constant(domain, PETSc.ScalarType(1.0 / rho1))
    inv_rho2 = fem.Constant(domain, PETSc.ScalarType(1.0 / rho2))
    k1_sq_rho1 = fem.Constant(domain, PETSc.ScalarType(k1**2 / rho1))
    k2_sq_rho2 = fem.Constant(domain, PETSc.ScalarType(k2**2 / rho2))

    # Formulación Débil: Lado Izquierdo a(p, q)
    a_vol = (
        inv_rho1 * ufl.inner(ufl.grad(p), ufl.grad(q)) * dx(1)
        + inv_rho2 * ufl.inner(ufl.grad(p), ufl.grad(q)) * dx(2)
        - k1_sq_rho1 * p * ufl.conj(q) * dx(1)
        - k2_sq_rho2 * p * ufl.conj(q) * dx(2)
    )

    coef_robin = fem.Constant(domain, PETSc.ScalarType(1j * omega / zs))
    a_boundary = coef_robin * p * ufl.conj(q) * ds(2)
    a_form = a_vol + a_boundary

    # Formulación Débil: Lado Derecho L(q) (Signo positivo tras despejar)
    source_val = fem.Constant(domain, PETSc.ScalarType(1j * omega * vin))
    L_form = source_val * ufl.conj(q) * ds(1)

    # Resolver Problema Variacional Lineal
    ph = fem.Function(V, name="presion_1d")
    prob = LinearProblem(
        a_form,
        L_form,
        bcs=[],
        u=ph,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="sol_sistema_mixto",
    )
    prob.solve()

    # =========================================================================
    # 4. COMPARACIÓN Y EVALUACIÓN DEL ERROR L2
    # =========================================================================
    x_nodes = domain.geometry.x[:, 0]
    sort_idx = np.argsort(x_nodes)
    x_coords = x_nodes[sort_idx]
    p_fem_vals = ph.x.array[sort_idx]

    # Evaluación exacta de la solución analítica en los mismos nodos
    p_analitica_vals = np.zeros_like(x_coords, dtype=complex)
    for i, x in enumerate(x_coords):
        if x <= x_interface:
            p_analitica_vals[i] = A1 * np.exp(-1j * k1 * x) + B1 * np.exp(1j * k1 * x)
        else:
            p_analitica_vals[i] = A2 * np.exp(-1j * k2 * x) + B2 * np.exp(1j * k2 * x)

    # Cálculo del Error Relativo L2 Discreto
    error_l2_rel = np.linalg.norm(p_fem_vals - p_analitica_vals) / np.linalg.norm(
        p_analitica_vals
    )

    print(f"\n -> Error L2 Relativo entre FEM y Analítico: {error_l2_rel:.4e}\n")

    # =========================================================================
    # 5. VISUALIZACIÓN DETALLADA PARA INFORME Y TESIS
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 5), dpi=100)

    # Sombreado de los subdominios de fluidos
    ax.axvspan(
        0.0,
        x_interface,
        color="#E0F2FE",
        alpha=0.6,
        label="Medio 1 (Aire)",
    )
    ax.axvspan(
        x_interface,
        L_val,
        color="#DCFCE7",
        alpha=0.6,
        label="Medio 2 (Agua / Gas)",
    )

    # Graficar Amplitudes de Presión
    ax.plot(
        x_coords,
        np.abs(p_analitica_vals),
        "k-",
        linewidth=2.5,
        label="|P| Analítica Exacta",
    )
    ax.plot(
        x_coords,
        np.abs(p_fem_vals),
        "r--",
        linewidth=1.8,
        label="|P| FEM (DOLFINx)",
    )

    # Marca vertical de la Interfaz
    ax.axvline(
        x=x_interface,
        color="#0284C7",
        linestyle="--",
        linewidth=2,
        label=f"Interfaz (x = {x_interface} m)",
    )

    # Cajas de información contextual dentro del gráfico
    max_p = np.max(np.abs(p_analitica_vals))
    ax.text(
        x_interface * 0.4,
        max_p * 0.85,
        f"MEDIO 1 (Aire)\nρ1 = {rho1} kg/m³\nc1 = {c1} m/s",
        fontsize=9.5,
        fontweight="bold",
        color="#0369A1",
        ha="center",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="#0284C7",
            alpha=0.85,
        ),
    )

    ax.text(
        x_interface + (L_val - x_interface) * 0.5,
        max_p * 0.85,
        f"MEDIO 2 (Agua/Gas)\nρ2 = {rho2} kg/m³\nc2 = {c2} m/s",
        fontsize=9.5,
        fontweight="bold",
        color="#15803D",
        ha="center",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="#16A34A",
            alpha=0.85,
        ),
    )

    # Configuración de Ejes y Títulos
    ax.set_title(
        f"Distribución de Presión Acústica en Medio Mixto 1D (Neumann - Robin)\n"
        f"Convergencia Numérica FEM vs. Analítica | Error L2 = {error_l2_rel:.3e}",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Posición x [m]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Amplitud de Presión |P(x)| [Pa]", fontsize=10, fontweight="bold")
    ax.set_xlim(0.0, L_val)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)

    plt.tight_layout()
    out_filename = "resultado_fluido_mixto_oficial.png"
    plt.savefig(out_filename, dpi=100, bbox_inches="tight")
    print(f"¡Gráfica exportada exitosamente como '{out_filename}'!")
    plt.show()


if __name__ == "__main__":
    simular_sistema_mixto_completo()
