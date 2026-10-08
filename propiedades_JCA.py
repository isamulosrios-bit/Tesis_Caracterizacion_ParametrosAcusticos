import matplotlib.pyplot as plt
import numpy as np

# Librerías de FEniCSx / DOLFINx
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc


# =============================================================================
# 1. MODELO JCA Y PROPIEDADES DE FLUIDO EQUIVALENTE
# =============================================================================
def jca_properties(
    f,
    phi,
    sigma,
    alpha_inf,
    Lambda_v,
    Lambda_t,
    rho_0=1.21,
    eta=1.81e-5,
    P_0=101325.0,
    gamma=1.4,
    Pr=0.71,
):
    """Calcula rho_eq(w) y K_eq(w) según el modelo JCA."""
    w = 2.0 * np.pi * f
    j = 1j

    term_viscous = 1.0 + j * (4.0 * (alpha_inf**2) * eta * rho_0 * w) / (
        (sigma**2) * (phi**2) * (Lambda_v**2)
    )
    rho_eq = (alpha_inf * rho_0 / phi) * (
        1.0 + (sigma * phi / (j * w * rho_0 * alpha_inf)) * np.sqrt(term_viscous)
    )

    term_thermal = 1.0 + j * (w * rho_0 * Pr * (Lambda_t**2)) / (16.0 * eta)
    H_w = 1.0 / (
        1.0 + (8.0 * eta / (j * w * rho_0 * Pr * (Lambda_t**2))) * np.sqrt(term_thermal)
    )
    K_eq = (gamma * P_0 / phi) / (gamma - (gamma - 1.0) * H_w)

    return rho_eq, K_eq


def obtener_propiedades_capa(capa, f, rho_0=1.21, c_0=343.0):
    """Retorna (rho_eq, K_eq, k_c, Z_c) para poroso o cavidad de aire."""
    w = 2.0 * np.pi * f
    if capa["tipo"] == "poroso":
        rho_eq, K_eq = jca_properties(f, **capa["params"], rho_0=rho_0)
    elif capa["tipo"] == "aire":
        rho_eq = rho_0
        K_eq = rho_0 * (c_0**2)
    else:
        raise ValueError(f"Tipo no reconocido: {capa['tipo']}")

    k_c = w * np.sqrt(rho_eq / K_eq)
    k_c = np.where(np.imag(k_c) > 0, -k_c, k_c)
    Z_c = np.sqrt(rho_eq * K_eq)
    Z_c = np.where(np.real(Z_c) < 0, -Z_c, Z_c)
    return rho_eq, K_eq, k_c, Z_c


# =============================================================================
# 2. MÉTODO ANALÍTICO: MATRIZ DE TRANSFERENCIA (TMM)
# =============================================================================
def resolver_tmm(f_vec, lista_capas, espesores, rho_0=1.21, c_0=343.0):
    """Calcula analíticamente la impedancia de superficie Zs y absorción."""
    Z_0 = rho_0 * c_0
    N_freq = len(f_vec)
    Zs = np.zeros(N_freq, dtype=complex)
    alpha = np.zeros(N_freq)
    R_cx = np.zeros(N_freq, dtype=complex)

    idx_0 = 0
    idx_1 = 1

    for idx, f in enumerate(f_vec):
        T_total = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]])
        for capa, d in zip(lista_capas, espesores):
            _, _, k_c, Z_c = obtener_propiedades_capa(capa, f, rho_0=rho_0, c_0=c_0)
            T_i = np.array(
                [
                    [np.cos(k_c * d), 1j * Z_c * np.sin(k_c * d)],
                    [1j * (1.0 / Z_c) * np.sin(k_c * d), np.cos(k_c * d)],
                ]
            )
            T_total = np.matmul(T_total, T_i)

        t11 = T_total.item(idx_0, idx_0)
        t21 = T_total.item(idx_1, idx_0)
        Zw = t11 / t21
        R = (Zw - Z_0) / (Zw + Z_0)

        Zs[idx] = Zw
        alpha[idx] = 1.0 - np.abs(R) ** 2
        R_cx[idx] = R

    return Zs, np.real(alpha), R_cx


# =============================================================================
# 3. SOLVER FEniCSx CON MALLA ALINEADA Y DOS MICRÓFONOS (H12)
# =============================================================================
def resolver_fem_dos_microfonos_alineado(
    f_vec,
    lista_capas,
    espesores,
    mu1=-0.111,
    mu2=-0.080,
    L_tubo=0.30,
    h_target=0.0005,  # h = 0.5 mm: asegura coincidencia exacta de nodos
    rho_0=1.21,
    c_0=343.0,
    vin=1.0,
):
    """
    Simula el Tubo de Impedancia en FEniCSx garantizando que todos los bordes,
    interfaces físicas y micrófonos coincidan exactamente con nodos de la malla.
    """
    comm = MPI.COMM_WORLD
    Z_0 = rho_0 * c_0
    N_capas_muestra = len(espesores)
    D_muestra = sum(espesores)

    x1 = -mu1
    x2 = -mu2
    s = x1 - x2

    # Determinación del número de elementos con paso común h
    nx_total = int(round((L_tubo + D_muestra) / h_target))
    N_total_subdom = 1 + N_capas_muestra

    # 1. Malla 1D global [-L_tubo, D_muestra]
    domain = mesh.create_interval(comm, nx=nx_total, points=[-L_tubo, D_muestra])

    # 2. Etiquetado de subdominios
    col_x = 0
    x_edges = np.array([-L_tubo, 0.0] + list(np.cumsum(espesores)))

    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    cell_midpoints = mesh.compute_midpoints(
        domain, domain.topology.dim, np.arange(num_cells, dtype=np.int32)
    )

    cell_indices, cell_markers = [], []
    for c_idx in range(num_cells):
        x_c = cell_midpoints[c_idx, col_x]
        for tag_idx in range(N_total_subdom):
            if x_edges[tag_idx] <= x_c <= x_edges[tag_idx + 1]:
                cell_indices.append(c_idx)
                cell_markers.append(tag_idx + 1)
                break

    cell_tags = mesh.meshtags(
        domain,
        domain.topology.dim,
        np.array(cell_indices, dtype=np.int32),
        np.array(cell_markers, dtype=np.int32),
    )
    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)

    # 3. Frontera del Altavoz en x = -L_tubo
    facets_inlet = mesh.locate_entities_boundary(
        domain, domain.topology.dim - 1, lambda x: np.isclose(x[col_x], -L_tubo)
    )
    boundary_tags = mesh.meshtags(
        domain, domain.topology.dim - 1, facets_inlet, np.full_like(facets_inlet, 1)
    )
    ds = ufl.Measure("ds", domain=domain, subdomain_data=boundary_tags)

    # 4. Formulación Variacional Sesquilineal
    V = fem.functionspace(domain, ("Lagrange", 1))
    p = ufl.TrialFunction(V)
    q = ufl.TestFunction(V)

    inv_rho_consts = [
        fem.Constant(domain, PETSc.ScalarType(1.0 + 0j)) for _ in range(N_total_subdom)
    ]
    w2_inv_k_consts = [
        fem.Constant(domain, PETSc.ScalarType(1.0 + 0j)) for _ in range(N_total_subdom)
    ]
    neumann_const = fem.Constant(domain, PETSc.ScalarType(1.0 + 0j))

    a_vol = 0
    for tag_i in range(1, N_total_subdom + 1):
        a_vol += inv_rho_consts[tag_i - 1] * ufl.inner(ufl.grad(p), ufl.grad(q)) * dx(
            tag_i
        ) - w2_inv_k_consts[tag_i - 1] * ufl.inner(p, q) * dx(tag_i)

    L_form = neumann_const * ufl.conj(q) * ds(1)

    ph = fem.Function(V, name="presion")
    problem = LinearProblem(
        a_vol,
        L_form,
        bcs=[],
        u=ph,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="twomic_solver_",
    )

    # 5. Localización de los micrófonos virtuales
    x_mu1 = np.array([[mu1, 0.0, 0.0]])
    x_mu2 = np.array([[mu2, 0.0, 0.0]])
    idx_cell1 = int(np.argmin(np.abs(cell_midpoints[:, col_x] - mu1)))
    idx_cell2 = int(np.argmin(np.abs(cell_midpoints[:, col_x] - mu2)))
    cell_mu1 = np.array([idx_cell1], dtype=np.int32)
    cell_mu2 = np.array([idx_cell2], dtype=np.int32)

    idx_air = 0
    inv_rho_consts[idx_air].value = PETSc.ScalarType(1.0 / rho_0)

    # 6. Bucle de Frecuencias
    N_freq = len(f_vec)
    H12_vec = np.zeros(N_freq, dtype=complex)
    R_vec = np.zeros(N_freq, dtype=complex)
    Zs_vec = np.zeros(N_freq, dtype=complex)
    alpha_vec = np.zeros(N_freq)

    for idx, f_val in enumerate(f_vec):
        w_val = 2.0 * np.pi * f_val
        k0 = w_val / c_0

        w2_inv_k_consts[idx_air].value = PETSc.ScalarType((w_val**2) / (rho_0 * c_0**2))

        for i_capa, capa in enumerate(lista_capas):
            idx_capa = 1 + i_capa
            rho_eq_i, K_eq_i, _, _ = obtener_propiedades_capa(
                capa, f_val, rho_0=rho_0, c_0=c_0
            )
            inv_rho_consts[idx_capa].value = PETSc.ScalarType(1.0 / rho_eq_i)
            w2_inv_k_consts[idx_capa].value = PETSc.ScalarType((w_val**2) / K_eq_i)

        neumann_const.value = PETSc.ScalarType(1j * w_val * vin)
        problem.solve()

        p1 = ph.eval(x_mu1, cell_mu1).item()
        p2 = ph.eval(x_mu2, cell_mu2).item()

        H12 = p2 / p1
        H12_vec[idx] = H12

        # Despeje de R (Norma ISO 10534-2 / ASTM E1050)
        H_I = np.exp(-1j * k0 * s)
        H_R = np.exp(1j * k0 * s)
        R = ((H12 - H_I) / (H_R - H12)) * np.exp(2j * k0 * x1)
        R_vec[idx] = R

        Zw = Z_0 * (1.0 + R) / (1.0 - R)
        Zs_vec[idx] = Zw
        alpha_vec[idx] = 1.0 - np.abs(R) ** 2

    return H12_vec, R_vec, Zs_vec, alpha_vec


# =============================================================================
# 4. FUNCIÓN PARA CÁLCULO DE MÉTRICAS DE ERROR
# =============================================================================
def calcular_metricas_error(y_num, y_ref):
    """Calcula el Error L2 relativo [%], Error MAE y Error Máximo."""
    diff = y_num - y_ref
    abs_err = np.abs(diff)
    mae = np.mean(abs_err)
    max_err = np.max(abs_err)
    norm_ref = np.linalg.norm(np.abs(y_ref))
    l2_rel_pct = (np.linalg.norm(abs_err) / norm_ref) * 100.0 if norm_ref != 0 else 0.0
    return l2_rel_pct, mae, max_err


# =============================================================================
# 5. EJECUCIÓN, EVALUACIÓN DE ERRORES Y GRAFICACIÓN
# =============================================================================
if __name__ == "__main__":
    # Rango de frecuencia
    f_vec = np.linspace(200, 3500, 200)

    # Configuración de muestra: Lana de Roca (3 cm) + Cavidad de Aire (2 cm)
    lana_roca = {
        "tipo": "poroso",
        "params": {
            "phi": 0.94,
            "sigma": 40000.0,
            "alpha_inf": 1.06,
            "Lambda_v": 60e-6,
            "Lambda_t": 120e-6,
        },
    }
    cavidad_aire = {"tipo": "aire"}

    lista_capas = [lana_roca, cavidad_aire]
    espesores = [0.03, 0.02]

    # Micrófonos virtuales (AcoustiTube 40 mm, Tabla 1 del paper)
    mu1 = -0.111
    mu2 = -0.080

    print("1. Calculando solución analítica clásica (TMM)...")
    Zs_tmm, alpha_tmm, R_tmm = resolver_tmm(f_vec, lista_capas, espesores)

    print("2. Calculando FEniCSx con Malla Alineada y 2 Micrófonos (h = 0.5 mm)...")
    H12_fem, R_fem, Zs_fem, alpha_fem = resolver_fem_dos_microfonos_alineado(
        f_vec, lista_capas, espesores, mu1=mu1, mu2=mu2, L_tubo=0.30, h_target=0.0005
    )

    # --- CÁLCULO DE ERRORES CUANTITATIVOS ---
    l2_alpha, mae_alpha, max_alpha = calcular_metricas_error(alpha_fem, alpha_tmm)
    l2_R, mae_R, max_R = calcular_metricas_error(R_fem, R_tmm)
    l2_Zs, mae_Zs, max_Zs = calcular_metricas_error(Zs_fem, Zs_tmm)

    print("\n" + "=" * 65)
    print("      TABLA DE ERRORES NUMÉRICOS: FEniCSx vs ANALÍTICO TMM")
    print("=" * 65)
    print("Magnitud        | Error L2 Relativo [%] | Error MAE    | Error Máximo")
    print("-" * 65)
    print(
        f"Absorción (alpha) | {l2_alpha:19.4e} % | {mae_alpha:12.4e} | {max_alpha:12.4e}"
    )
    print(f"Reflexión (R)     | {l2_R:19.4e} % | {mae_R:12.4e} | {max_R:12.4e}")
    print(f"Impedancia (Zs)   | {l2_Zs:19.4e} % | {mae_Zs:12.4e} | {max_Zs:12.4e}")
    print("=" * 65 + "\n")

    # --- GRAFICACIÓN EN 4 PANELES CON ERRORES ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 9.5), dpi=110)
    fig.suptitle(
        f"Validación Tubo de Impedancia FEniCSx vs TMM (Malla Alineada $h=0.5$ mm)\n"
        f"Posición micrófonos: $\\mu_1 = {mu1 * 1000:.0f}$ mm, $\\mu_2 = {mu2 * 1000:.0f}$ mm",
        fontsize=13,
        fontweight="bold",
    )

    # Panel 1: Coeficiente de Absorción
    ax1.plot(f_vec, alpha_tmm, "b-", linewidth=2.0, label="Analítico (TMM)")
    ax1.plot(
        f_vec,
        alpha_fem,
        "r--",
        linewidth=2.0,
        label=f"FEniCSx ($e_{{L2}}={l2_alpha:.3f}\\%$)",
    )
    ax1.set_xlabel("Frecuencia [Hz]")
    ax1.set_ylabel(r"Coeficiente de Absorción $\alpha$")
    ax1.set_title(f"Absorción Sonora $\\alpha$ (Error Máx = {max_alpha:.2e})")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Panel 2: Impedancia de Superficie
    ax2.plot(f_vec, Zs_tmm.real, "b-", label="Re($Z_s$) TMM")
    ax2.plot(f_vec, Zs_fem.real, "r--", label="Re($Z_s$) FEM")
    ax2.plot(f_vec, Zs_tmm.imag, "c-", label="Im($Z_s$) TMM")
    ax2.plot(f_vec, Zs_fem.imag, "m--", label="Im($Z_s$) FEM")
    ax2.set_xlabel("Frecuencia [Hz]")
    ax2.set_ylabel(r"Impedancia $Z_s$ [Pa$\cdot$s/m]")
    ax2.set_title(f"Impedancia $Z_s(f)$ (Error $L_2 = {l2_Zs:.3f}\\%$)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    # Panel 3: Coeficiente de Reflexión
    ax3.plot(f_vec, np.abs(R_tmm), "k-", linewidth=2.0, label="|R| TMM")
    ax3.plot(
        f_vec,
        np.abs(R_fem),
        "k--",
        linewidth=2.0,
        label=f"|R| FEM ($e_{{L2}}={l2_R:.3f}\\%$)",
    )
    ax3.plot(f_vec, R_tmm.real, "b-", alpha=0.5, label="Re(R) TMM")
    ax3.plot(f_vec, R_fem.real, "r--", alpha=0.5, label="Re(R) FEM")
    ax3.plot(f_vec, R_tmm.imag, "c-", alpha=0.5, label="Im(R) TMM")
    ax3.plot(f_vec, R_fem.imag, "m--", alpha=0.5, label="Im(R) FEM")
    ax3.set_xlabel("Frecuencia [Hz]")
    ax3.set_ylabel("Amplitud Fasorial")
    ax3.set_title(f"Coeficiente de Reflexión $R$ (Error $L_2 = {l2_R:.3f}\\%$)")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()

    # Panel 4: Curvas de Error Residual Directo |FEM - TMM|
    err_alpha_vec = np.abs(alpha_fem - alpha_tmm)
    err_R_vec = np.abs(R_fem - R_tmm)
    ax4.semilogy(
        f_vec,
        err_alpha_vec,
        "r-",
        linewidth=2.0,
        label=r"Error Absoluto $|\alpha_{FEM} - \alpha_{TMM}|$",
    )
    ax4.semilogy(
        f_vec,
        err_R_vec,
        "b--",
        linewidth=1.8,
        label=r"Error Absoluto $|R_{FEM} - R_{TMM}|$",
    )
    ax4.axvline(
        x=556.25,
        color="gray",
        linestyle=":",
        label=r"$f_{min}$ tubo 40 mm (ISO 10534-2)",
    )
    ax4.set_xlabel("Frecuencia [Hz]")
    ax4.set_ylabel("Error Residual Absoluto (Escala Log)")
    ax4.set_title("Espectro de Error Residual FEM vs TMM")
    ax4.grid(True, which="both", linestyle="--", alpha=0.6)
    ax4.legend()

    plt.tight_layout()
    nombre_archivo = "validacion_con_metricas_error.png"
    plt.savefig(nombre_archivo, dpi=300)
    print(f"Gráfico con errores guardado como '{nombre_archivo}'.")
    plt.show()
