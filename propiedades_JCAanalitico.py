import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# 1. MODELO JCA TEÓRICO (VALOR TEÓRICO EXACTO)
# =============================================================================
def calcular_jca_exacto(
    f,
    phi,
    sigma,
    alpha_inf,
    Lambda_v,
    Lambda_t,
    rho_0=1.21,
    c_0=343.0,
    eta=1.81e-5,
    P_0=101325.0,
    gamma=1.4,
    Pr=0.71,
):
    """Calcula Z_c y k_c teóricos exactos a partir de los parámetros microestructurales."""
    w = 2.0 * np.pi * f
    j = 1j

    # Densidad dinámica equivalente rho_eq(w)
    term_visc = 1.0 + j * (4.0 * (alpha_inf**2) * eta * rho_0 * w) / (
        (sigma**2) * (phi**2) * (Lambda_v**2)
    )
    rho_eq = (alpha_inf * rho_0 / phi) * (
        1.0 + (sigma * phi / (j * w * rho_0 * alpha_inf)) * np.sqrt(term_visc)
    )

    # Módulo de compresibilidad dinámico K_eq(w)
    term_term = 1.0 + j * (w * rho_0 * Pr * (Lambda_t**2)) / (16.0 * eta)
    H_w = 1.0 / (
        1.0 + (8.0 * eta / (j * w * rho_0 * Pr * (Lambda_t**2))) * np.sqrt(term_term)
    )
    K_eq = (gamma * P_0 / phi) / (gamma - (gamma - 1.0) * H_w)

    # Parámetros intrínsecos de propagación
    Z_c = np.sqrt(rho_eq * K_eq)
    k_c = w * np.sqrt(rho_eq / K_eq)

    # Condición física de pasividad: Re(Zc) > 0 y Im(kc) < 0
    if np.real(Z_c) < 0:
        Z_c = -Z_c
    if np.imag(k_c) > 0:
        k_c = -k_c

    return Z_c, k_c


# =============================================================================
# 2. ENSAYO TMM BICAPA (MUESTRA POROSA + CAVIDAD DE AIRE)
# =============================================================================
def matriz_T(k, Z, d):
    """Matriz de transferencia 2x2 para una capa acústica de espesor d."""
    return np.array(
        [
            [np.cos(k * d), 1j * Z * np.sin(k * d)],
            [1j * (1.0 / Z) * np.sin(k * d), np.cos(k * d)],
        ],
        dtype=complex,
    )


def ensayo_bicapa_tmm(f, Z_c, k_c, d_poroso, L_aire, rho_0=1.21, c_0=343.0):
    """Calcula la impedancia de superficie Zs y absorción alfa del sistema

    bicapa (Poroso d + Aire L) sobre respaldo rígido (v = 0).
    """
    Z_0 = rho_0 * c_0
    k_0 = (2.0 * np.pi * f) / c_0

    # Ensamble de la matriz total: T_tot = Tp(d) * Ta(L)
    Tp = matriz_T(k_c, Z_c, d_poroso)
    Ta = matriz_T(k_0, Z_0, L_aire)
    T_tot = np.matmul(Tp, Ta)

    # Respaldo rígido en el fondo v(d + L) = 0 -> Zs = T11 / T21
    t11 = T_tot.item(0, 0)
    t21 = T_tot.item(1, 0)
    Zs = t11 / t21

    # Coeficientes de reflexión y absorción
    R = (Zs - Z_0) / (Zs + Z_0)
    alpha = float(1.0 - np.abs(R) ** 2)

    return Zs, alpha


# =============================================================================
# 3. ALGORITMO INVERSO DE UTSUNO (MÉTODO DE LAS DOS CAVIDADES)
# =============================================================================
def reconstruir_utsuno_vectorial(
    Zs1_vec, Zs2_vec, L1, L2, d_poroso, f_vec, rho_0=1.21, c_0=343.0
):
    """Reconstruye Z_c(f) y k_c(f) a partir de las impedancias de superficie

    obtenidas con dos espesores de cavidad posterior (L1 y L2).
    """
    Z_0 = rho_0 * c_0
    w_vec = 2.0 * np.pi * f_vec
    k0_vec = w_vec / c_0

    # Impedancias en la cara posterior x = d debidas a cada cavidad de aire
    Z1_vec = -1j * Z_0 / np.tan(k0_vec * L1)
    Z2_vec = -1j * Z_0 / np.tan(k0_vec * L2)

    # 1. Reconstrucción de la impedancia característica Z_c
    num_Zc = Zs1_vec * Zs2_vec * (Z1_vec - Z2_vec) - Z1_vec * Z2_vec * (
        Zs1_vec - Zs2_vec
    )
    den_Zc = (Z1_vec - Z2_vec) - (Zs1_vec - Zs2_vec)
    Zc_rec = np.sqrt(num_Zc / den_Zc)
    Zc_rec = np.where(np.real(Zc_rec) < 0, -Zc_rec, Zc_rec)

    # 2. Reconstrucción del número de onda k_c (con desenvolvimiento de fase continuo)
    # La fase de kc se obtiene a partir de la relación de impedancias y se asegura que sea continua?
    termino = ((Zs1_vec + Zc_rec) / (Zs1_vec - Zc_rec)) * (
        (Z1_vec - Zc_rec) / (Z1_vec + Zc_rec)
    )
    fase_desenvuelta = np.unwrap(np.angle(termino))
    kc_rec = (1.0 / (2j * d_poroso)) * (np.log(np.abs(termino)) + 1j * fase_desenvuelta)

    return Zc_rec, kc_rec


# =============================================================================
# 4. PROGRAMA PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    # Constantes ambientales estándar (aire)
    rho_0 = 1.21  # Densidad del aire [kg/m^3]
    c_0 = 343.0  # Velocidad del sonido [m/s]

    # Parámetros del material poroso (Lana de roca típica)
    params_poroso = {
        "phi": 0.94,  # Porosidad
        "sigma": 40000.0,  # Resistividad al flujo [N s / m^4]
        "alpha_inf": 1.06,  # Tortuosidad
        "Lambda_v": 60e-6,  # Longitud característica viscosa [m]
        "Lambda_t": 120e-6,  # Longitud característica térmica [m]
    }
    d_poroso = 0.03  # Espesor de la muestra: 3 cm

    # Cavidades de aire ensayadas en el tubo de impedancia
    L1 = 0.02  # Cavidad de aire 1: 2 cm
    L2 = 0.04  # Cavidad de aire 2: 4 cm

    # Vector de frecuencias (200 puntos entre 200 Hz y 3500 Hz)
    f_vec = np.linspace(200, 3500, 200)
    N_freq = len(f_vec)

    # Arreglos de almacenamiento
    Zc_exacto = np.zeros(N_freq, dtype=complex)
    kc_exacto = np.zeros(N_freq, dtype=complex)
    Zs1_vec = np.zeros(N_freq, dtype=complex)
    alpha1_vec = np.zeros(N_freq)
    Zs2_vec = np.zeros(N_freq, dtype=complex)
    alpha2_vec = np.zeros(N_freq)

    # Bucle directo frecuencia por frecuencia
    for idx, f in enumerate(f_vec):
        # Valor teórico exacto (JCA)
        Zc_val, kc_val = calcular_jca_exacto(f, **params_poroso, rho_0=rho_0, c_0=c_0)
        Zc_exacto[idx] = Zc_val
        kc_exacto[idx] = kc_val

        # Ensayo con Cavidad L1
        Zs1, a1 = ensayo_bicapa_tmm(
            f, Zc_val, kc_val, d_poroso, L1, rho_0=rho_0, c_0=c_0
        )
        Zs1_vec[idx] = Zs1
        alpha1_vec[idx] = a1

        # Ensayo con Cavidad L2
        Zs2, a2 = ensayo_bicapa_tmm(
            f, Zc_val, kc_val, d_poroso, L2, rho_0=rho_0, c_0=c_0
        )
        Zs2_vec[idx] = Zs2
        alpha2_vec[idx] = a2

    # Reconstrucción de Utsuno a partir de las dos respuestas superficiales
    Zc_reconstruido, kc_reconstruido = reconstruir_utsuno_vectorial(
        Zs1_vec, Zs2_vec, L1, L2, d_poroso, f_vec, rho_0=rho_0, c_0=c_0
    )

    # --- MÉTRICA DE ERROR ESTÁNDAR: Error Relativo Porcentual Global ---
    err_Zc = (
        np.linalg.norm(np.abs(Zc_reconstruido - Zc_exacto))
        / np.linalg.norm(np.abs(Zc_exacto))
    ) * 100.0
    err_kc = (
        np.linalg.norm(np.abs(kc_reconstruido - kc_exacto))
        / np.linalg.norm(np.abs(kc_exacto))
    ) * 100.0

    print("\n" + "=" * 65)
    print("   VALIDACIÓN ANALÍTICA: MÉTODO DE LAS DOS CAVIDADES (ISO 10534-2)")
    print("=" * 65)
    print(f"Error Relativo Global en Impedancia Zc : {err_Zc:.4e} %")
    print(f"Error Relativo Global en Número de Onda kc: {err_kc:.4e} %")
    print("=" * 65 + "\n")

    # =========================================================================
    # 5. GRÁFICAS COMPARATIVAS OFICIALES (3 PANELES LIMPIOS)
    # =========================================================================
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8), dpi=120)
    fig.suptitle(
        "Validación Analítica: Método de las Dos Cavidades (Utsuno et al. / ISO"
        " 10534-2)\n"
        f"Muestra porosa: $d = {d_poroso * 100:.0f}$ cm | Cavidades de aire: $L_1 ="
        f" {L1 * 100:.0f}$ cm, $L_2 = {L2 * 100:.0f}$ cm",
        fontsize=12,
        fontweight="bold",
    )

    # Panel 1: Absorción en los dos ensayos
    ax1.plot(
        f_vec,
        alpha1_vec,
        "b-",
        linewidth=2.0,
        label=f"Cavidad $L_1 = {L1 * 100:.0f}$ cm",
    )
    ax1.plot(
        f_vec,
        alpha2_vec,
        "r--",
        linewidth=2.0,
        label=f"Cavidad $L_2 = {L2 * 100:.0f}$ cm",
    )
    ax1.set_xlabel("Frecuencia [Hz]")
    ax1.set_ylabel(r"Coeficiente de Absorción $\alpha$")
    ax1.set_title(r"Ensayos en Tubo: Absorción Sonora $\alpha(f)$")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    # Panel 2: Impedancia Característica Zc
    ax2.plot(f_vec, Zc_exacto.real, "k-", linewidth=2.2, label=r"Re($Z_c$) JCA Teórico")
    ax2.plot(
        f_vec,
        Zc_reconstruido.real,
        "b--",
        linewidth=1.8,
        label=r"Re($Z_c$) Dos Cavidades",
    )
    ax2.plot(
        f_vec, Zc_exacto.imag, "gray", linewidth=2.2, label=r"Im($Z_c$) JCA Teórico"
    )
    ax2.plot(
        f_vec,
        Zc_reconstruido.imag,
        "r--",
        linewidth=1.8,
        label=r"Im($Z_c$) Dos Cavidades",
    )
    ax2.set_xlabel("Frecuencia [Hz]")
    ax2.set_ylabel(r"Impedancia Característica $Z_c$ [Pa$\cdot$s/m]")
    ax2.set_title(f"Impedancia $Z_c(f)$\n(Error relativo: {err_Zc:.2e}%)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    # Panel 3: Número de Onda kc
    ax3.plot(f_vec, kc_exacto.real, "k-", linewidth=2.2, label=r"Re($k_c$) JCA Teórico")
    ax3.plot(
        f_vec,
        kc_reconstruido.real,
        "g--",
        linewidth=1.8,
        label=r"Re($k_c$) Dos Cavidades",
    )
    ax3.plot(
        f_vec, kc_exacto.imag, "gray", linewidth=2.2, label=r"Im($k_c$) JCA Teórico"
    )
    ax3.plot(
        f_vec,
        kc_reconstruido.imag,
        "m--",
        linewidth=1.8,
        label=r"Im($k_c$) Dos Cavidades",
    )
    ax3.set_xlabel("Frecuencia [Hz]")
    ax3.set_ylabel(r"Número de Onda $k_c$ [rad/m]")
    ax3.set_title(f"Número de Onda $k_c(f)$\n(Error relativo: {err_kc:.2e}%)")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()

    plt.tight_layout()
    nombre_archivo = "validacion_dos_cavidades_oficial.png"
    plt.savefig(nombre_archivo, dpi=300)
    print(f"Gráfico guardado exitosamente como '{nombre_archivo}'.")
    plt.show()
