"""
Método de dos cavidades (Utsuno): TMM analítico vs FEM (DOLFINx)
================================================================

Convención temporal: e^{+jωt}  ->  p(x) = A e^{-jk x} + B e^{+jk x}

Geometría (x = 0 en la CARA FRONTAL de la muestra porosa):

    x=-L_tubo       -x1       -x2     0         d        d+L
       |  aire     m1        m2     | POROSO  | CAVIDAD |/// pared rígida
       Pin ->                      |  (JCA)  |  (aire) |

 - TMM : solo muestra + cavidad (sin aire de tubo).
 - FEM : tubo + muestra + cavidad; R se traslada a x=0 con e^{2jk0 x1}.

Forma débil (con q = función de test):
  ∫ 1/ρ_eq ∇p·∇q dΩ − ∫ ω²/K_eq p q dΩ + ∫_izq jk0/ρ0 p q ds
        [ + ∫_der jω/Zb p q ds  (solo si Zb finito; pared rígida => 0) ]
        = ∫_izq 2jk0 Pin/ρ0 q ds
"""

import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem, geometry, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

assert np.issubdtype(PETSc.ScalarType, np.complexfloating), (
    "Este script requiere DOLFINx/PETSc en modo COMPLEJO (entorno fenics-complex)."
)  # comprueba si PETSc está configurado para trabajar con números complejos, lo cual es necesario para este script. Si no es así, el assert hace una parada de emergencia y muestra un mensaje de error indicando que se necesita el entorno fenics-complex.

RHO0 = 1.21
C0 = 343.0
ETA = 1.81e-5
P0 = 101325.0
GAMMA = 1.4
PR = 0.71


# =============================================================================
# 1. MODELO JCA  (rho_eq, K_eq, Zc, kc)
# =============================================================================
def jca(f, phi, sigma, alpha_inf, Lambda_v, Lambda_t):
    w = 2.0 * np.pi * f
    j = 1j

    rho_eq = (RHO0 * alpha_inf / phi) * (
        1.0
        - j
        * (sigma * phi / (w * RHO0 * alpha_inf))
        * np.sqrt(
            1.0
            + j
            * (4.0 * alpha_inf**2 * ETA * RHO0 * w)
            / (sigma**2 * phi**2 * Lambda_v**2)
        )
    )

    a = 8.0 * ETA / (RHO0 * PR * Lambda_t**2 * w)
    b = RHO0 * PR * Lambda_t**2 * w / (16.0 * ETA)
    bracket = 1.0 - j * a * np.sqrt(1.0 + j * b)
    K_eq = (GAMMA * P0 / phi) / (GAMMA - (GAMMA - 1.0) / bracket)

    Zc = np.sqrt(rho_eq * K_eq)
    kc = w * np.sqrt(rho_eq / K_eq)
    if Zc.real < 0:
        Zc = -Zc
    if kc.imag > 0:  # e^{+jωt}: onda atenuada => Im(kc) < 0
        kc = -kc
    return rho_eq, K_eq, Zc, kc


# =============================================================================
# 2. TMM ANALÍTICO:  Ttot = T_poroso · T_aire ,  Zs = T11/T21
# =============================================================================
def matriz_T(k, Z, d):
    c, s = np.cos(k * d), np.sin(k * d)
    return np.array([[c, 1j * Z * s], [1j * s / Z, c]], dtype=complex)


def zs_tmm(f, Zc, kc, d, L):
    Z0 = RHO0 * C0
    k0 = 2.0 * np.pi * f / C0
    T = matriz_T(kc, Zc, d) @ matriz_T(k0, Z0, L)
    Zs = T[0, 0] / T[1, 0]
    R = (Zs - Z0) / (Zs + Z0)
    return Zs, R


# =============================================================================
# 3. FUNCIONES AUXILIARES MODULARES (Estilo Docente)
# =============================================================================
def tubo_variational_form(P, q, rho_f, K_f, c_w2, c_rad, c_Zb, c_src, dx, ds, Zb):
    """
    Construye y retorna la forma bilineal (a) y lineal (Lf) para el problema
    acústico del tubo con el modelo JCA y condiciones de contorno.
    """
    a = (
        (1.0 / rho_f) * ufl.inner(ufl.grad(P), ufl.grad(q)) * dx
        - (c_w2 / K_f) * ufl.inner(P, q) * dx
        + c_rad * ufl.inner(P, q) * ds(1)
    )
    if np.isfinite(Zb):
        a += c_Zb * ufl.inner(P, q) * ds(2)
    Lf = ufl.inner(c_src, q) * ds(1)
    return a, Lf


# =============================================================================
# 4. FEM EN DOLFINx  (LinearProblem)
# =============================================================================
class TuboFEM:
    """Funcion clase hace: Tubo + muestra + cavidad. La malla y las formas se crean UNA vez;
    en cada frecuencia solo se actualizan constantes/propiedades y se resuelve."""

    def __init__(self, params, d, L, x1, x2, L_tubo=0.12, h=2.5e-4, Zb=np.inf, Pin=1.0):
        self.params, self.d, self.L = params, d, L
        self.x1, self.x2, self.Pin, self.Zb = x1, x2, Pin, Zb  # Función de inicio

        x_min, x_max = -L_tubo, d + L
        N = int(round((x_max - x_min) / h))  # número de elementos (celdas) en la malla
        self.msh = msh = mesh.create_interval(
            MPI.COMM_SELF, N, [x_min, x_max]
        )  # Se crea la malla

        # Los micrófonos y la interfaz deben caer en NODOS exactos (sin interpolar)
        xs = msh.geometry.x[
            :, 0
        ]  # Extrae las coordenadas de los nodos de la malla en toda la dirección x
        for xc in (-x1, -x2, 0.0, d):
            assert np.min(np.abs(xs - xc)) < 1e-9, (
                f"x={xc} si no cae en un nodo: se ajusta h para que sea divisor de las distancias."
            )
        # El assert evalua: ¿Hay un nodo de la malla que caiga exactamente sobre la posición del micrófono o de la interfaz?
        # el 1e-9 dice: Si la distancia entre el nodo y el punto es menor a un nanómetro, se acépta como exacto.
        # np.min(np.abs(xs - xc)) calcula la distancia mínima entre el nodo y el punto xc. El cual es el punto que se busca para los micrófonos y la interfaz. Si no hay ningún nodo que caiga exactamente sobre el punto, se lanza un error con un mensaje indicando que se debe ajustar el tamaño del elemento h para que sea divisor de las distancias.

        # Propiedades por celda (DG0): aire o poroso
        make_space = (
            getattr(fem, "functionspace", None) or fem.FunctionSpace
        )  # Asegura que el codigo funcione sin importar la version de dolfinx.
        V0 = make_space(
            msh, ("DG", 0)
        )  # funciones de prueba discontinuas de orden 0 (constantes por celda) en la malla msh.
        xc_cells = V0.tabulate_dof_coordinates()[
            :, 0
        ]  # Obtiene las coordenadas de los grados de libertad (dofs) de la función de prueba V0. En este caso, como V0 es DG0, cada celda tiene un dof asociado a su centroide. La coordenada x de cada celda se extrae con [:, 0].
        self.mask_poroso = (xc_cells > 0.0) & (xc_cells < d)
        self.rho_f = fem.Function(V0)
        self.K_f = fem.Function(V0)

        V = make_space(msh, ("Lagrange", 1))
        P, q = ufl.TrialFunction(V), ufl.TestFunction(V)

        # Fronteras: 1 = izquierda (radiación + fuente), 2 = derecha (pared)
        fdim = msh.topology.dim - 1
        izq = mesh.locate_entities(msh, fdim, lambda x: np.isclose(x[0], x_min))
        der = mesh.locate_entities(msh, fdim, lambda x: np.isclose(x[0], x_max))
        idx = np.hstack(
            [izq, der]
        ).astype(
            np.int32
        )  # concatena los índices de las entidades de la frontera izquierda y derecha en un solo array. Esto se hace para poder asignar valores diferentes a cada frontera en el siguiente paso.
        val = np.hstack([np.full_like(izq, 1), np.full_like(der, 2)]).astype(np.int32)
        order = np.argsort(
            idx
        )  # ordena los índices de las entidades de la frontera para que estén en orden ascendente. Esto es necesario porque la función mesh.meshtags espera que los índices estén en orden ascendente.
        ft = mesh.meshtags(msh, fdim, idx[order], val[order])
        ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
        dx = ufl.Measure("dx", domain=msh)

        S = PETSc.ScalarType  # Asegura que las constantes sean del mismo tipo que el tipo de datos de PETSc (complejo en este caso). Esto es importante para evitar errores de tipo al realizar operaciones con estas constantes en el problema lineal.
        # S(0) es un cero de tipo complejo.
        self.c_w2 = fem.Constant(msh, S(0))  # ω²
        self.c_rad = fem.Constant(msh, S(0))  # j k0 / ρ0
        self.c_src = fem.Constant(msh, S(0))  # 2 j k0 Pin / ρ0
        self.c_Zb = fem.Constant(msh, S(0))  # jω / Zb

        # Construcción modular de la forma variacional
        a, Lf = tubo_variational_form(
            P,
            q,
            self.rho_f,
            self.K_f,
            self.c_w2,
            self.c_rad,
            self.c_Zb,
            self.c_src,
            dx,
            ds,
            Zb,
        )
        # El isfinitive Zb significa que si Zb es finito, se añade el término de la frontera derecha (pared) al sistema lineal. Si Zb es infinito (pared rígida), este término no se añade.

        opts = {
            "ksp_type": "preonly",
            "pc_type": "lu",
        }  # Biblioteca de resolución de sistemas lineales: se utiliza un solucionador directo (LU) para resolver el sistema lineal resultante del problema de FEM. Esto es adecuado para problemas pequeños o medianos, donde la factorización LU es eficiente y estable.
        try:  # DOLFINx >= 0.10 exige petsc_options_prefix versiones reciente.
            self.problem = LinearProblem(
                a, Lf, bcs=[], petsc_options=opts, petsc_options_prefix="tubo_"
            )
        except (
            TypeError
        ):  # DOLFINx < 0.10 no tiene petsc_options_prefix. Versiones clasicas.
            self.problem = LinearProblem(a, Lf, bcs=[], petsc_options=opts)

        # Celdas donde evaluar los micrófonos
        self.pts = np.array(
            [[-x1, 0.0, 0.0], [-x2, 0.0, 0.0]], dtype=np.float64
        )  # Puntos de los micrófonos en coordenadas 3D (x, y, z). Se colocan en la línea central del tubo (y=0, z=0) a las posiciones -x1 y -x2. La forma de la matriz es (2, 3), donde 2 es el número de micrófonos y 3 es el número de coordenadas espaciales.
        tree = geometry.bb_tree(
            msh, msh.topology.dim
        )  # Se construye un árbol de búsqueda de cajas delimitadoras (bounding box tree) para la malla msh. Este árbol permite realizar consultas rápidas sobre la geometría de la malla, como encontrar qué celdas contienen ciertos puntos.
        cand = geometry.compute_collisions_points(
            tree, self.pts
        )  # Se realiza una consulta de colisión entre los puntos de los micrófonos (self.pts) y el árbol de búsqueda de cajas delimitadoras (tree).
        coll = geometry.compute_colliding_cells(
            msh, cand, self.pts
        )  # Se identifican las celdas de la malla que colisionan con los puntos de los micrófonos.
        self.cells = np.array(
            [coll.links(i)[0] for i in range(2)], dtype=np.int32
        )  # Se extraen los índices de las celdas que contienen los puntos de los micrófonos y se almacenan en un array de tipo entero de 32 bits.

    def zs(self, f):
        w = 2.0 * np.pi * f
        k0 = w / C0
        Z0 = RHO0 * C0
        rho_eq, K_eq, _, _ = jca(
            f, **self.params
        )  # Se calculan las propiedades efectivas de la muestra porosa (densidad y módulo de compresibilidad) utilizando el modelo JCA para la frecuencia f y los parámetros proporcionados.

        # actualizar propiedades y constantes (las formas NO se recompilan). Esto con la condicion where(self.mask_poroso, rho_eq, RHO0) significa que en las celdas donde mask_poroso es True (dentro de la muestra porosa), se asigna rho_eq, y en las celdas donde mask_poroso is False (fuera de la muestra porosa), se asigna RHO0 (densidad del aire). Lo mismo ocurre para K_f.
        self.rho_f.x.array[:] = np.where(
            self.mask_poroso, rho_eq, RHO0
        )  # reemplaza los valores de la función rho_f con rho_eq en las celdas porosas y con RHO0 en las celdas de aire.
        self.K_f.x.array[:] = np.where(self.mask_poroso, K_eq, RHO0 * C0**2)
        self.c_w2.value = w**2
        self.c_rad.value = 1j * k0 / RHO0
        self.c_src.value = 2.0j * k0 * self.Pin / RHO0
        if np.isfinite(self.Zb):
            self.c_Zb.value = 1j * w / self.Zb
        # A las constantes simbolicas del problema se les asignan los valores correspondientes a la frecuencia actual f. Esto permite que el sistema lineal se resuelva con las propiedades correctas para cada frecuencia sin necesidad de recompilar las formas.

        uh = self.problem.solve()  # Campo de presión acústica complejo en toda la malla, resuelto a partir del problema lineal definido anteriormente. La variable uh contiene los valores de presión en cada nodo de la malla.

        P1, P2 = uh.eval(self.pts, self.cells)[:, 0]
        # [:, 0] extrae la componente x de la presión evaluada en los puntos de los micrófonos. P1 corresponde al micrófono en -x1 y P2 al micrófono en -x2.

        # Post-proceso ISO 10534-2 con R referido a x=0 (cara de la muestra)
        s = self.x1 - self.x2
        H12 = P2 / P1
        R = (
            (H12 - np.exp(-1j * k0 * s))
            / (np.exp(1j * k0 * s) - H12)
            * np.exp(2j * k0 * self.x1)
        )
        Zs = Z0 * (1.0 + R) / (1.0 - R)
        return Zs, R


# =============================================================================
# 5. UTSUNO (dos cavidades)
# =============================================================================
def utsuno(Zs1, Zs2, L1, L2, d, f):
    Z0 = RHO0 * C0
    k0 = 2.0 * np.pi * f / C0
    Z1 = -1j * Z0 / np.tan(k0 * L1)
    Z2 = -1j * Z0 / np.tan(k0 * L2)

    num = Zs1 * Zs2 * (Z1 - Z2) - Z1 * Z2 * (Zs1 - Zs2)
    den = (Z1 - Z2) - (Zs1 - Zs2)
    Zc = np.sqrt(num / den)
    Zc = np.where(Zc.real < 0, -Zc, Zc)

    X = ((Zs1 + Zc) * (Z1 - Zc)) / ((Zs1 - Zc) * (Z1 + Zc))
    fase = np.unwrap(
        np.angle(X)
    )  # np.unwrap elimina los saltos de fase de 2π en la fase de X, proporcionando una representación continua de la fase a lo largo de la frecuencia.
    kc = (np.log(np.abs(X)) + 1j * fase) / (2j * d)
    return Zc, kc


# =============================================================================
# 6. PROGRAMA PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    params = dict(
        phi=0.94, sigma=40000.0, alpha_inf=1.06, Lambda_v=60e-6, Lambda_t=120e-6
    )
    d = 0.03
    L1, L2 = 0.02, 0.04
    x1, x2 = 0.08, 0.06
    L_tubo = 0.12
    D_tubo = 0.05
    h = 2.5e-4  # Longitud de elemento de la malla (m). Ajustar para que los micrófonos y la interfaz caigan en nodos exactos.
    f_vec = np.linspace(
        200, 3500, 200
    )  # Frecuencia mínima y máxima (Hz) y número de puntos. Ajustar f_max < fc para que no aparezcan modos no planos.
    fc = 1.84 * C0 / (np.pi * D_tubo)
    print(f"Frecuencia de corte del tubo fc = 1.84·c0/(π·D) = {fc:.0f} Hz")
    if f_vec[-1] > fc:
        print("  ADVERTENCIA: f_max supera fc (aparecen modos no planos).")

    fem1 = TuboFEM(params, d, L1, x1, x2, L_tubo, h)
    fem2 = TuboFEM(params, d, L2, x1, x2, L_tubo, h)

    n = len(f_vec)
    Zs1_t, Zs2_t, Zs1_f, Zs2_f = (np.zeros(n, dtype=complex) for _ in range(4))
    Zc_ex, kc_ex = np.zeros(n, dtype=complex), np.zeros(n, dtype=complex)

    for i, f in enumerate(f_vec):
        _, _, Zc_ex[i], kc_ex[i] = jca(f, **params)
        Zs1_t[i], _ = zs_tmm(f, Zc_ex[i], kc_ex[i], d, L1)
        Zs2_t[i], _ = zs_tmm(f, Zc_ex[i], kc_ex[i], d, L2)
        Zs1_f[i], _ = fem1.zs(f)
        Zs2_f[i], _ = fem2.zs(f)

    Zc_t, kc_t = utsuno(Zs1_t, Zs2_t, L1, L2, d, f_vec)
    Zc_f, kc_f = utsuno(Zs1_f, Zs2_f, L1, L2, d, f_vec)

    # Se tiene Zc_t de la TMM y Zc_f del FEM, ambos usando el método de Utsuno para combinar las dos cavidades. Ahora se comparan los resultados de ambos métodos con el modelo JCA exacto.

    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b) * 100.0
    print("\n" + "=" * 62)
    print(f"Zs  FEM vs TMM (directo)        : {rel(Zs1_f, Zs1_t):.5f} %  (cav. 1)")
    print(f"Zc  TMM+Utsuno vs JCA exacto    : {rel(Zc_t, Zc_ex):.5f} %")
    print(f"kc  TMM+Utsuno vs JCA exacto    : {rel(kc_t, kc_ex):.5f} %")
    print(f"Zc  FEM+Utsuno vs TMM+Utsuno    : {rel(Zc_f, Zc_t):.5f} %")
    print(f"kc  FEM+Utsuno vs TMM+Utsuno    : {rel(kc_f, kc_t):.5f} %")
    print("=" * 62)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=120)
    fig.suptitle(
        "Método de dos cavidades (Utsuno): JCA exacto vs TMM vs FEM (DOLFINx)",
        fontweight="bold",
    )

    ax1.plot(f_vec, Zc_ex.real, "k-", lw=2.5, label=r"Re($Z_c$) JCA exacto")
    ax1.plot(f_vec, Zc_t.real, "c:", lw=2, label=r"Re($Z_c$) TMM+Utsuno")
    ax1.plot(f_vec, Zc_f.real, "r--", lw=1.5, label=r"Re($Z_c$) FEM+Utsuno")
    ax1.plot(
        f_vec, Zc_ex.imag, "-", color="gray", lw=2.5, label=r"Im($Z_c$) JCA exacto"
    )
    ax1.plot(f_vec, Zc_t.imag, "y:", lw=2, label=r"Im($Z_c$) TMM+Utsuno")
    ax1.plot(f_vec, Zc_f.imag, "b--", lw=1.5, label=r"Im($Z_c$) FEM+Utsuno")
    ax1.set(
        xlabel="Frecuencia [Hz]",
        ylabel=r"$Z_c$ [Pa·s/m]",
        title=f"Impedancia característica (err FEM-TMM: {rel(Zc_f, Zc_t):.4f}%)",
    )
    ax1.grid(alpha=0.5)
    ax1.legend(fontsize=8)

    ax2.plot(f_vec, kc_ex.real, "k-", lw=2.5, label=r"Re($k_c$) JCA exacto")
    ax2.plot(f_vec, kc_t.real, "c:", lw=2, label=r"Re($k_c$) TMM+Utsuno")
    ax2.plot(f_vec, kc_f.real, "g--", lw=1.5, label=r"Re($k_c$) FEM+Utsuno")
    ax2.plot(
        f_vec, kc_ex.imag, "-", color="gray", lw=2.5, label=r"Im($k_c$) JCA exacto"
    )
    ax2.plot(f_vec, kc_t.imag, "y:", lw=2, label=r"Im($k_c$) TMM+Utsuno")
    ax2.plot(f_vec, kc_f.imag, "m--", lw=1.5, label=r"Im($k_c$) FEM+Utsuno")
    ax2.set(
        xlabel="Frecuencia [Hz]",
        ylabel=r"$k_c$ [rad/m]",
        title=f"Número de onda (err FEM-TMM: {rel(kc_f, kc_t):.4f}%)",
    )
    ax2.grid(alpha=0.5)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("validacion_utsuno_tmm_fem_dolfinx.png", dpi=200)
    plt.show()
