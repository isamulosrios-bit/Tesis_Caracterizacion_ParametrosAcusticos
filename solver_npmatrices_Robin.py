import matplotlib
import sympy as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def resolver_sistema_simbolico_oficial():
    """Solución analítica simbólica oficial para el sistema acústico 1D bicapa

    con condición de Neumann no homogénea en la entrada (x = 0) y Robin en la salida
    (x = L). Exporta en PNG usando mathtext y muestra el LaTeX por consola.
    """
    # 1. Definir símbolos algebraicos en modo simbólico puro
    A1, B1, A2, B2 = sp.symbols("A_1 B_1 A_2 B_2", complex=True)
    k1, k2 = sp.symbols("k_1 k_2", real=True, positive=True)
    rho1, rho2 = sp.symbols(r"\rho_1 \rho_2", real=True, positive=True)
    vin = sp.symbols("v_{in}", real=True)
    omega = sp.symbols(r"\omega", real=True, positive=True)
    zs = sp.symbols("Z_s", real=True, positive=True)
    L = sp.symbols("L", real=True, positive=True)
    xint = sp.symbols("x_{int}", real=True, positive=True)

    # 2. Plantear las 4 ecuaciones del sistema 4x4
    # Eq 1: Condición de Neumann no homogénea en la entrada (x = 0) -> v_1(0) = v_in
    eq1 = sp.Eq(A1 - B1, (omega * rho1 * vin) / k1)

    # Eq 2: Continuidad de Presión en la interfaz (x = xint) -> P_1(xint) = P_2(xint)
    eq2 = sp.Eq(
        B1 * (sp.exp(xint * sp.I * k1) + sp.exp(-xint * sp.I * k1))
        - A2 * sp.exp(-xint * sp.I * k2)
        - B2 * sp.exp(xint * sp.I * k2),
        -((omega * rho1 * vin) / k1) * sp.exp(-xint * sp.I * k1),
    )

    # Eq 3: Continuidad de Velocidad Normal en la interfaz (x = xint) -> v_1(xint) = v_2(xint)
    eq3 = sp.Eq(
        (k1 / rho1) * B1 * (sp.exp(xint * sp.I * k1) - sp.exp(-xint * sp.I * k1))
        - (k2 / rho2)
        * (-A2 * sp.exp(-xint * sp.I * k2) + B2 * sp.exp(xint * sp.I * k2)),
        omega * vin * sp.exp(-xint * sp.I * k1),
    )

    # Eq 4: Condición de Robin en la frontera derecha (x = L) -> P_2(L) = Z_s * v_2(L)
    eq4 = sp.Eq(
        A2 * (omega * rho2 / zs - k2) * sp.exp(-L * sp.I * k2)
        + B2 * (omega * rho2 / zs + k2) * sp.exp(L * sp.I * k2),
        0,
    )

    # 3. Resolver el sistema lineal y simplificar
    sol = sp.linsolve([eq1, eq2, eq3, eq4], (A1, B1, A2, B2))
    s_A1, s_B1, s_A2, s_B2 = [sp.simplify(term) for term in list(sol)[0]]

    constantes = [
        (r"A_1", s_A1),
        (r"B_1", s_B1),
        (r"A_2", s_A2),
        (r"B_2", s_B2),
    ]

    # Imprimir por consola (bash) los códigos LaTeX listos para copiar y pegar en el PDF
    print("\n" + "=" * 80)
    print(" CÓDIGOS LATEX PARA COPIAR Y PEGAR EN TU TESIS:")
    print("=" * 80)
    for nombre, expr in constantes:
        latex_code = sp.latex(expr)
        print(f"\n% --- Coeficiente {nombre} ---")
        print(f"\\begin{{equation}}\n    {nombre} = {latex_code}\n\\end{{equation}}")
    print("\n" + "=" * 80 + "\n")

    # 4. Generar el reporte visual PNG utilizando Matplotlib mathtext corregido
    fig = plt.figure(figsize=(16, 22), dpi=300)
    fig.suptitle(
        r"$\mathbf{Solución\ Analítica\ Simbólica\ -\ Sistema\ Bicapa\ (Neumann-Robin)}$",
        fontsize=14,
        y=0.98,
        fontweight="bold",
    )

    for idx, (nombre, expr) in enumerate(constantes, 1):
        ax = fig.add_subplot(4, 1, idx)
        ax.axis("off")

        latex_expr_str = sp.latex(expr)
        # Usamos bloques separados y permitimos que Matplotlib renderice adecuadamente
        render_str = f"${nombre} = {latex_expr_str}$"

        ax.text(
            0.5,
            0.5,
            render_str,
            fontsize=6.5,  # Tamaño reducido para evitar desbordes visuales en el PNG
            ha="center",
            va="center",
            transform=ax.transAxes,
            wrap=True,
        )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

    nombre_png = "solucion_analitica_oficial.png"
    plt.savefig(nombre_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"¡Imagen PNG generada con éxito como: {nombre_png}!")
    return s_A1, s_B1, s_A2, s_B2


if __name__ == "__main__":
    print("Calculando el sistema analítico bicapa...")
    resolver_sistema_simbolico_oficial()
    print("¡Proceso completado!")
