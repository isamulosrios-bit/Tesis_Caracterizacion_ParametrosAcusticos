import matplotlib
import sympy as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def resolver_sistema_simbolico(usar_sommerfeld=False):
    A1, B1, A2, B2 = sp.symbols("A_1 B_1 A_2 B_2", complex=True)
    k1, k2 = sp.symbols("k_1 k_2", real=True, positive=True)
    rho1, rho2 = sp.symbols(r"\rho_1 \rho_2", real=True, positive=True)
    Pin = sp.symbols("P_{in}", real=True)
    L = sp.symbols("L", real=True, positive=True)
    xint = sp.symbols("x_{int}", real=True, positive=True)

    # Ecuaciones base comunes
    eq1 = sp.Eq(A1 + B1, Pin)
    eq2 = sp.Eq(
        A1 * sp.exp(-xint * sp.I * k1)
        + B1 * sp.exp(xint * sp.I * k1)
        - (A2 * sp.exp(-xint * sp.I * k2) + B2 * sp.exp(xint * sp.I * k2)),
        0,
    )
    eq3 = sp.Eq(
        (-sp.I * k1 / rho1)
        * (A1 * sp.exp(-xint * sp.I * k1) - B1 * sp.exp(xint * sp.I * k1))
        - (
            (-sp.I * k2 / rho2) * A2 * sp.exp(-xint * sp.I * k2)
            + (sp.I * k2 / rho2) * B2 * sp.exp(L * sp.I * k2)
        ),
        0,
    )

    # Condición de contorno en el extremo derecho
    if not usar_sommerfeld:
        # Pared Rígida: dP/dx = 0
        eq4 = sp.Eq(
            -sp.I * k2 * A2 * sp.exp(-sp.I * k2)
            + sp.I * k2 * B2 * sp.exp(L * sp.I * k2),
            0,
        )
        sufijo = "pared_rigida"
        titulo_cond = "Pared Rígida"
    else:
        # Sommerfeld: dP/dx - j k P = 0
        eq4 = sp.Eq(
            (
                -sp.I * k2 * A2 * sp.exp(-sp.I * k2)
                + sp.I * k2 * B2 * sp.exp(L * sp.I * k2)
            )
            - sp.I * k2 * (A2 * sp.exp(-sp.I * k2) + B2 * sp.exp(L * sp.I * k2)),
            0,
        )
        sufijo = "sommerfeld"
        titulo_cond = "Condición de Sommerfeld"

    # Resolver el sistema lineal
    sol = sp.linsolve([eq1, eq2, eq3, eq4], (A1, B1, A2, B2))
    s_A1, s_B1, s_A2, s_B2 = [sp.simplify(term) for term in list(sol)[0]]

    constantes = [
        (r"A_1", s_A1),
        (r"B_1", s_B1),
        (r"A_2", s_A2),
        (r"B_2", s_B2),
    ]

    # --- SALIDA EN BASH (TERMINAL) ---
    print("\n" + "=" * 90)
    print(f" CÓDIGOS LATEX PARA LA TESIS - CASO: {titulo_cond.upper()} ")
    print("=" * 90)
    for nombre, expr in constantes:
        latex_code = sp.latex(expr)
        print(f"\n% --- Coeficiente {nombre} ({sufijo}) ---")
        print(f"\\begin{{equation}}\n    {nombre} = {latex_code}\n\\end{{equation}}")
    print("\n" + "=" * 90 + "\n")

    # Generar la imagen PNG correspondiente
    fig = plt.figure(figsize=(16, 20), dpi=300)
    fig.suptitle(
        rf"$\mathbf{{Solución\ Analítica\ Simbólica\ - \ Sistema\ Bicapa\ ({titulo_cond})}}$",
        fontsize=14,
        y=0.96,
        fontweight="bold",
    )

    for idx, (nombre, expr) in enumerate(constantes, 1):
        ax = fig.add_subplot(4, 1, idx)
        ax.axis("off")
        render_str = f"${nombre} = {sp.latex(expr)}$"
        ax.text(
            0.5,
            0.5,
            render_str,
            fontsize=6,  # Tamaño reducido para evitar desbordes gráficos
            ha="center",
            va="center",
            transform=ax.transAxes,
            wrap=True,
        )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    nombre_png = f"solucion_{sufijo}.png"
    plt.savefig(nombre_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Imagen PNG generada con éxito: {nombre_png}")


if __name__ == "__main__":
    print("Calculando y exportando casos analíticos...")
    resolver_sistema_simbolico(usar_sommerfeld=False)
    resolver_sistema_simbolico(usar_sommerfeld=True)
    print("¡Proceso completado con éxito!")
