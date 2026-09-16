import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def resolver_sistema_simbolico(usar_sommerfeld=False):
    A1, B1, A2, B2 = sp.symbols('A_1 B_1 A_2 B_2', complex=True)
    k1, k2 = sp.symbols('k_1 k_2', real=True, positive=True)
    rho1, rho2 = sp.symbols(r'\rho_1 \rho_2', real=True, positive=True)
    Pin = sp.symbols('P_{in}', real=True)
    
    # Ecuaciones base comunes
    eq1 = sp.Eq(A1 + B1, Pin)
    eq2 = sp.Eq(A1 * sp.exp(-0.5 * sp.I * k1) + B1 * sp.exp(0.5 * sp.I * k1) - 
                (A2 * sp.exp(-0.5 * sp.I * k2) + B2 * sp.exp(0.5 * sp.I * k2)), 0)
    eq3 = sp.Eq((-sp.I * k1 / rho1) * (A1 * sp.exp(-0.5 * sp.I * k1) - B1 * sp.exp(0.5 * sp.I * k1)) - 
                ((-sp.I * k2 / rho2) * A2 * sp.exp(-0.5 * sp.I * k2) + (sp.I * k2 / rho2) * B2 * sp.exp(1.0 * sp.I * k2)), 0)
    
    # Condición de contorno en el extremo derecho (x = 1.0)
    if not usar_sommerfeld:
        # Pared Rígida: dP/dx = 0
        eq4 = sp.Eq(-sp.I * k2 * A2 * sp.exp(-sp.I * k2) + sp.I * k2 * B2 * sp.exp(1.0 * sp.I * k2), 0)
        sufijo = "pared_rigida"
        titulo_cond = "Pared Rígida"
    else:
        # Sommerfeld: dP/dx - j k P = 0
        eq4 = sp.Eq((-sp.I * k2 * A2 * sp.exp(-sp.I * k2) + sp.I * k2 * B2 * sp.exp(1.0 * sp.I * k2)) - 
                    sp.I * k2 * (A2 * sp.exp(-sp.I * k2) + B2 * sp.exp(1.0 * sp.I * k2)), 0)
        sufijo = "sommerfeld"
        titulo_cond = "Condición de Sommerfeld"
        
    sol = sp.linsolve([eq1, eq2, eq3, eq4], (A1, B1, A2, B2))
    s_A1, s_B1, s_A2, s_B2 = [sp.simplify(term) for term in list(sol)[0]]
    
    # Generar la imagen correspondiente
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(rf"$\mathbf{{Solución\ Analítica\ Simbólica\ - \ Sistema\ Bicapa\ ({titulo_cond})}}$", fontsize=14, y=0.96)
    
    ax1 = fig.add_subplot(411)
    ax1.axis('off')
    ax1.set_title(r"$A_1 = " + sp.latex(s_A1) + r"$", fontsize=9, pad=12)
    
    ax2 = fig.add_subplot(412)
    ax2.axis('off')
    ax2.set_title(r"$B_1 = " + sp.latex(s_B1) + r"$", fontsize=9, pad=12)
    
    ax3 = fig.add_subplot(413)
    ax3.axis('off')
    ax3.set_title(r"$A_2 = " + sp.latex(s_A2) + r"$", fontsize=9, pad=12)
    
    ax4 = fig.add_subplot(414)
    ax4.axis('off')
    ax4.set_title(r"$B_2 = " + sp.latex(s_B2) + r"$", fontsize=9, pad=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    nombre_pdf = f"solucion_{sufijo}.pdf"
    nombre_png = f"solucion_{sufijo}.png"
    
    plt.savefig(nombre_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generado con éxito: {nombre_png}")

if __name__ == "__main__":
    print("Calculando y exportando casos...")
    resolver_sistema_simbolico(usar_sommerfeld=False)
    resolver_sistema_simbolico(usar_sommerfeld=True)
    print("¡Proceso completado!")