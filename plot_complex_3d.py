import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -------------------------------------------------
# User-defined complex function f(z)
# -------------------------------------------------
def f(z):
    return np.exp(1j * z)   # example: exp(i z)

# -------------------------------------------------
# Which part to plot: "re", "im", or "abs"
# -------------------------------------------------
plot_part = "im"   # options: "re", "im", "abs"

# -------------------------------------------------
# Domain definition
# -------------------------------------------------
xs, xe = -2*np.pi, 2*np.pi
ys, ye = -2.0, 2.0
Nx, Ny = 300, 200

x = np.linspace(xs, xe, Nx)
y = np.linspace(ys, ye, Ny)
X, Y = np.meshgrid(x, y)

Z = X + 1j * Y
F = f(Z)

# -------------------------------------------------
# Select data to plot
# -------------------------------------------------
if plot_part == "re":
    data = np.real(F)
    zlabel = r"$\Re(f(z))$"
elif plot_part == "im":
    data = np.imag(F)
    zlabel = r"$\Im(f(z))$"
elif plot_part == "abs":
    data = np.abs(F)
    zlabel = r"$|f(z)|$"
else:
    raise ValueError("plot_part must be 're', 'im', or 'abs'")

# -------------------------------------------------
# 3D surface plot
# -------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    X, Y, data,
    rstride=2,
    cstride=2,
    linewidth=0,
    antialiased=True
)

fig.colorbar(surf, ax=ax, shrink=0.7)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(zlabel)

ax.set_title(r"3D plot of " + zlabel)

plt.tight_layout()
plt.show()

