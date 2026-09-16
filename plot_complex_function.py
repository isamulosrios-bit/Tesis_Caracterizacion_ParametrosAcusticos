import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -------------------------------------------------
# Domain definition
# -------------------------------------------------
xs, xe = -2*np.pi, 2*np.pi
ys, ye = -2.0, 2.0
Nx, Ny = 300, 200

x = np.linspace(xs, xe, Nx)
y = np.linspace(ys, ye, Ny)
X, Y = np.meshgrid(x, y)

# -------------------------------------------------
# Complex function
# -------------------------------------------------
Z = X + 1j * Y
F = np.exp(1j * Z)

ReF  = np.real(F)
ImF  = np.imag(F)
AbsF = np.abs(F)

# -------------------------------------------------
# Figure layout: 2 rows × 3 columns
# -------------------------------------------------
fig = plt.figure(figsize=(18, 10))

data_sets = [
    (ReF,  r"$\Re(e^{\mathrm{i}z})$"),
    (ImF,  r"$\Im(e^{\mathrm{i}z})$"),
    (AbsF, r"$|e^{\mathrm{i}z}|$")
]

# -------------------------------------------------
# First row: contour plots
# -------------------------------------------------
for i, (data, title) in enumerate(data_sets, start=1):
    ax = fig.add_subplot(2, 3, i)
    c = ax.contourf(X, Y, data, levels=40)
    fig.colorbar(c, ax=ax)
    ax.set_title(title + " (contour)")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

# -------------------------------------------------
# Second row: 3D surface plots
# -------------------------------------------------
for i, (data, title) in enumerate(data_sets, start=4):
    ax = fig.add_subplot(2, 3, i, projection="3d")
    surf = ax.plot_surface(
        X, Y, data,
        rstride=2, cstride=2,
        linewidth=0,
        antialiased=True)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -------------------------------------------------
# Domain definition
# -------------------------------------------------
xs, xe = -2*np.pi, 2*np.pi
ys, ye = -2.0, 2.0
Nx, Ny = 300, 200

x = np.linspace(xs, xe, Nx)
y = np.linspace(ys, ye, Ny)
X, Y = np.meshgrid(x, y)

# -------------------------------------------------
# Complex function
# -------------------------------------------------
Z = X + 1j * Y
F = np.exp(1j * Z)

ReF  = np.real(F)
ImF  = np.imag(F)
AbsF = np.abs(F)

# -------------------------------------------------
# Figure layout: 2 rows × 3 columns
# -------------------------------------------------
fig = plt.figure(figsize=(18, 10))

data_sets = [
    (ReF,  r"$\Re(e^{\mathrm{i}z})$"),
    (ImF,  r"$\Im(e^{\mathrm{i}z})$"),
    (AbsF, r"$|e^{\mathrm{i}z}|$")
]

# -------------------------------------------------
# First row: contour plots
# -------------------------------------------------
for i, (data, title) in enumerate(data_sets, start=1):
    ax = fig.add_subplot(2, 3, i)
    c = ax.contourf(X, Y, data, levels=40)
    fig.colorbar(c, ax=ax)
    ax.set_title(title + " (contour)")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

# -------------------------------------------------
# Second row: 3D surface plots
# -------------------------------------------------
for i, (data, title) in enumerate(data_sets, start=4):
    ax = fig.add_subplot(2, 3, i, projection="3d")
    surf = ax.plot_surface(
        X, Y, data,
        rstride=2, cstride=2,
        linewidth=0,
        antialiased=True
    )
    fig.colorbar(surf, ax=ax, shrink=0.6)
    ax.set_title(title + " (surface)")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel("value")

plt.tight_layout()
plt.show()

fig.colorbar(surf, ax=ax, shrink=0.6)
ax.set_title(title + " (surface)")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel("value")

plt.tight_layout()
plt.show()

