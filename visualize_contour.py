import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Parameters
# -------------------------------
L_pml = 2.0
sigma_max = 5.0
k = 5.0
N = 600

x = np.linspace(0, L_pml, N)

# -------------------------------
# Damping profiles
# -------------------------------
def sigma_linear(x):
    return sigma_max * x / L_pml

def sigma_quadratic(x):
    return sigma_max * (x / L_pml)**2

def sigma_cubic(x):
    return sigma_max * (x / L_pml)**3

profiles = {
    "linear": sigma_linear,
    "quadratic": sigma_quadratic,
    "cubic": sigma_cubic
}

# -------------------------------
# Compute complex contours
# -------------------------------
plt.figure(figsize=(7,5))

for name, sigma_fun in profiles.items():
    sigma = sigma_fun(x)
    dZdx = 1 + 1j * sigma / k
    Z = np.cumsum(dZdx) * (x[1] - x[0])
    plt.plot(Z.real, Z.imag, label=name)

# Reference: real axis
plt.plot(x, np.zeros_like(x), "k--", linewidth=1, label="real axis")

plt.xlabel("Re(Z)")
plt.ylabel("Im(Z)")
plt.title("PML Complex Contours for Different Damping Profiles")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()

