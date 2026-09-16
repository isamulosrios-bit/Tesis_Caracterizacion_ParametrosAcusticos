import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Parameters
# -------------------------------
k = 5.0              # wavenumber
L = 2.0              # length of the contour
N = 500              # number of points along contour
alpha = 0.3          # imaginary part slope of the contour

# -------------------------------
# Complex contour
# -------------------------------
s = np.linspace(0, L, N)
x = s + 1j * alpha * s  # complex contour

# -------------------------------
# Helmholtz solution along contour
# -------------------------------
A = 1.0   # amplitude of outgoing wave
B = 0.0   # amplitude of incoming wave (0 for outgoing only)

u = A * np.exp(1j * k * x) + B * np.exp(-1j * k * x)

# -------------------------------
# Plot results
# -------------------------------
plt.figure(figsize=(14, 4))

# Real part
plt.subplot(1,3,1)
plt.plot(s, np.real(u))
plt.title("Re(u)")
plt.xlabel("s (along contour)")
plt.grid(True)

# Imaginary part
plt.subplot(1,3,2)
plt.plot(s, np.imag(u))
plt.title("Im(u)")
plt.xlabel("s (along contour)")
plt.grid(True)

# Absolute value
plt.subplot(1,3,3)
plt.plot(s, np.abs(u))
plt.title("|u| (modulus)")
plt.xlabel("s (along contour)")
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------
# Optional: 2D visualization of |u| in complex plane
# -------------------------------
# grid in complex plane
x_real = np.linspace(0, L, 200)
x_imag = np.linspace(0, alpha*L, 200)
X, Y = np.meshgrid(x_real, x_imag)
Z = X + 1j*Y
U = A * np.exp(1j*k*Z)

plt.figure(figsize=(6,5))
plt.contourf(X, Y, np.abs(U), levels=50, cmap='viridis')
plt.colorbar(label='|u|')
plt.title("Modulus of Helmholtz solution in complex plane")
plt.xlabel("Re(x)")
plt.ylabel("Im(x)")
plt.show()

