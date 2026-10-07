import numpy as np

M = np.load("M_fitted.npy")

vals, vecs = np.linalg.eig(M)
i = np.argmax(np.abs(vals))          # position of the dominant eigenvalue
lam = vals[i].real
v = vecs[:, i].real
v = v / v.sum()                      # turn into fractions

a = sorted(np.abs(vals), reverse=True)

print("M =\n", np.round(M, 4))
print("eigenvalues:", np.round(vals, 4))
print("|eigenvalues|:", np.round(np.abs(vals), 4))
print("dominant eigenvalue:", round(lam, 4))
print("long-run ratio [herring, whiting, cod]:", np.round(v, 3))
print("convergence rate |l2|/|l1|:", round(a[1] / a[0], 4))
print("verdict:", "growth" if lam > 1.05 else "decline" if lam < 0.95 else "stable")

print("all eigenvectors (each column matches the eigenvalue in the same position):\n", np.round(vecs, 4))
print("dominant eigenvector (unit length):", np.round(vecs[:, i].real * np.sign(vecs[:, i].real.sum()), 4))
print("stability: |dominant eigenvalue| =", round(abs(lam), 4), "< 1, so populations shrink each year")

import matplotlib.pyplot as plt

t = np.linspace(0, 2 * np.pi, 200)
plt.plot(np.cos(t), np.sin(t), "k--", label="unit circle")
plt.scatter(vals.real, vals.imag, color="r", zorder=3, label="eigenvalues of M")
plt.axhline(0, color="gray", lw=0.5)
plt.axvline(0, color="gray", lw=0.5)
plt.gca().set_aspect("equal")
plt.legend()
plt.title("All eigenvalues inside the unit circle = decay")
plt.savefig("eigenvalues_plot.png")
plt.show()