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