
import numpy as np
import matplotlib.pyplot as plt

SPECIES = ["Herring", "Whiting", "Cod"]

results = np.load("model_results.npz")

M = results["M"]
X = results["X"]
Y = results["Y"]
years = results["years"]

print("=" * 60)
print("PERSON 4 - DIAGONALIZATION, FORECASTING AND CONVERGENCE")
print("=" * 60)

print("\nTransition Matrix M:")
print(np.round(M, 4))


# ============================================================
# 1. DIAGONALIZATION
# ============================================================

eigenvalues, eigenvectors = np.linalg.eig(M)

P = eigenvectors
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

print("\n1. DIAGONALIZATION")

print("\nEigenvalues:")
print(np.round(eigenvalues, 4))

print("\nP:")
print(np.round(P, 4))

print("\nD:")
print(np.round(D, 4))

print("\nP^-1:")
print(np.round(P_inv, 4))

M_reconstructed = P @ D @ P_inv

diagonalization_error = np.linalg.norm(M - M_reconstructed)

print("\nReconstructed M = P D P^-1:")
print(np.round(M_reconstructed, 4))

print("\nDiagonalization error:")
print(diagonalization_error)

if np.linalg.matrix_rank(P) == 3:
    print("Result: M is diagonalizable.")
else:
    print("Result: M does not have three independent eigenvectors.")


# ============================================================
# 2. VERIFY M^n = P D^n P^-1
# ============================================================

print("\n2. VERIFYING M^n")

n = 5

M_power_direct = np.linalg.matrix_power(M, n)

D_power = np.linalg.matrix_power(D, n)

M_power_diagonalized = P @ D_power @ P_inv

power_error = np.linalg.norm(
    M_power_direct - M_power_diagonalized
)

print("\nn =", n)

print("\nM^n using direct multiplication:")
print(np.round(M_power_direct, 4))

print("\nM^n using diagonalization:")
print(np.round(M_power_diagonalized, 4))

print("\nDifference between the two:")
print(power_error)


# ============================================================
# 3. FUTURE FORECAST
# ============================================================

print("\n3. FUTURE FORECAST")

# The model uses Y = X @ M.T.
# Y contains the next-year states corresponding to X.
# Start from the final observed population in Y.

last_population = Y[-1].copy()
last_observed_year = int(years[-1]) + 1

future_years_count = 10

future_predictions = []

current_population = last_population.copy()

for i in range(future_years_count):

    current_population = current_population @ M.T

    future_predictions.append(
        current_population.copy()
    )

future_predictions = np.real_if_close(
    np.array(future_predictions)
).real

future_years = np.arange(
    last_observed_year + 1,
    last_observed_year + future_years_count + 1
)

print("\nLast observed year:", last_observed_year)

print("\nStarting population:")
print(np.round(last_population, 2))

print("\nFuture predictions:")

for i in range(future_years_count):

    print(
        int(future_years[i]),
        "Herring =", round(future_predictions[i, 0], 2),
        "Whiting =", round(future_predictions[i, 1], 2),
        "Cod =", round(future_predictions[i, 2], 2)
    )


# ============================================================
# 4. FUTURE FORECAST GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    future_years,
    future_predictions[:, 0],
    marker="o",
    label="Herring"
)

plt.plot(
    future_years,
    future_predictions[:, 1],
    marker="o",
    label="Whiting"
)

plt.plot(
    future_years,
    future_predictions[:, 2],
    marker="o",
    label="Cod"
)

plt.xlabel("Year")
plt.ylabel("Predicted Population")
plt.title("Future Ecosystem Population Forecast")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "future_forecast.png",
    dpi=300
)

plt.show()


# ============================================================
# 5. CONVERGENCE ANALYSIS
# ============================================================

print("\n5. CONVERGENCE ANALYSIS")

initial_vectors = np.array([
    [100, 100, 100],
    [1000, 100, 20],
    [50, 1000, 50],
    [500, 500, 500]
], dtype=float)

iterations = 20

all_ratios = []

for initial in initial_vectors:

    current = initial.copy()

    ratios = []

    for i in range(iterations + 1):

        total = np.sum(current)

        if total <= 0 or np.any(current < 0):
            break

        ratio = current / total

        ratios.append(ratio)

        current = current @ M.T

    all_ratios.append(np.array(ratios))


print("\nNormalized population proportions at the final valid iteration:")

for i, ratios in enumerate(all_ratios):

    print("\nInitial population:")
    print(initial_vectors[i])

    if len(ratios) > 0:
        print("Final normalized components:")
        print(np.round(ratios[-1], 4))
    else:
        print("No valid nonnegative trajectory.")


# ============================================================
# 6. CONVERGENCE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

for i, ratios in enumerate(all_ratios):

    if len(ratios) == 0:
        continue

    steps = range(len(ratios))

    for j, species in enumerate(SPECIES):

        plt.plot(
            steps,
            ratios[:, j],
            label=f"Initial {i + 1} - {species}"
        )

plt.xlabel("Iteration")
plt.ylabel("Population Proportion")

plt.title("Convergence of Population-State Components")

plt.legend(fontsize=7)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "convergence_analysis.png",
    dpi=300
)

plt.show()


# ============================================================
# 7. DOMINANT EIGENVALUE
# ============================================================

print("\n7. DOMINANT EIGENVALUE")

dominant_index = np.argmax(
    np.abs(eigenvalues)
)

dominant_eigenvalue = eigenvalues[
    dominant_index
]

spectral_radius = abs(dominant_eigenvalue)

print("Dominant eigenvalue:")
print(dominant_eigenvalue)

print("\nMagnitude of dominant eigenvalue:")
print(spectral_radius)

if spectral_radius < 1:

    print("Dominant mode decays.")

elif spectral_radius > 1:

    print("Dominant mode grows.")

else:

    print("Dominant mode has constant magnitude.")
