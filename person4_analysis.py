import numpy as np
import matplotlib.pyplot as plt

SPECIES = ["Herring", "Whiting", "Cod"]

# ============================================================
# 1. LOAD EXISTING MODEL RESULTS
# ============================================================

results = np.load("model_results.npz")

M = results["M"]
X = results["X"]
Y = results["Y"]
years = results["years"]

print("============================================================")
print("PERSON 4 - DIAGONALIZATION, FORECASTING AND CONVERGENCE")
print("============================================================")

print("\nTransition Matrix M:")
print(np.round(M, 4))


# ============================================================
# 2. DIAGONALIZATION
# ============================================================

eigenvalues, eigenvectors = np.linalg.eig(M)

P = eigenvectors
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

print("\n------------------------------------------------------------")
print("2. DIAGONALIZATION")
print("------------------------------------------------------------")

print("\nEigenvalues:")
print(np.round(eigenvalues, 4))

print("\nP:")
print(np.round(P, 4))

print("\nD:")
print(np.round(D, 4))

print("\nP^-1:")
print(np.round(P_inv, 4))


# Verify M = P D P^-1

M_reconstructed = P @ D @ P_inv

diagonalization_error = np.linalg.norm(
    M - M_reconstructed
)

print("\nReconstructed M = P D P^-1:")
print(np.round(M_reconstructed, 4))

print("\nDiagonalization error:")
print(diagonalization_error)

if diagonalization_error < 1e-10:
    print("Result: M is successfully diagonalized.")
else:
    print("Result: Numerical reconstruction has a noticeable error.")


# ============================================================
# 3. VERIFY M^n = P D^n P^-1
# ============================================================

print("\n------------------------------------------------------------")
print("3. VERIFYING M^n")
print("------------------------------------------------------------")

n = 5

M_power_direct = np.linalg.matrix_power(M, n)

D_power = np.linalg.matrix_power(D, n)

M_power_diagonalized = P @ D_power @ P_inv

power_error = np.linalg.norm(
    M_power_direct - M_power_diagonalized
)

print("\nn =", n)

print("\nM^n using direct matrix multiplication:")
print(np.round(M_power_direct, 4))

print("\nM^n using diagonalization:")
print(np.round(M_power_diagonalized, 4))

print("\nDifference between the two:")
print(power_error)


# ============================================================
# 4. MATRIX NORM
# ============================================================

print("\n------------------------------------------------------------")
print("4. MATRIX NORM")
print("------------------------------------------------------------")

matrix_norm = np.linalg.norm(M, "fro")

print("Frobenius norm of M:")
print(round(matrix_norm, 4))


# ============================================================
# 5. FUTURE FORECAST
# ============================================================

print("\n------------------------------------------------------------")
print("5. FUTURE FORECAST")
print("------------------------------------------------------------")

# The project uses:
# Y = X @ M.T
#
# Therefore:
# next_population = current_population @ M.T

last_population = X[-1].copy()

future_years_count = 10

future_predictions = []

current_population = last_population.copy()

for i in range(future_years_count):

    current_population = current_population @ M.T

    future_predictions.append(
        current_population.copy()
    )

future_predictions = np.array(future_predictions)

future_years = np.arange(
    int(years[-1]) + 1,
    int(years[-1]) + future_years_count + 1
)

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
# 6. FUTURE FORECAST GRAPH
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
# 7. CONVERGENCE ANALYSIS
# ============================================================

print("\n------------------------------------------------------------")
print("7. CONVERGENCE ANALYSIS")
print("------------------------------------------------------------")

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

    for i in range(iterations):

        total = np.sum(np.abs(current))

        if total != 0:

            ratio = np.abs(current) / total

        else:

            ratio = np.zeros(3)

        ratios.append(ratio)

        current = current @ M.T

    all_ratios.append(
        np.array(ratios)
    )

all_ratios = np.array(all_ratios)

print("\nNormalized absolute population components after 20 iterations:")

for i in range(len(initial_vectors)):

    print("\nInitial population:")
    print(initial_vectors[i])

    print("Final normalized components:")

    print(
        np.round(
            all_ratios[i, -1],
            4
        )
    )


# ============================================================
# 8. CONVERGENCE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

for i in range(len(initial_vectors)):

    plt.plot(
        range(iterations),
        all_ratios[i, :, 0],
        label=f"Initial {i + 1} - Herring"
    )

    plt.plot(
        range(iterations),
        all_ratios[i, :, 1],
        linestyle="--",
        label=f"Initial {i + 1} - Whiting"
    )

    plt.plot(
        range(iterations),
        all_ratios[i, :, 2],
        linestyle=":",
        label=f"Initial {i + 1} - Cod"
    )

plt.xlabel("Iteration")
plt.ylabel("Normalized Absolute Component")

plt.title(
    "Convergence of Population-State Components"
)

plt.legend(fontsize=7)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "convergence_analysis.png",
    dpi=300
)

plt.show()


# ============================================================
# 9. DOMINANT EIGENVALUE
# ============================================================

print("\n------------------------------------------------------------")
print("9. DOMINANT EIGENVALUE")
print("------------------------------------------------------------")

dominant_index = np.argmax(
    np.abs(eigenvalues)
)

dominant_eigenvalue = eigenvalues[
    dominant_index
]

print("Dominant eigenvalue:")
print(dominant_eigenvalue)

print("\nMagnitude of dominant eigenvalue:")
print(abs(dominant_eigenvalue))

if abs(dominant_eigenvalue) < 1:

    print(
        "Long-term behavior: "
        "population magnitude tends to decline."
    )

elif abs(dominant_eigenvalue) > 1:

    print(
        "Long-term behavior: "
        "population magnitude tends to grow."
    )

else:

    print(
        "Long-term behavior: "
        "approximately persistent."
    )


# ============================================================
# 10. MODEL LIMITATIONS
# ============================================================

print("\n------------------------------------------------------------")
print("10. MODEL LIMITATIONS")
print("------------------------------------------------------------")

print("""
The transition matrix assumes that the relationship between
the populations from one year to the next remains approximately
constant.

Real ecosystems are more complicated because they are affected by:

1. Weather and climate changes
2. Disease and parasites
3. Food availability
4. Fishing and human intervention
5. Habitat changes
6. Migration
7. Seasonal effects
8. Nonlinear predator-prey interactions
9. Changes in environmental conditions

The fitted transition matrix can also produce negative components
when repeatedly applied. Negative populations are not physically
meaningful, showing that the linear model has limitations for
long-term forecasting.

Therefore, the linear transition matrix provides a simplified
mathematical model rather than an exact representation of the
real ecosystem.
""")


# ============================================================
# 11. SUMMARY
# ============================================================

print("\n============================================================")
print("PERSON 4 SUMMARY")
print("============================================================")

print(
    "1. Diagonalization error:",
    diagonalization_error
)

print(
    "2. M^n verification error:",
    power_error
)

print(
    "3. Frobenius norm of M:",
    round(matrix_norm, 4)
)

print(
    "4. Dominant eigenvalue:",
    dominant_eigenvalue
)

print(
    "5. Dominant eigenvalue magnitude:",
    abs(dominant_eigenvalue)
)

print("\nFiles generated:")

print("- future_forecast.png")
print("- convergence_analysis.png")

print("\nPerson 4 analysis completed.")
