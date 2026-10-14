import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SPECIES = ["Herring", "Whiting", "Cod"]

data = pd.read_csv("cleaned_ecosystem_data.csv")
transitions = pd.read_csv("valid_transitions.csv").sort_values("From Year").reset_index(drop=True)

X = []
Y = []

for i in range(len(transitions)):
    from_year = transitions.iloc[i]["From Year"]
    to_year = transitions.iloc[i]["To Year"]

    current = data[data["Year"] == from_year][SPECIES].values[0]
    next_year = data[data["Year"] == to_year][SPECIES].values[0]

    X.append(current)
    Y.append(next_year)

X = np.array(X, dtype=float)
Y = np.array(Y, dtype=float)

to_years = transitions["To Year"].values

print("Total transitions:", len(X))

# Train/test split(80/20)
split = int(0.8 * len(X))

X_train = X[:split]
Y_train = Y[:split]

X_test = X[split:]
Y_test = Y[split:]

print("Training transitions:", len(X_train))
print("Testing transitions:", len(X_test))

# Least squares fitting
M = np.linalg.lstsq(X_train, Y_train, rcond=None)[0].T

np.set_printoptions(precision=4, suppress=True)

print("\nTransition Matrix M:")
print(M)

# Predictions for training data(to evaluate M)
Y_pred_train = X_train @ M.T

# Predictions for test data(using the M)
Y_pred = X_test @ M.T

print("\nPredicted values (test):")
print(np.round(Y_pred, 2))

print("\nActual values (test):")
print(Y_test)

# Prediction error
error = Y_test - Y_pred

print("\nPrediction Error (test):")
print(np.round(error, 2))

# Mean Squared Error
mse = np.mean(error ** 2, axis=0)

print("\nMean Squared Error:")
for i in range(len(SPECIES)):
    print(SPECIES[i] + ":", round(mse[i], 2))

# Plot

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

for i, ax in enumerate(axes):

    # Actual values
    ax.plot(
        to_years,
        np.concatenate([Y_train[:, i], Y_test[:, i]]),
        "k-o",
        ms=3,
        label="Actual"
    )

    # Predicted training values
    ax.plot(
        to_years[:split],
        Y_pred_train[:, i],
        "b--x",
        ms=4,
        label="Predicted (train)"
    )

    # Predicted test values
    ax.plot(
        to_years[split:],
        Y_pred[:, i],
        "r--x",
        ms=5,
        label="Predicted (test)"
    )

    # Train/test split line
    ax.axvline(
        to_years[split],
        color="gray",
        ls="-.",
        lw=1.5
    )

    ax.axhline(
        0,
        color="gray",
        lw=0.5
    )

    ax.set_ylabel(SPECIES[i])
    ax.grid(alpha=0.3)

axes[0].legend(
    ncol=3,
    fontsize=8,
    loc="upper left"
)

axes[-1].set_xlabel("Year")

fig.suptitle(
    "Actual vs Predicted Ecosystem Populations (train | test)"
)

fig.tight_layout()

plt.show()

#saving the files 

np.save("M_fitted.npy", M)

np.savez(
    "model_results.npz",
    M=M,
    X=X,
    Y=Y,
    X_train=X_train,
    Y_train=Y_train,
    X_test=X_test,
    Y_test=Y_test,
    Y_pred_train=Y_pred_train,
    Y_pred=Y_pred,
    years=to_years
)

print("\nFiles saved:")
print("M_fitted.npy")
print("model_results.npz")