import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the dataset
data = pd.read_csv("hinkley.csv")

# First column has the species names
species = data.iloc[:, 0]

# Find the three species
herring = data[species.str.contains("Clupea harengus", case=False, na=False)].iloc[0]
whiting = data[species.str.contains("Merlangius merlangus", case=False, na=False)].iloc[0]
cod = data[species.str.contains("Gadus morhua", case=False, na=False)].iloc[0]

# Get the years
years = []
for col in data.columns[1:]:
    try:
        years.append(int(col))
    except:
        pass

# Get the population values
herring_values = []
whiting_values = []
cod_values = []

for year in years:
    col = str(year)

    herring_values.append(pd.to_numeric(herring[col], errors="coerce"))
    whiting_values.append(pd.to_numeric(whiting[col], errors="coerce"))
    cod_values.append(pd.to_numeric(cod[col], errors="coerce"))

# Put everything into one table
clean_data = pd.DataFrame({
    "Year": years,
    "Herring": herring_values,
    "Whiting": whiting_values,
    "Cod": cod_values
})

# Remove rows with missing values
clean_data = clean_data.dropna().reset_index(drop=True)

print("\nCleaned data:")
print(clean_data)

# Population data
populations = clean_data[["Herring", "Whiting", "Cod"]].values

# Create X and Y
X_list = []
Y_list = []
years_used = []

for i in range(len(clean_data) - 1):

    year1 = clean_data.iloc[i]["Year"]
    year2 = clean_data.iloc[i + 1]["Year"]

    # Only use consecutive years
    if year2 - year1 == 1:
        X_list.append(populations[i])
        Y_list.append(populations[i + 1])
        years_used.append([year1, year2])

# Convert to matrices
X = np.array(X_list).T
Y = np.array(Y_list).T

# Find rank of X
rank_X = np.linalg.matrix_rank(X)

print("\nX matrix:")
print(X)

print("\nY matrix:")
print(Y)

print("\nRank of X =", rank_X)

# Save the files
clean_data.to_csv("cleaned_ecosystem_data.csv", index=False)

pd.DataFrame(X).to_csv("X_matrix.csv", index=False)
pd.DataFrame(Y).to_csv("Y_matrix.csv", index=False)

pd.DataFrame(
    years_used,
    columns=["From Year", "To Year"]
).to_csv("valid_transitions.csv", index=False)

# Plot the populations
plt.figure(figsize=(12, 6))

plt.plot(clean_data["Year"], clean_data["Herring"],
         marker="o", label="Herring")

plt.plot(clean_data["Year"], clean_data["Whiting"],
         marker="o", label="Whiting")

plt.plot(clean_data["Year"], clean_data["Cod"],
         marker="o", label="Cod")

plt.xlabel("Year")
plt.ylabel("Annual abundance")
plt.title("Herring, Whiting and Cod Population Dynamics at Hinkley Point")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("ecosystem_population_plot.png")

plt.show()