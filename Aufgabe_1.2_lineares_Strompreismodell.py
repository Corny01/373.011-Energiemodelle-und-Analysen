import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Pfad zur CSV-Datei
csv_datei_pfad_demand = r"C:\Users\corne\Documents\373.011-Energiemodelle-und-Analysen\PL_2019_Demand.csv"
csv_datei_pfad_generation = r"C:\Users\corne\Documents\373.011-Energiemodelle-und-Analysen\PL_2019_Generation.csv"
#csv_datei_pfad_capacity = r"C:\Users\corne\Documents\373.011-Energiemodelle-und-Analysen\PL_2019_Installed Capacity.csv"
csv_datei_pfad_price = r"C:\Users\corne\Documents\373.011-Energiemodelle-und-Analysen\PL_2019_Prices.csv"


# Daten einlesen
demand = pd.read_csv(csv_datei_pfad_demand, index_col="Time", parse_dates=True)
generation = pd.read_csv(csv_datei_pfad_generation, index_col="Time", parse_dates=True)
#capacity = pd.read_csv(csv_datei_pfad_capacity, index_col="Time", parse_dates=True)
price = pd.read_csv(csv_datei_pfad_price, index_col="Time", parse_dates=True)

renewable_energy_sources = ['Solar', 'WindOnShore', 'WindOffShore', 'Hydro', 'HydroStorage', 'HydroPumpedStorage', 'Marine', 'Geothermal', 'Biomass', 'Waste', 'OtherRenewable']
generation['RenewablesTotal'] = generation[renewable_energy_sources].sum(axis=1)

explaining_variables = pd.merge(demand, generation['RenewablesTotal'], on="Time", how="inner")


# Aufteilung der Daten in Trainings- und Testsets
X = explaining_variables # erklärende Variablen
y = price["Price"]  # Zielvariable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modellbildung (lineare Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Gleichung des Modells ausgeben
coefficients = model.coef_  # Koeffizienten des Modells
intercept = model.intercept_  # Intercept des Modells
decimal_places = 6
equation = "price = " + " + ".join([f"{coeff:.{decimal_places}f} * {['demand', 'RenewableGeneration'][i]}" for i, coeff in enumerate(coefficients)]) + f" + {intercept:.{decimal_places}f}"

# Mean squared Error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# t-Statistiken
X_with_intercept = sm.add_constant(X_train)  # Füge eine Konstante hinzu, um den Intercept zu berechnen
model_sm = sm.OLS(y_train, X_with_intercept).fit()
t_values = model_sm.tvalues[1:]  # Wir entfernen den Intercept aus den t-Werten

# Adjustiertes Bestimmtheitsmaß (R^2_adj)
y_pred = model.predict(X_train)
r_squared_adj = r2_score(y_train, y_pred)  # Berechnung des adjustierten Bestimmtheitsmaßes

# Ausgabe der Ergebnisse
print("Gleichung des Modells:")
print(equation)
print("Mean Squared Error:", mse)
result_df = pd.DataFrame({
    "Koeffizienten": coefficients,
    "t-Statistiken": t_values,
})
print("Geschätzte Koeffizienten und t-Statistiken:")
print(result_df)

print("\nAdjustiertes Bestimmtheitsmaß (R^2_adj):")
print(r_squared_adj)
