import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

decimal_places = 6

# Pfad zur CSV-Datei
csv_datei_pfad_demand = r"C:\Users\corne\Documents\373.011-Energiemodelle-und-Analysen\PL_2019_Demand.csv"
csv_datei_pfad_generation = r"C:\Users\corne\Documents\373.011-Energiemodelle-und-Analysen\PL_2019_Generation.csv"
csv_datei_pfad_price = r"C:\Users\corne\Documents\373.011-Energiemodelle-und-Analysen\PL_2019_Prices.csv"


# Daten einlesen
demand = pd.read_csv(csv_datei_pfad_demand, index_col="Time", parse_dates=True)
generation = pd.read_csv(csv_datei_pfad_generation, index_col="Time", parse_dates=True)
price = pd.read_csv(csv_datei_pfad_price, index_col="Time", parse_dates=True)

renewable_energy_sources = ['Solar', 'WindOnShore', 'WindOffShore', 'Hydro', 'HydroStorage', 'HydroPumpedStorage', 'Marine', 'Geothermal', 'Biomass', 'Waste', 'OtherRenewable']
generation['RenewablesTotal'] = generation[renewable_energy_sources].sum(axis=1)
generation['Total'] = generation.sum(axis=1)
generation['ShareRenewables'] = generation['RenewablesTotal'] / generation['Total']


expl_vars_demand_generation = pd.merge(demand, generation['RenewablesTotal'], on="Time", how="inner")
expl_vars_demand_sharegeneration = pd.merge(demand, generation['ShareRenewables'], on="Time", how="inner")

# Aufteilung der Daten in Trainings- und Testsets
X_1 = expl_vars_demand_generation
X_2 = expl_vars_demand_sharegeneration
y = price["Price"]  # Zielvariable
X_1_train, X_1_test, y_train, y_test = train_test_split(X_1, y, test_size=0.2, random_state=42)
X_2_train, X_2_test, y_train, y_test = train_test_split(X_2, y, test_size=0.2, random_state=42)


# Modellbildung (lineare Regression)
model1 = LinearRegression()
model1.fit(X_1_train, y_train)
model2 = LinearRegression()
model2.fit(X_2_train, y_train)

# Gleichung des Modells ausgeben
coefficients_1 = model1.coef_  # Koeffizienten des Modells
intercept_1 = model1.intercept_  # Intercept des Modells
equation1 = "price = " + " + ".join([f"{coeff:.{decimal_places}f} * {['demand', 'RenewableGeneration'][i]}" for i, coeff in enumerate(coefficients_1)]) + f" + {intercept_1:.{decimal_places}f}"
coefficients_2 = model2.coef_  # Koeffizienten des Modells
intercept_2 = model2.intercept_  # Intercept des Modells
equation2 = "price = " + " + ".join([f"{coeff:.{decimal_places}f} * {['demand', 'RenewableShare'][i]}" for i, coeff in enumerate(coefficients_2)]) + f" + {intercept_2:.{decimal_places}f}"


# Mean squared Error
y_1_pred = model1.predict(X_1_test)
mse1 = mean_squared_error(y_test, y_1_pred)
y_2_pred = model2.predict(X_2_test)
mse2 = mean_squared_error(y_test, y_2_pred)

# t-Statistiken
X_1_with_intercept = sm.add_constant(X_1_train)  # Füge eine Konstante hinzu, um den Intercept zu berechnen
model1_sm = sm.OLS(y_train, X_1_with_intercept).fit()
t1_values = model1_sm.tvalues[1:]  # Wir entfernen den Intercept aus den t-Werten
X_2_with_intercept = sm.add_constant(X_2_train)  # Füge eine Konstante hinzu, um den Intercept zu berechnen
model2_sm = sm.OLS(y_train, X_2_with_intercept).fit()
t2_values = model2_sm.tvalues[1:]  # Wir entfernen den Intercept aus den t-Werten

# Adjustiertes Bestimmtheitsmaß (R^2_adj)
y_1_pred = model1.predict(X_1_train)
r1_squared_adj = r2_score(y_train, y_1_pred)  # Berechnung des adjustierten Bestimmtheitsmaßes
y_2_pred = model2.predict(X_2_train)
r2_squared_adj = r2_score(y_train, y_2_pred)  # Berechnung des adjustierten Bestimmtheitsmaßes


# Ausgabe der Ergebnisse
print("Gleichung des Modells 1:")
print(equation1)
print("Mean Squared Error 1:", mse1)
result1_df = pd.DataFrame({
    "Koeffizienten": coefficients_1,
    "t-Statistiken": t1_values,
})
print("Geschätzte Koeffizienten und t-Statistiken 1:")
print(result1_df)

print("\nAdjustiertes Bestimmtheitsmaß (R^2_adj):")
print(r1_squared_adj)

print("Gleichung des Modells 2:")
print(equation2)
print("Mean Squared Error 2:", mse2)
result2_df = pd.DataFrame({
    "Koeffizienten": coefficients_2,
    "t-Statistiken": t2_values,
})
print("Geschätzte Koeffizienten und t-Statistiken 2:")
print(result2_df)

print("\nAdjustiertes Bestimmtheitsmaß (R^2_adj):")
print(r2_squared_adj)

y_1_pred_total = model1.predict(X_1)
y_2_pred_total = model2.predict(X_2)