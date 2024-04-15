import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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

data = pd.merge(demand, generation['RenewablesTotal'], on="Time", how="inner")
data = pd.merge(data, price, on="Time", how="inner")

data['log_Demand'] = np.log(data['Demand'])


# Aufteilen der Daten in unabhängige Variablen (X) und abhängige Variable (y)
X = data[['log_Demand', 'RenewablesTotal']]  # Unabhängige Variablen
y = data['Price']  # Abhängige Variable

# Aufteilen der Daten in Trainingsdaten und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hinzufügen eines Intercept-Terms zu den Trainingsdaten
X_train = sm.add_constant(X_train)

# Modellbildung mit den Trainingsdaten
model = sm.OLS(np.log(y_train), X_train)  # Hier log transformieren wir y_train
results = model.fit()

# Ausgabe der Ergebnisse für die Trainingsdaten
print("Trainingsdaten:")
print(results.summary())

# Vorhersage mit den Testdaten
X_test = sm.add_constant(X_test)
y_pred = np.exp(results.predict(X_test))  # Umkehr der Log-Transformation

# Ausgabe der zusätzlichen Metriken für die Testdaten
print("\nZusätzliche Metriken für die Testdaten:")
# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
# Adjustiertes Bestimmtheitsmaß (R^2_adj)
n = len(y_test)
p = len(results.params)
r_squared_adj = 1 - (1 - r2_score(y_test, y_pred)) * ((n - 1) / (n - p - 1))
print(f"Adjusted R-squared: {r_squared_adj}")

print("Modellierungsfunktion für das lineare-logarithmische Modell:")
print(f"Price = {results.params[0]:.4f} + {results.params[1]:.4f} * Demand + {results.params[2]:.4f} * log(RenewablesTotal)")