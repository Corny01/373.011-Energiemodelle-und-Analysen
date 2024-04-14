import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
df = pd.merge(demand, generation['RenewablesTotal'], on="Time", how="inner")
df = pd.merge(df, price, on="Time", how="inner")
print(df.head())

# Erstellen einer Verzögerungsmatrix (LAG-Matrix) für die unabhängigen Variablen
lags = 3  # Anzahl der Verzögerungen
lagged_df = pd.concat([df['Demand'].shift(i) for i in range(lags)] +
                      [df['RenewablesTotal'].shift(i) for i in range(lags)], axis=1)
lagged_df.columns = [f'Demand_lag{i+1}' for i in range(lags)] + [f'RenewablesTotal_lag{i+1}' for i in range(lags)]

# Kombinieren der verzögerten Variablen mit der abhängigen Variable und Entfernen von NaN-Werten
lagged_df['Price'] = df['Price']
lagged_df = lagged_df.dropna()

# Aufteilen der Daten in Trainings- und Testsets
train_size = int(0.8 * len(lagged_df))
train, test = lagged_df[:train_size], lagged_df[train_size:]

# Modellbildung mit VAR
model = VAR(train)
lag_order = 2  # Ordnung des LAG-Modells
results = model.fit(lag_order)

# Prognose mit dem Modell
forecast = results.forecast(train.values[-lag_order:], steps=len(test))


# Modellbildung mit VAR
model = VAR(train)
lag_order = 2  # Annahme: LAG-Modell mit Ordnung 2
results = model.fit(lag_order)

# Prognose mit dem Modell
forecast = results.forecast(train.values[-lag_order:], steps=len(test))

# Berechnen der Vorhersagen für das Testset
y_pred = forecast[:, 0]  # Annahme: Die abhängige Variable ist die erste Spalte

# Mean Squared Error (MSE)
mse = mean_squared_error(test['Price'], y_pred)
print("Mean Squared Error (MSE):")
print(mse)

# Adjustiertes Bestimmtheitsmaß (R^2_adj)
r_squared_adj = r2_score(test['Price'], y_pred)
print("\nAdjusted R-squared:")
print(r_squared_adj)

# Geschätzte Koeffizienten
coefficients = results.params['Price']  # Annahme: Die abhängige Variable ist 'price'
print("\nGeschätzte Koeffizienten:")
print(coefficients)

# t-Statistiken
t_values = results.tvalues['Price']  # Annahme: Die abhängige Variable ist 'price'
print("\nt-Statistiken:")
print(t_values)