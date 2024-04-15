import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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
data = pd.merge(price, demand, on="Time", how="inner")
data = pd.merge(data, generation['RenewablesTotal'], on="Time", how="inner")

"""ts = data['Price']

# Plotten der Autokorrelationsfunktion (ACF)
plot_acf(ts, lags=1000)
plt.title('Autokorrelationsfunktion (ACF)')
plt.show()

# Plotten der partiellen Autokorrelationsfunktion (PACF)
plot_pacf(ts, lags=1000)
plt.title('Partielle Autokorrelationsfunktion (PACF)')
plt.show()"""

# Erstellen einer Verzögerungsmatrix (LAG-Matrix) für die unabhängigen Variablen
lags = 24  # Anzahl der Verzögerungen
lagged_data = pd.concat([data['Demand'].shift(i) for i in range(lags)] +
                      [data['RenewablesTotal'].shift(i) for i in range(lags)], axis=1)
lagged_data.columns = [f'Demand_lag{i+1}' for i in range(lags)] + [f'RenewablesTotal_lag{i+1}' for i in range(lags)]

# Kombinieren der verzögerten Variablen mit der abhängigen Variable und Entfernen von NaN-Werten
lagged_data['Price'] = data['Price']
lagged_data = lagged_data.dropna()

# Aufteilen der Daten in Trainings- und Testsets
train_size = int(0.8 * len(lagged_data))
train, test = lagged_data[:train_size], lagged_data[train_size:]

# Modellbildung mit VAR
model = VAR(train)
lag_order = 4300  # Ordnung des LAG-Modells
results = model.fit(lag_order)

# Prognose mit dem Modell
forecast = results.forecast(train.values[-lag_order:], steps=len(test))

# Berechnen der Vorhersagen für das Testset
y_pred = forecast[:, -1]  # Annahme: Die abhängige Variable ist die erste Spalte
print(y_pred)
print(train)
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
"""
# t-Statistiken
t_values = results.tvalues['Price']  # Annahme: Die abhängige Variable ist 'price'
print("\nt-Statistiken:")
print(t_values)
"""
forecast_total = results.forecast(lagged_data.values[-lag_order:], steps=len(lagged_data))

# Berechnen der Vorhersagen für das Testset
y_pred_total = forecast_total[:, -1]  # Annahme: Die abhängige Variable ist die erste Spalte
