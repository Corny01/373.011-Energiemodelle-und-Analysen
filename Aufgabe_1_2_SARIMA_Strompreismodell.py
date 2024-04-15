import pandas as pd
import statsmodels.api as sm
import itertools


# Pfad zur CSV-Datei
csv_datei_pfad_demand = r"C:\Users\corne\Documents\373.011-Energiemodelle-und-Analysen\PL_2019_Demand.csv"
csv_datei_pfad_generation = r"C:\Users\corne\Documents\373.011-Energiemodelle-und-Analysen\PL_2019_Generation.csv"
csv_datei_pfad_price = r"C:\Users\corne\Documents\373.011-Energiemodelle-und-Analysen\PL_2019_Prices.csv"


# Daten einlesen
demand = pd.read_csv(csv_datei_pfad_demand, index_col='Time', parse_dates=['Time'])
generation = pd.read_csv(csv_datei_pfad_generation, index_col='Time', parse_dates=['Time'])
price = pd.read_csv(csv_datei_pfad_price, index_col='Time', parse_dates=['Time'])

renewable_energy_sources = ['Solar', 'WindOnShore', 'WindOffShore', 'Hydro', 'HydroStorage', 'HydroPumpedStorage', 'Marine', 'Geothermal', 'Biomass', 'Waste', 'OtherRenewable']
generation['RenewablesTotal'] = generation[renewable_energy_sources].sum(axis=1)

data = pd.merge(demand, generation['RenewablesTotal'], on="Time", how="inner")
data = pd.merge(data, price, on="Time", how="inner")

# Aufteilen der Daten in Trainings- und Testdaten
train_size = int(len(data) * 0.8)  # 80% der Daten für das Training verwenden
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Start- und Endzeitpunkte für Trainings- und Testdaten festlegen
start_time_train = data.index[0]
end_time_train = data.index[train_size - 1]
start_time_test = data.index[train_size]
end_time_test = data.index[-1]

# Werte für p, d, q, P, D, Q, s definieren
p = d = q = range(0, 3) # Beispielwerte
P = D = Q = range(0, 2) # Beispielwerte
s = 24  # Beispielwert für saisonale Periode (z. B. Stunden pro Tag)

# Kombinationen der Parameter erstellen
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], s) for x in list(itertools.product(P, D, Q))]

## Unabhängige Variablen für Trainings- und Testdaten auswählen
train_independent_vars = train_data[['Demand', 'RenewablesTotal']]
test_independent_vars = test_data[['Demand', 'RenewablesTotal']]

# Abhängige Variable für Trainings- und Testdaten auswählen
train_dependent_var = train_data['Price']
test_dependent_var = test_data['Price']

order = (1, 1, 1)  # Beispielwerte für ARIMA-Ordnung (p, d, q)
seasonal_order = (0, 1, 1, 12)  # Beispielwerte für saisonale ARIMA-Ordnung (P, D, Q, s)

"""# Modellparameter auswählen
best_aic = float('inf')
best_params = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.SARIMAX(train_dependent_var, exog=train_independent_vars, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
            results = mod.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_params = (param, param_seasonal)
        except:
            continue

# Beste Parameter ausgeben
print("Beste AIC: ", best_aic)
print("Beste Parameter: ", best_params)
"""
# SARIMA-Modell erstellen und anpassen mit Trainingsdaten
sarima_model = sm.tsa.SARIMAX(train_dependent_var, exog=train_independent_vars, order=(1, 1, 1), seasonal_order=(0, 1, 1, 24))
sarima_result = sarima_model.fit()

# Vorhersagen mit dem trainierten Modell für Testdaten
forecast = sarima_result.predict(start=test_data.index[0], end=test_data.index[-1], exog=test_independent_vars, dynamic=True)

# Ausgabe der Vorhersagen
print(forecast)