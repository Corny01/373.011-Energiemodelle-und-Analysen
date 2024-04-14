import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
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

# explaining_variables = pd.merge(demand, generation['RenewablesTotal'], on="Time", how="inner")


# Aufteilung der Daten in Trainings- und Testsets
# X = explaining_variables    # erklärende Variablen
y = price["Price"]  # Zielvariable
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logarithmieren der Daten
log_X1 = np.log(demand)
log_X2 = np.log(generation)
log_y = np.log(y)

# Anpassen des Potenzmodells durch lineare Regression auf den logarithmierten Daten
A = np.column_stack([np.ones_like(log_X1), log_X1, log_X2])
beta, residuals, _, _ = np.linalg.lstsq(A, log_y, rcond=None)

# Extrahieren der geschätzten Koeffizienten aus beta
K = np.exp(beta[0])
beta1 = beta[1]
beta2 = beta[2]

# Ausgabe der geschätzten Koeffizienten
print("Geschätzte Koeffizienten:")
print(f"K: {K}")
print(f"beta1: {beta1}")
print(f"beta2: {beta2}")