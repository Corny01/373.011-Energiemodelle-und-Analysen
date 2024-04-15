import pandas as pd
import matplotlib.pyplot as plt

# Pfad zur CSV-Datei
csv_datei_pfad_demand = r"C:\Users\corne\Documents\373.011-Energiemodelle-und-Analysen\PL_2019_Demand.csv"
csv_datei_pfad_generation = r"C:\Users\corne\Documents\373.011-Energiemodelle-und-Analysen\PL_2019_Generation.csv"
csv_datei_pfad_price = r"C:\Users\corne\Documents\373.011-Energiemodelle-und-Analysen\PL_2019_Prices.csv"


# Daten einlesen
demand = pd.read_csv(csv_datei_pfad_demand, index_col="Time", parse_dates=True)
generation = pd.read_csv(csv_datei_pfad_generation, index_col="Time", parse_dates=True)
price = pd.read_csv(csv_datei_pfad_price, index_col="Time")

renewable_energy_sources = ['Solar', 'WindOnShore', 'WindOffShore', 'Hydro', 'HydroStorage', 'HydroPumpedStorage', 'Marine', 'Geothermal', 'Biomass', 'Waste', 'OtherRenewable']
generation['RenewablesTotal'] = generation[renewable_energy_sources].sum(axis=1)
generation['Total'] = generation.sum(axis=1)
generation['ShareRenewables'] = generation['RenewablesTotal'] / generation['Total']

# Spalten auswählen
generation_total = generation['Total']
generation_renewable = generation['RenewablesTotal']
generation_share = generation['ShareRenewables']
time_axis = pd.read_csv(csv_datei_pfad_price)['Time']
time_axis_LAG = time_axis.drop(time_axis.index[:23], inplace=False)

from Aufgabe_1_2_lineares_Strompreismodell import y_1_pred_total, y_2_pred_total
from Aufgabe_1_2_logarithmisches_Strompreismodell import y_pred_total_normal
from Aufgabe_1_2_LAG_Strompreismodell import y_pred_total

price['y_1_pred_total'] = y_1_pred_total
price['y_2_pred_total'] = y_2_pred_total
price['y_pred_total_normal'] = y_pred_total_normal
price_LAG = price.drop(price.index[:23], inplace=False)
price_LAG['y_pred_total'] = y_pred_total


diff_y_1_pred_total = price['y_1_pred_total'] - price['Price']
diff_y_2_pred_total = price['y_2_pred_total'] - price['Price']
diff_y_pred_total_normal = price['y_pred_total_normal'] - price['Price']
diff_y_pred_total = price_LAG['y_pred_total'] - price_LAG['Price']
"""
# Scatter-Plot erstellen
plt.scatter(demand['Demand'], price['Price'])
plt.xlabel('demand')
plt.ylabel('price')
plt.title('Scatter-Plot demand <-> price')
plt.scatter(demand['Demand'], price['Price'], s=1, marker='o', edgecolor='lightgreen', color='lightgreen', linewidths=0)
plt.show()

plt.scatter(generation['RenewablesTotal'], price['Price'])
plt.xlabel('generation_renewable')
plt.ylabel('price')
plt.title('Scatter-Plot generation_renewable <-> price')
plt.scatter(generation['RenewablesTotal'], price['Price'], s=1, edgecolor='none', color='cyan', linewidths=0)
plt.show()

plt.scatter(generation['ShareRenewables'], price['Price'])
plt.xlabel('generation_share')
plt.ylabel('price')
plt.title('Scatter-Plot generation_share <-> price')
plt.scatter(generation['ShareRenewables'], price['Price'], s=1, edgecolor='none', color='yellow', linewidths=0)
plt.show()

plt.scatter(generation['Total'], price['Price'])
plt.xlabel('generation_total')
plt.ylabel('price')
plt.title('Scatter-Plot generation_total <-> price')
plt.scatter(generation['Total'], price['Price'], s=1, edgecolor='none', color='black', linewidths=0)
plt.show()

plt.scatter(price['y_1_pred_total'], price['Price'])
plt.xlabel('predicted price by production of renewables')
plt.ylabel('real price')
plt.title('Scatter-Plot predected price <-> real price')
plt.scatter(price['y_1_pred_total'], price['Price'], s=1, edgecolor='none', color='black', linewidths=0.01, marker=None)
plt.show()

plt.scatter(price['y_2_pred_total'], price['Price'])
plt.xlabel('predicted price by share of renewables')
plt.ylabel('real price')
plt.title('Scatter-Plot predected price <-> real price')
plt.scatter(price['y_2_pred_total'], price['Price'], s=1, edgecolor='none', color='black', linewidths=0)
plt.show()

plt.scatter(price['y_pred_total_normal'], price['Price'])
plt.xlabel('predicted price by logarithmic regression')
plt.ylabel('real price')
plt.title('Scatter-Plot predected price <-> real price')
plt.scatter(price['y_pred_total_normal'], price['Price'], s=1, edgecolor='none', color='black', linewidths=0)
plt.show()

# Liniendiagramm erstellen
plt.figure(figsize=(150, 30))
plt.plot(time_axis, price['Price'], label='real price')
plt.plot(time_axis, price['y_1_pred_total'], label='linear regression: demand, renewable_total')
plt.plot(time_axis, diff_y_1_pred_total, label='difference')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Comparison of real price and predicted price by linear regression (demand, renewable_total)')
plt.legend()
plt.xticks(time_axis[::168])
plt.show()

plt.figure(figsize=(150, 30))
plt.plot(time_axis, price['Price'], label='real price')
plt.plot(time_axis, price['y_2_pred_total'], label='linear regression: demand, renewable_share')
plt.plot(time_axis, diff_y_2_pred_total, label='difference')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Comparison of real price and predicted price by linear regression (demand, renewable_share)')
plt.legend()
plt.xticks(time_axis[::168])
plt.show()

plt.figure(figsize=(150, 30))
plt.plot(time_axis, price['Price'], label='real price')
plt.plot(time_axis, price['y_pred_total_normal'], label='logarithmic regression')
plt.plot(time_axis, diff_y_pred_total_normal, label='difference')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Comparison of real price and predicted price by logarithmic regression (demand, renewable_total)')
plt.legend()
plt.xticks(time_axis[::168])
plt.show()
"""
plt.figure(figsize=(150, 30))
plt.plot(time_axis_LAG, price_LAG['Price'], label='real price')
plt.plot(time_axis_LAG, price_LAG['y_pred_total'], label='predected price by LAG regression')
plt.plot(time_axis_LAG, diff_y_pred_total, label='difference')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Comparison of real price and predicted price by LAG regression (demand, renewable_total)')
plt.legend()
plt.xticks(time_axis[::168])
plt.show()

