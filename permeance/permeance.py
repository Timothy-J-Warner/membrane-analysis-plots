import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# Import .csv data 'outputs.csv'
plot_data = pd.read_csv('outputs.csv')

# Convert data frames into usable nd arrays
pressure = plot_data['Pressure (bar)'].to_numpy()
flux = [plot_data[f'Flux {i} (LMH)'].to_numpy() for i in range(3)]

lin_reg = [stats.linregress(pressure, flux[i]) for i in range(3)]

print(lin_reg)

# print(lin_reg)

# fig = plt.figure()
# first_line, = plt.plot(pressure, flux_0, 'o', label='Flux 0')
# second_line, = plt.plot(pressure, flux_1, 'o', label='Flux 1')
# third_line, = plt.plot(pressure, flux_2, 'o', label='Flux 2')
# plt.xlabel('Pressure')
# plt.ylabel('Flux (LMH)')
# plt.axis((0, max(pressure) * 1.2, 0, max([max(flux_0), max(flux_1), max(flux_2)]) * 1.2))
# plt.legend(loc='lower right')
# plt.savefig('permeance.svg')
# plt.savefig('permeance.pdf')
# plt.savefig('permeance.jpg', dpi=300)
