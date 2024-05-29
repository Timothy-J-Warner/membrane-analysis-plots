import matplotlib.pyplot as plt
import pandas as pd

plot_data = pd.read_csv('permeance/outputs.csv')

pressure = plot_data['Pressure (bar)'].to_numpy()
flux_0 = plot_data['Flux 0 (LMH)'].to_numpy()
flux_1 = plot_data['Flux 1 (LMH)'].to_numpy()
flux_2 = plot_data['Flux 2 (LMH)'].to_numpy()

fig = plt.figure()
first_line, = plt.plot(pressure, flux_0, 'o', label='Flux 0')
second_line, = plt.plot(pressure, flux_1, 'o', label='Flux 1')
third_line, = plt.plot(pressure, flux_2, 'o', label='Flux 2')
plt.xlabel('Pressure')
plt.ylabel('Flux (LMH)')
plt.axis((0, max(pressure) * 1.2, 0, max([max(flux_0), max(flux_1), max(flux_2)]) * 1.2))
plt.legend(loc='lower right')
plt.savefig('permeance/permeance.svg')
plt.savefig('permeance/permeance.pdf')
plt.savefig('permeance/permeance.jpg', dpi=300)
