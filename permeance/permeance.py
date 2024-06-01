import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

# Import .csv data 'outputs.csv'
plot_data = pd.read_csv('outputs.csv')

# Convert data frames into usable nd arrays
pressure = plot_data['Pressure (bar)'].to_numpy()
flux = [plot_data[f'Flux {i} (LMH)'].to_numpy() for i in range(3)]

lin_reg = [stats.linregress(pressure, flux[i]) for i in range(3)]

intercept = np.zeros(3)
slope = np.zeros(3)

for i in range(3):
    intercept[i] = lin_reg[i].intercept
    slope[i] = lin_reg[i].slope

fig = plt.figure()
first_line, = plt.plot(pressure, flux[0], 'o', c='#88ccee', label='Flux 0')
first_trend, = plt.plot(pressure, slope[0] * pressure + intercept[0], '--', c='#88ccee')
second_line, = plt.plot(pressure, flux[1], 'v', c='#ddcc77', label='Flux 1')
second_trend, = plt.plot(pressure, slope[1] * pressure + intercept[1], '--', c='#ddcc77')
third_line, = plt.plot(pressure, flux[2], '^', c='#cc6677', label='Flux 2')
third_trend, = plt.plot(pressure, slope[2] * pressure + intercept[2], '--', c='#cc6677')

plt.xlabel('Pressure')
plt.ylabel(u'Flux (Lm\u207b\u00b2Hr\u207b\u00b9)')
plt.axis((0, max(pressure) * 1.2, 0, max([max(flux[0]), max(flux[1]), max(flux[2])]) * 1.2))
# plt.legend(loc='lower right')
plt.savefig('permeance.svg')
plt.savefig('permeance.pdf')
plt.savefig('permeance.jpg', dpi=300)
