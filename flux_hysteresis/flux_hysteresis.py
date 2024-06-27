import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

# Import .csv data 'outputs.csv'
plot_data = pd.read_csv('outputs.csv')

# Convert data frames into usable nd arrays
pressure = plot_data['Pressure (bar)'].to_numpy()
flux = [plot_data[f'Flux {i} (LMH)'].to_numpy() for i in range(3)]


fig = plt.figure()
first_line, = plt.plot(pressure, flux[0], '-', marker='s', c='#88ccee', label='Flux 0')
second_line, = plt.plot(pressure, flux[1], '-', marker='^', c='#ddcc77', label='Flux 1')
third_line, = plt.plot(pressure, flux[2], '-', marker='v', c='#cc6677', label='Flux 2')

plt.xlabel('Pressure')
plt.ylabel(u'Flux (Lm\u207b\u00b2Hr\u207b\u00b9)')
plt.axis((0, max(pressure) * 1.2, 0, max([max(flux[0]), max(flux[1]), max(flux[2])]) * 1.2))
# plt.legend(loc='lower right')
plt.savefig('results/permeance.svg')
plt.savefig('results/permeance.pdf')
plt.savefig('results/permeance.jpg', dpi=300)


