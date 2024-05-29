# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plot_data = pd.read_csv('flux_decline_data.csv')

time = plot_data['Time (mins)'].to_numpy()
flux_0 = plot_data['Flux 0 (LMH)'].to_numpy()
flux_1 = plot_data['Flux 1 (LMH)'].to_numpy()
flux_2 = plot_data['Flux 2 (LMH)'].to_numpy()

fig = plt.figure()
first_line, = plt.plot(time, flux_0, label='Flux 0')
second_line, = plt.plot(time, flux_1, label='Flux 1')
third_line, = plt.plot(time, flux_2, label='Flux 2')
plt.xlabel('Time (mins)')
plt.ylabel('Flux (LMH)')
plt.axis((0, max(time), 0, max([max(flux_0), max(flux_1), max(flux_2)])*1.2))
plt.legend(loc='lower right')
plt.savefig('flux_decline.svg')
plt.savefig('flux_decline.pdf')
plt.savefig('flux_decline.jpg', dpi=300)
