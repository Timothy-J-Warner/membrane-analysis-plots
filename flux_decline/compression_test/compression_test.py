# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plot_data_30 = pd.read_csv('inputs/flux_decline_data_30.csv')
plot_data_45 = pd.read_csv('inputs/flux_decline_data_45.csv')
plot_data_80 = pd.read_csv('inputs/flux_decline_data_80.csv')

time_30 = plot_data_30['Time (mins)'].to_numpy()
flux_30 = [plot_data_30[f'Flux {i} (LMH)'].to_numpy() for i in range(3)]
time_45 = plot_data_45['Time (mins)'].to_numpy()
flux_45 = [plot_data_45[f'Flux {i} (LMH)'].to_numpy() for i in range(3)]
time_80 = plot_data_80['Time (mins)'].to_numpy()
flux_80 = [plot_data_80[f'Flux {i} (LMH)'].to_numpy() for i in range(3)]

fig = plt.figure()
first_line_30, = plt.plot(time_30, flux_30[0], '-', c='#ddcc77', label='30 PSI')
second_line_30, = plt.plot(time_30, flux_30[1], '--', c='#ddcc77')
third_line_30, = plt.plot(time_30, flux_30[2], '-.', c='#ddcc77')

first_line_45, = plt.plot(time_45, flux_45[0], '-', c='#88ccee', label='45 PSI')
second_line_45, = plt.plot(time_45, flux_45[1], '--', c='#88ccee')
third_line_45, = plt.plot(time_45, flux_45[2], '-.', c='#88ccee')

first_line_80, = plt.plot(time_80, flux_80[0], '-', c='#cc6677', label='80 PSI')
second_line_80, = plt.plot(time_80, flux_80[1], '--', c='#cc6677')
third_line_80, = plt.plot(time_80, flux_80[2], '-.', c='#cc6677')


plt.xlabel('Time (mins)')
plt.ylabel(u'Flux (Lm\u207b\u00b2Hr\u207b\u00b9)')
plt.axis((0, max(time_45), 0, max([max(flux_80[0]), max(flux_80[1]), max(flux_80[2])])*1.2))
plt.legend(loc='lower right')
plt.savefig('results/Compression.svg')
plt.savefig('results/Compression.pdf')
plt.savefig('results/Compression.jpg', dpi=300)