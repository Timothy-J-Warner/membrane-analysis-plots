import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

plot_data = pd.read_csv('flux_decline_data.csv')

time = plot_data['Time (mins)'].to_numpy()
flux = [plot_data[f'Flux {i} (LMH)'].to_numpy() for i in range(3)]

lin_reg = [stats.linregress(time, flux[i]) for i in range(3)]

intercept = np.zeros(3)
slope = np.zeros(3)

for i in range(3):
    intercept[i] = lin_reg[i].intercept
    slope[i] = lin_reg[i].slope

fig = plt.figure()
first_line, = plt.plot(time, flux[0], c='#88ccee', label='Flux 0')
second_line, = plt.plot(time, flux[1], c='#ddcc77', label='Flux 1')
third_line, = plt.plot(time, flux[2], c='#cc6677', label='Flux 2')
plt.xlabel('Time (mins)')
plt.ylabel('Flux (LMH)')
plt.axis((0, max(time), 0, max([max(flux[0]), max(flux[1]), max(flux[2])])*1.2))
# plt.legend(loc='lower right')
plt.savefig('flux_decline.svg')
plt.savefig('flux_decline.pdf')
plt.savefig('flux_decline.jpg', dpi=300)

output_data = {
    'Gradient (LMH/min)': slope
}

df_outputs = pd.DataFrame(output_data)
df_outputs.to_csv("linear_flux_decline_rate.csv", index=False)
