import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

plot_data = pd.read_csv('flux_decline_data.csv')

time = plot_data['Time (mins)'].to_numpy()
flux = [plot_data[f'Flux {i} (LMH)'].to_numpy() for i in range(3)]


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


parameter_bounds = (0, [4000, np.inf, 4000])
initial_guess = [1500, 0.01, 2500]
maxfev = 800


popt, pcov = curve_fit(func, time, flux[0], p0=initial_guess, bounds=parameter_bounds, maxfev=maxfev)

residuals = flux[0] - func(time, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((flux[0]-np.mean(flux[0]))**2)
r_squared = 1 - (ss_res / ss_tot)

print(popt)
print(r_squared)

fig = plt.figure()

first_line, = plt.plot(time, flux[0], 'o', c='#88ccee', label='Flux 0')
second_line, = plt.plot(time, func(time, *popt), c='#ddcc77', label='Trend')


plt.xlabel('Time (mins)')
plt.ylabel(u'Flux (Lm\u207b\u00b2Hr\u207b\u00b9)')
plt.axis((0, max(time), 0, max([max(flux[0]), max(flux[1]), max(flux[2])])*1.2))
plt.savefig('results/test.jpg', dpi=300)

plt.close()

lin_reg = [stats.linregress(time, flux[i]) for i in range(3)]

intercept = np.zeros(3)
slope = np.zeros(3)
percent_decline = np.zeros(3)
flux_output = np.zeros(3)

for i in range(3):
    intercept[i] = lin_reg[i].intercept
    slope[i] = lin_reg[i].slope
    percent_decline[i] = slope[i] / flux[i][1] * 100
    flux_output[i] = flux[i][0]

slope = np.append(slope, np.mean(slope))
percent_decline = np.append(percent_decline, np.mean(percent_decline))
flux_output = np.append(flux_output, np.mean(flux_output))

fig = plt.figure()
first_line, = plt.plot(time, flux[0], c='#88ccee', label='Flux 0')
second_line, = plt.plot(time, flux[1], c='#ddcc77', label='Flux 1')
third_line, = plt.plot(time, flux[2], c='#cc6677', label='Flux 2')
plt.xlabel('Time (mins)')
plt.ylabel(u'Flux (Lm\u207b\u00b2Hr\u207b\u00b9)')
plt.axis((0, max(time), 0, max([max(flux[0]), max(flux[1]), max(flux[2])])*1.2))
# plt.legend(loc='lower right')
plt.savefig('results/flux_decline.svg')
plt.savefig('results/flux_decline.pdf')
plt.savefig('results/flux_decline.jpg', dpi=300)

output_description = ['1', '2', '3', 'Mean']

output_data = {
    'Specimen': output_description, u'Initial Flux (Lm\u207b\u00b2Hr\u207b\u00b9/min)': flux_output, u'Gradient (Lm\u207b\u00b2Hr\u207b\u00b9/min)': slope,
    'Flux Decline (%/min)': percent_decline
}

df_outputs = pd.DataFrame(output_data)
df_outputs.to_csv("results/experiment_information.csv", index=False)
