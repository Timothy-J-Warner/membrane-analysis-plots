import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

t0 = pd.read_csv('0_mins.csv')


time_0 = t0['Time (mins)'].to_numpy()
flux_0 = plot_data['Average Flux (LMH)'].to_numpy()
flux_0_std = plot_data['Standard Deviation (LMH)'].to_numpy()
max_flux = average_flux + flux_std
min_flux = average_flux - flux_std


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


parameter_bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
initial_guess = [1500, 0.01, 2500]
maxfev = 800


popt, pcov = curve_fit(func, time, average_flux, p0=initial_guess, bounds=parameter_bounds, maxfev=maxfev)
residuals = average_flux - func(time, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((average_flux-np.mean(average_flux))**2)
r_squared = 1 - (ss_res / ss_tot)

parameter_values = np.concatenate((popt, [r_squared]))

fig = plt.figure()
average_line, = plt.plot(time, average_flux, c='k', label='Flux')
max_line = plt.plot(time, max_flux, ':', c='k', label='Max and Min Flux')
min_line = plt.plot(time, min_flux, ':', c='k')
model_line, = plt.plot(time, func(time, *popt), '--', c='r', label='Model')

plt.xlabel('Time (mins)')
plt.ylabel(u'Flux (Lm\u207b\u00b2Hr\u207b\u00b9)')
plt.axis((0, max(time), 0, max(average_flux)*1.2))
plt.legend(loc='lower right')
# plt.savefig('results/flux_decline.svg')
# plt.savefig('results/flux_decline.pdf')
plt.savefig('results/flux_decline.jpg', dpi=300)

plt.close()

model_paremeters = ['a', 'b', 'c', 'R2']

exponential_model = {
    'Model Parameters': model_paremeters, u'Parameter Values': parameter_values
}

df_outputs = pd.DataFrame(exponential_model)
df_outputs.to_csv("results/exp_model_parameters.csv", index=False)
