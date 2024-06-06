import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

plot_data = pd.read_csv('flux_decline_data.csv')


time = plot_data['Time (mins)'].to_numpy()
average_flux = plot_data['Average Flux (LMH)'].to_numpy()


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

fig = plt.figure()
first_line, = plt.plot(time, residuals, 'o', c='b', label='Flux')

plt.xlabel('Time (mins)')
plt.ylabel(u'Residual')
# plt.axis((0, max(time), 0, max(average_flux)*1.2))
plt.legend(loc='lower right')
plt.savefig('results/flux_decline.svg')
plt.savefig('results/flux_decline.pdf')
plt.savefig('results/residual.jpg', dpi=300)

plt.close()

parameter_values = np.concatenate((popt, [r_squared]))

fig = plt.figure()
first_line, = plt.plot(time, average_flux, c='b', label='Flux')
second_line, = plt.plot(time, func(time, *popt), '--', c='b', label='Model')

plt.xlabel('Time (mins)')
plt.ylabel(u'Flux (Lm\u207b\u00b2Hr\u207b\u00b9)')
plt.axis((0, max(time), 0, max(average_flux)*1.2))
plt.legend(loc='lower right')
# plt.savefig('results/flux_decline.svg')
# plt.savefig('results/flux_decline.pdf')
plt.savefig('results/flux_decline.jpg', dpi=300)

plt.close()

x = np.linspace(0, 1000, 1001)
y = func(x, *popt[0:3])

figure = plt.figure()
plt.plot(x, y)
plt.xlabel('Time (mins)')
plt.ylabel(u'Flux (Lm\u207b\u00b2Hr\u207b\u00b9)')
plt.axis((0, max(x), 0, max(y)*1.2))

plt.savefig('results/exp_model.jpg', dpi=300)


model_paremeters = ['a', 'b', 'c', 'R2']

exponential_model = {
    'Model Parameters': model_paremeters, u'Parameter Values': parameter_values
}

df_outputs = pd.DataFrame(exponential_model)
df_outputs.to_csv("results/exp_model_parameters.csv", index=False)
