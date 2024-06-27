import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

average_plot_data = pd.read_csv('average_compression_data.csv')
regression_data = pd.read_csv('regression_data.csv')

average_time = average_plot_data['Time (mins)'].to_numpy()
average_flux = average_plot_data[f'Flux (LMH)'].to_numpy()
flux_std = average_plot_data['Standard Deviation (LMH)']

time = regression_data['Time (mins)']
flux = regression_data['Flux (LMH)']

x = np.linspace(0, 130, 1001)


def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c


exp_parameter_bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
exp_initial_guess = [1500, 0.01, 2500]
exp_maxfev = 800

exp_popt, exp_pcov = curve_fit(exp_func, time, flux, p0=exp_initial_guess,
                               bounds=exp_parameter_bounds, maxfev=exp_maxfev)
exp_residuals = average_flux - exp_func(average_time, *exp_popt)
exp_ss_res = np.sum(exp_residuals**2)
ss_tot = np.sum((average_flux-np.mean(average_flux))**2)
exp_r_squared = 1 - (exp_ss_res / ss_tot)


# def lin_func(x, a, c):
#     return a * x + c
#
#
# lin_popt, lin_pcov = curve_fit(lin_func, time, flux)
# lin_residuals = flux - lin_func(time, *lin_popt)
# lin_ss_res = np.sum(lin_residuals**2)
# ss_tot = np.sum((flux-np.mean(flux))**2)
# lin_r_squared = 1 - (lin_ss_res / ss_tot)


fig = plt.figure()
exp_residuals, = plt.plot(average_time, exp_residuals, 's', c='k', label='Exponential Residuals')
# lin_residuals, = plt.plot(time, lin_residuals, 's', c='r', label='Linear Residuals')

plt.xlabel('Time (mins)')
plt.ylabel(u'Residual')
# plt.legend(loc='lower right')
# plt.savefig('results/flux_decline.svg')
# plt.savefig('results/flux_decline.pdf')
plt.savefig('results/residual.jpg', dpi=300)

plt.close()



fig = plt.figure()
first_line, = plt.plot(average_time, average_flux, 's', c='k', label='Flux')
second_line, = plt.plot(x, exp_func(x, *exp_popt), '--', c='r', label='Exponential Model')
# third_line, = plt.plot(time, lin_func(time, *lin_popt), ':', c='r', label='Linear Model')

plt.xlabel('Time (mins)')
plt.ylabel(u'Flux (Lm\u207b\u00b2Hr\u207b\u00b9)')
plt.axis((-1, 130, 0, max(average_flux)*1.2))
plt.legend(loc='lower right')
plt.errorbar(average_time, average_flux, yerr=flux_std, fmt='none', ecolor='black', capsize=3)
# plt.savefig('results/flux_decline.svg')
# plt.savefig('results/flux_decline.pdf')
plt.savefig('results/flux_decline.jpg', dpi=300)

exp_parameter_description = ['a', 'b', 'c', 'R2']
exp_parameter_value = np.concatenate((exp_popt, [exp_r_squared]))

exp_model = {
    'Exponential Parameter Name': exp_parameter_description, u'Parameter Value': exp_parameter_value
}

df_exp_model = pd.DataFrame(exp_model)
df_exp_model.to_csv("results/exp_model_parameters.csv", index=False)

# lin_parameter_description = ['a', 'c', 'R2']
# lin_parameter_value = np.concatenate((lin_popt, [lin_r_squared]))
#
# lin_model = {
#     'Linear Parameter Name': lin_parameter_description, u'Parameter Value': lin_parameter_value
# }
#
# df_lin_model = pd.DataFrame(lin_model)
# df_lin_model.to_csv("results/lin_model_parameters.csv", index=False)