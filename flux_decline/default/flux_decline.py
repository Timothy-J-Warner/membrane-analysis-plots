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


parameter_bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
initial_guess = [1500, 0.01, 2500]
maxfev = 800

model_param = np.zeros([3, 3])
model_r_squared = np.zeros(3)

for i in range(3):
    popt, pcov = curve_fit(func, time, flux[i], p0=initial_guess, bounds=parameter_bounds, maxfev=maxfev)
    residuals = flux[i] - func(time, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((flux[i]-np.mean(flux[i]))**2)
    r_squared = 1 - (ss_res / ss_tot)
    model_param[i] = popt
    model_r_squared[i] = r_squared

model_r_squared = model_r_squared[:, np.newaxis]
model_param = np.concatenate((model_param, model_r_squared), axis=1)

fig = plt.figure()
first_line, = plt.plot(time, flux[0], c='#88ccee', label='Flux 0')
second_line, = plt.plot(time, flux[1], c='#ddcc77', label='Flux 1')
third_line, = plt.plot(time, flux[2], c='#cc6677', label='Flux 2')
plt.xlabel('Time (mins)')
plt.ylabel(u'Flux (Lm\u207b\u00b2Hr\u207b\u00b9)')
plt.axis((0, max(time), 0, max([max(flux[0]), max(flux[1]), max(flux[2])])*1.2))
plt.legend(loc='lower right')
plt.savefig('results/flux_decline.svg')
plt.savefig('results/flux_decline.pdf')
plt.savefig('results/flux_decline.jpg', dpi=300)

output_description = ['a', 'b', 'c', 'R2']

output_data = {
    'Model Parameters': output_description, u'Specimen 1': model_param[0],
    u'Specimen 2': model_param[1], u'Specimen 3': model_param[2]
}

df_outputs = pd.DataFrame(output_data)
df_outputs.to_csv("results/exp_model_parameters.csv", index=False)
