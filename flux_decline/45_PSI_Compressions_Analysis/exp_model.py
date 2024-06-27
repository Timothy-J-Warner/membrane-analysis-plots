import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

plot_data = pd.read_csv('results/exp_model_parameters.csv')
exp_parameters = plot_data['Parameter Value'].to_numpy()


def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c


x = np.linspace(0, 1000, 1001)
y = exp_func(x, *exp_parameters[0:3])

figure = plt.figure()
plt.plot(x, y)
plt.xlabel('Time (mins)')
plt.ylabel(u'Flux (Lm\u207b\u00b2Hr\u207b\u00b9)')
plt.axis((0, 200, 0, max(y)*1.2))

plt.savefig('results/exp_model.jpg', dpi=300)

