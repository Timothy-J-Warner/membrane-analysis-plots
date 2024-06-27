import matplotlib.pyplot as plt

fig, ax = plt.subplots()

specimen = ['Sample 1', 'Sample 2', 'Sample 5', 'Sample 6']
value = [1620, 1740, 1220, 1000]
bar_colors = ['tab:orange', 'tab:orange', 'tab:orange', 'tab:orange']

ax.bar(specimen, value, color=bar_colors, edgecolor='black')

ax.set_ylabel(u'BET area (m\u00b2/g)', fontsize=12)
ax.set_ylim(0, 2000)

plt.savefig('results/bet_results_summary.svg')
plt.savefig('results/bet_results_summary.jpg', dpi=300)

plt.close()
