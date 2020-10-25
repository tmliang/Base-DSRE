import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

model = ['CNN', 'PCNN', 'BiGRU']
marker = ['o', 'x', 'v']
color = ['red', 'cadetblue', 'sandybrown']
plt.ylim([0.6, 1.0])
plt.xlim([0.0, 0.3])
for i, name in enumerate(model):
    p = np.load('result/'+name+'/precision.npy')
    r = np.load('result/'+name+'/recall.npy')
    # p = p[r < 0.3]
    # r = r[r < 0.3]
    plt.plot(r, p, color=color[i], label=name, marker=marker[i], lw=1,  markevery=0.1, ms=5)
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.legend(loc="upper right", prop={'size': 12})
plt.grid(True)
plt.tight_layout()
plot_path = 'plot_pr.pdf'
plt.savefig(plot_path, format="pdf")
plt.show()