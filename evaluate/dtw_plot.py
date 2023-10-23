import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from dtgw.dtgw_ import dtw_wrapper, sakoe_chiba_band_wrapper

n = 10
idx = np.linspace(0,6.28,num=n)

a = np.sin(idx) + np.random.uniform(size=n)/10.0
b = np.cos(idx)

window = 5
region = sakoe_chiba_band_wrapper(n, n, window)

cost = cdist(a[:,np.newaxis], b[:,np.newaxis])

cost, path = dtw_wrapper(cost, region)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(5,5), tight_layout=True)

# ax3.set_box_aspect(1)
ax3.set_visible(False)
# ax1.plot(b, np.arange(n), "x-")
# ax4.plot(np.arange(n), a, "kx-")
ax1.plot(b, np.arange(n))
ax4.plot(np.arange(n), a, "k")
offset = np.ones(len(path))*0.5
ax2.scatter(path[:,0]+offset, path[:,1]+offset, color="gray", marker="s", s=220)
ax2.set_ylim(bottom=0,top=n)
ax2.set_xlim(left=0,right=n)
ax2.plot(np.arange(0,n+1,1), np.arange(0,n+1,1)+window, "k")
ax2.plot(np.arange(0,n+1,1), np.arange(0,n+1,1)-window, "k")

ax2.set_xticks([i for i in range(n)])
ax2.set_xticklabels(["" for i in range(n)])
ax2.set_yticks([i for i in range(n)])
ax2.set_yticklabels(["" for i in range(n)])
ax2.grid(visible=True)
plt.subplots_adjust(wspace=0, hspace=0)

ax2.set_box_aspect(1)
ax1.set_axis_off()
ax4.set_axis_off()
ax1.set_box_aspect(3)
ax4.set_box_aspect(1/3)

for tick in ax2.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
for tick in ax2.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)



fig.tight_layout()

fig.savefig("../figures/dtw.pdf")

# axes = a.figure.axes


# axes[0].set_ylabel("")
# axes[0].set_xticks([])
# axes[0].set_xticklabels([])
# axes[0].set_yticks([])
# axes[0].set_yticklabels([])
# axes[0].set_frame_on(False)
# axes[0].set_box_aspect(3)

# axes[1].set_box_aspect(1)
# axes[1].set_frame_on(False)
# axes[1].set_xticks([i for i in range(n)])
# axes[1].set_xticklabels(["" for i in range(n)])
# axes[1].grid(visible=True)
# for tick in axes[1].xaxis.get_major_ticks():
#     tick.tick1line.set_visible(False)
#     tick.tick2line.set_visible(False)
#     tick.label1.set_visible(False)
#     tick.label2.set_visible(False)
# axes[1].set_yticks([i for i in range(n)])
# axes[1].set_yticklabels(["" for i in range(n)])
# for tick in axes[1].yaxis.get_major_ticks():
#     tick.tick1line.set_visible(False)
#     tick.tick2line.set_visible(False)
#     tick.label1.set_visible(False)


# axes[1].plot.set_linewidth(4)
# eps = 0.1
# axes[1].set_xlim(left=-eps, right=n-1+eps)
# axes[1].set_ylim(bottom=-eps, top=n-1+eps)

# axes[2].set_box_aspect(0.334)
# axes[2].set_xlabel("")
# axes[2].set_xticks([])
# axes[2].set_xticklabels([])
# axes[2].set_yticks([])
# axes[2].set_yticklabels([])
# axes[2].set_frame_on(False)
# axes[2].

from dtw import *

asdf=dtw(a, b, keep_internals=True).plot(type="twoway",offset=-2,linewidth=2)
axes = asdf.figure.axes
axes[0].set_axis_off()
asdf.figure.figsize=(5,5)
asdf.figure.savefig("../figures/dtw_align.pdf")