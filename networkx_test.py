# import networkx as nx
# import matplotlib.pyplot as plt
# G = nx.triangular_lattice_graph(m=10,n=10, periodic=False, with_positions=True, 
#                                create_using=None)
# pos = nx.get_node_attributes(G, 'pos')
# nx.draw(G, pos=pos, with_labels=True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from  mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear
from mpl_toolkits.axisartist import Subplot

def curvelinear_test1(fig):
    def tr(x, y):
        x, y = np.asarray(x), np.asarray(y)
        return .8*x+.2*y, .2*x + .8*y
    def inv_tr(x,y):
        x, y = np.asarray(x), np.asarray(y)
        return 1.333*x + -.333*y, -.333*x + 1.333*y

    grid_helper = GridHelperCurveLinear((tr, inv_tr))
    ax1 = Subplot(fig, 1, 1, 1, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    xx, yy = tr([3, 6], [5.0, 10.])
    ax1.plot(xx, yy)

    ax1.set_aspect(1.)
    ax1.set_xlim(-10, 10.)
    ax1.set_ylim(-10, 10.)

    ax1.axis["t"]=ax1.new_floating_axis(0, 0.)
    ax1.axis["t2"]=ax1.new_floating_axis(1, 0.)
    ax1.grid(True)

fig = plt.figure()
curvelinear_test1(fig)
plt.show()