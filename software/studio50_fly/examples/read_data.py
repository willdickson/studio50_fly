import sys
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1]
data = h5py.File(filename,'r')

#param = json.loads(data.attrs['jsonparam'])
#for k,v in param['config'].items():
#    print(k,v)
#print()
#
#for k,v in param['param'].items():
#    print(k,v)
#print()

t = np.array(data['t'])
dt = t[1:] - t[:-1]

plt.plot(dt)
plt.xlabel('index')
plt.ylabel('t (sec)')
plt.grid(True)
plt.show()



