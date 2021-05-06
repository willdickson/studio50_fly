import sys
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1]
data = h5py.File(filename,'r')

attr = json.loads(data.attrs['jsonparam'])

#for k,v in attr['config'].items():
#    print(k,v)
#print()
#
#for k,v in attr['param'].items():
#    print(k,v)
#
#print()
#print()
#print()

cal_data = attr['cal_data']
arena_contour = np.array(cal_data['position']['contour'])
x_arena = arena_contour[:,0,0]
y_arena = arena_contour[:,0,1]


t_total = np.array(data['t_total'])
t_trial = np.array(data['t_trial'])
body_angle = np.array(data['body_angle'])

# Put angle in range (0,pi)
mask = body_angle < 0.0
body_angle[mask] = body_angle[mask] + np.pi

# Extract fly position
pos = np.array(data['position'])
x = pos[:,0]
y = pos[:,1]

plt.figure(1)
plt.plot(t_total, np.rad2deg(body_angle),'.-')
plt.xlabel('t (sec)')
plt.ylabel('angle (deg)')
plt.grid(True)


plt.figure(2)
plt.plot(x_arena, y_arena, 'r')
plt.plot(x,y)
plt.xlabel('x (pix)')
plt.ylabel('y (pix)')
plt.grid(True)
plt.axis('equal')
plt.show()

plt.show()



