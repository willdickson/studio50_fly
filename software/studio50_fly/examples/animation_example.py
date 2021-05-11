import sys
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class AnimationExample:

    def __init__(self, filename):
        self.data = h5py.File(filename,'r')
        self.extract_data()

        self.ind = 0
        self.update_interval = 1 # ms
        self.line_len = 20
        self.step = 1

        self.fig, self.ax = plt.subplots()
        plt.xlabel('x (pix)')
        plt.ylabel('y (pix)')
        plt.grid(True)
        plt.axis('equal')

        self.x_fly_line = []
        self.y_fly_line = []
        self.fly_line, = plt.plot(self.x_fly_line, self.y_fly_line, 'b')
        self.body_line, = plt.plot([], [], 'g', linewidth=2) 
        self.arena_line, = plt.plot(self.x_arena, self.y_arena, 'r')




    def extract_data(self):
        attr = json.loads(self.data.attrs['jsonparam'])

        cal_data = attr['cal_data']
        arena_contour = np.array(cal_data['position']['contour'])
        pos = np.array(self.data['position'])

        trial_param = attr['param']
        self.schedule = trial_param['schedule']

        self.t_total = np.array(self.data['t_total']) 
        self.x_fly = pos[:,0]
        self.y_fly = pos[:,1]
        self.x_arena = arena_contour[:,0,0]
        self.y_arena = arena_contour[:,0,1]
        self.body_vector = np.array(self.data['body_vector'])
        self.num_pts = self.x_fly.shape[0]
        self.trial_num = np.array(self.data['trial_num'])

    def update(self, frame):
        if self.ind < self.num_pts:
            x = self.x_fly[self.ind]
            y = self.y_fly[self.ind]
            self.x_fly_line.append(x)
            self.y_fly_line.append(y)
            self.fly_line.set_data(self.x_fly_line, self.y_fly_line)

            vec = self.body_vector[self.ind]
            x_body_line = [x - 0.5*self.line_len*vec[0], x + 0.5*self.line_len*vec[0]]
            y_body_line = [y - 0.5*self.line_len*vec[1], y + 0.5*self.line_len*vec[1]]
            self.body_line.set_data(x_body_line, y_body_line)

            trial_num = self.trial_num[self.ind]
            plt.title(f'{self.schedule[trial_num]}, ind: {self.ind}')
            self.ind += self.step 
            
        return (self.fly_line, self.body_line)

    def run(self):
        ani = FuncAnimation(self.fig, self.update, interval=self.update_interval)
        plt.show()

# -----------------------------------------------------------------------------
if __name__ == '__main__':



    filename = sys.argv[1]
    example = AnimationExample(filename)
    example.run()







