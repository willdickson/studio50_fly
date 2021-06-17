import ezdxf
import numpy as np
import matplotlib.pyplot as plt

def inch_to_mm(val):
    return 25.4*val

def mm_to_inch(val):
    return val/25.4

def get_sawtooth_point_list(param):

    inner_radius = param['arena_radius'] - 0.5*param['sawtooth_depth']
    outer_radius = param['arena_radius'] + 0.5*param['sawtooth_depth']

    # Compute angle between points on inner radius
    angle_rad = np.arccos(1.0 - (param['sawtooth_width']**2)/(2.0*inner_radius**2))
    angle_deg = np.rad2deg(angle_rad)

    # Modify slightly so that it evenly divides cicle
    num_angle = int(360/angle_deg)
    angle_deg = 360/num_angle
    angle_rad = np.rad2deg(angle_deg)

    # Create list of points for sawtool (on both inner and outer radii)
    angle_array = np.linspace(0,2.0*np.pi, 2*num_angle+1, endpoint=True)
    pt_list = []
    for i, angle in enumerate(angle_array):
        if i%2 == 0:
            radius = inner_radius
        else:
            radius = outer_radius
        x = radius*np.cos(angle)
        y = radius*np.sin(angle)
        pt_list.append((x,y))
    return pt_list


def create_sawtooth_arena(filename, param, display=False):
    doc = ezdxf.new('R2010')  
    doc.units = ezdxf.units.IN 
    msp = doc.modelspace()  
    doc.layers.new(name='sawtooth', dxfattribs={'linetype': 'SOLID', 'color': 7})
    pt_list = get_sawtooth_point_list(param)
    if display:
        x_list = [x for (x,y) in pt_list]
        y_list = [y for (x,y) in pt_list]
        plt.plot(x_list, y_list)
        plt.axis('equal')
        plt.show()
    for i in range(len(pt_list)-1):
        msp.add_line(pt_list[i], pt_list[i+1], dxfattribs={'layer': 'sawtooth'})  
    doc.saveas(filename)


def create_sawtooth_test_array(filename, param_list, margin=0.5, display=False):
    doc = ezdxf.new('R2010')  
    doc.units = ezdxf.units.IN 
    msp = doc.modelspace()  
    doc.layers.new(name='sawtooth', dxfattribs={'linetype': 'SOLID', 'color': 7})
    offset = 0
    offset_list = [offset]
    for i in range(1,len(param_list)):
        offset += param_list[i-1]['arena_radius'] + param_list[i]['arena_radius'] + margin
        offset_list.append(offset)
    for param, offset in zip(param_list, offset_list):
        pt_list = get_sawtooth_point_list(param)
        pt_list = [(x+offset, y) for (x,y) in pt_list]
        if display:
            x_list = [x for (x,y) in pt_list]
            y_list = [y for (x,y) in pt_list]
            plt.plot(x_list, y_list)
            plt.axis('equal')
        for j in range(len(pt_list)-1):
            msp.add_line(pt_list[j], pt_list[j+1], dxfattribs={'layer': 'sawtooth'})  
    if display:
        plt.show()

    doc.saveas(filename)



# -----------------------------------------------------------------------------

if 0:
    param = {
            'arena_radius'    : 0.5,
            'sawtooth_depth'  : mm_to_inch(1.0),
            'sawtooth_width'  : mm_to_inch(0.5),
            }
    
    
    create_sawtooth_arena('sawtooth.dxf', param, display=True)

if 1:
    param_list = [ 
            { 
                'arena_radius'    : 0.5,
                'sawtooth_depth'  : mm_to_inch(1.0),
                'sawtooth_width'  : mm_to_inch(0.5),
            },
            { 
                'arena_radius'    : 0.5,
                'sawtooth_depth'  : mm_to_inch(2.0),
                'sawtooth_width'  : mm_to_inch(1.0),
            },
            { 
                'arena_radius'    : 0.5,
                'sawtooth_depth'  : mm_to_inch(3.0),
                'sawtooth_width'  : mm_to_inch(1.5),
            },
            ]

    create_sawtooth_test_array('sawtooth_array.dxf', param_list,  margin=0.25, display=True)



