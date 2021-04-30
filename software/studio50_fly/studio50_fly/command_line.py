import os
import click 
import datetime
from .trials import Trials 
from .calibration import run_position_calibration
from .calibration import run_homography_calibration

@click.group()
def cli():
    """ Command line interface for the studio50-flly walking arena.  """
    pass

@click.command()
@click.argument('cal_type')
def cal(cal_type):
    """ run system calibration.

    CAL_TYPE: type of calibration homography (hom) or position (pos)
    """
    cal_type = cal_type.lower()
    if cal_type in ('homography','hom'):
        run_homography_calibration()
    elif cal_type in ('position', 'pos'):
        run_position_calibration()

def default_data_file():
    now = datetime.datetime.now()
    now_str = now.strftime('%Y_%m_%d_%H_%M_%S')
    return f'data_{now_str}.hdf5'

@click.command()
@click.argument('param_file', type=click.Path(exists=True))
@click.argument('data_file', default=default_data_file,  type=click.Path())
def fly(param_file, data_file):
    """ run experimental trials. 

    PARAM_FILE: json configuration file containing the trial parameters 
    DATA_FILE:  name of file to use for logging trial data
    """
    Trials(param_file, data_file).run()

cli.add_command(cal)
cli.add_command(fly)
