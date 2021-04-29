import click 
import cv2
from .calibration import run_position_calibration
from .calibration import run_homography_calibration

@click.group()
def cli():
    pass

@click.command()
@click.argument('cal_type')
def cal(cal_type):
    cal_type = cal_type.lower()
    if cal_type == 'homography' or cal_type == 'hom':
        run_homography_calibration()
    if cal_type == 'position' or cal_type == 'pos':
        run_position_calibration()

@click.command()
def run():
    print('running trial!')

cli.add_command(cal)
cli.add_command(run)
