import click 
from .calibration import run_homography_calibration

@click.group()
def main():
    pass


@click.command()
def cal():
    run_homography_calibration()



@click.command()
def run():
    print('running trial!')


main.add_command(cal)
main.add_command(run)
