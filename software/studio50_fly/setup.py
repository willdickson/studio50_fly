from setuptools import setup, find_packages

setup(
    name='studio50_fly',
    version='0.1',
    description = 'control and data acquistion software for the studio50_fly walking arena',
    author='Will Dickson',
    author_email='wbd@caltech',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    packages=find_packages(exclude=['examples',]),
    entry_points = {
        'console_scripts' : ['studio50 = studio50_fly.command_line:cli'],
        },
)
