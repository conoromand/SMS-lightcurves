from setuptools import setup

setup(
    name='SMS-lightcurves',
    version='1.0.0',
    description='Explosions from supermassive stars',
    author='Cédric Jockel, Kyohei Kawaguchi, Sho Fujibayashi, and Masaru Shibata (SMS-lightcurves code); Conor Omand (Redback plugin)',
    packages=['SMS-lightcurves'],
    install_requires=['redback>=1.12.0', 'numpy', 'scipy', 'matplotlib'],
    entry_points={
        'redback.model.modules': [
            'SMS_lightcurves = SMS-lightcurves.models',
        ],
    },
)