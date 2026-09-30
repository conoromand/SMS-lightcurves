from setuptools import setup, find_packages

setup(
    name='SMS_lightcurves',
    version='1.0.0',
    description='Explosions from supermassive stars',
    author='Cédric Jockel, Kyohei Kawaguchi, Sho Fujibayashi, and Masaru Shibata (SMS-lightcurves code); Conor Omand (Redback plugin)',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['redback>=1.12.0', 'numpy', 'scipy', 'matplotlib'],
    entry_points={
        'redback.model.modules': [
            'smssn_models = SMS_lightcurves.models',
        ],
        'redback.model.priors': [
            'smssn_priors = SMS_lightcurves.prior_provider:get_prior',
        ],
    },
)