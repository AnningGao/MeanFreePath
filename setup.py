import setuptools

setuptools.setup(
    name='mfp',
    version='0.1',
    author='Anning Gao, J. Xavier Prochaska',
    author_email='anninggao211@gmail.com, jxp@ucsc.edu',
    description='Calculate the mean free path of ionizing photons in the IGM using stacked quasar spectra.',
    url='https://github.com/AnningGao/MeanFreePath',
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib', 'astropy', 'emcee', 'extinction', 'tqdm', 'linetools']
)
