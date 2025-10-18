from setuptools import setup, find_packages

setup(
    name='atrackcs', 
    version='2.0.0', 
    description='ATRACKCS is a Python package for the detection and tracking of convective systems using image-processing techniques.',
    author='Vanessa Robledo',
    author_email='vanessa-robledodelgado@uiowa.edu',
    url='https://github.com/ATRACKCS/ATRACKCS',
    license='GPL-3.0-or-later',
    
    # === CORRECCIÓN AQUÍ: Solo se incluye esta línea ===
    # Esta es la forma correcta de encontrar e incluir tus submódulos 'atrackcs.*'
    packages=find_packages(include=['atrackcs', 'atrackcs.*']),
    
    # Dependencies required by your library
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'xarray',
        'dask',
        'geopandas',
        'shapely',
        'rioxarray',
        'rasterio',
        'matplotlib',
        'tqdm',
        'requests',
        'joblib',
        'psutil',
        'cdsapi',
        'cads-api-client',
    ],
    
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Atmospheric Science'
        ],
    
    # Optional: If you have non-code files (like data)
    include_package_data=True,
    
    # También puedes añadir:
    python_requires='>=3.8',
)