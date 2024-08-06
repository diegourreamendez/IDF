from setuptools import setup, find_packages

setup(
    name="idf_analysis",
    version="0.1",
    packages=find_packages(),
    description="A library for performing Intensity-Duration-Frequency (IDF) analysis on rainfall data",
    long_description="IDFAnalysis is a class that encapsulates methods for calculating annual maximum intensities, fitting statistical models, generating IDF curves, and plotting results.",
    author="Diego Urrea",
    author_email="urread@uninca.es",
    url="https://github.com/diegourreamendez/IDF",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scipy',
        'statsmodels',
        'scikit-learn',
        'fitter'
    ],
)