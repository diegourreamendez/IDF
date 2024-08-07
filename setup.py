from setuptools import setup, find_packages

setup(
    name="idf",
    version="0.1.0",
    packages=find_packages(),
    description="A library for performing Intensity-Duration-Frequency (IDF) analysis on rainfall data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Diego Urrea",
    author_email="urread@unincan.es",
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