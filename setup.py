from setuptools import find_packages, setup


setup(
    name='storage',
    author='Ricky',
    version='0.1.0',
    packages=find_packages(include=['storage', 'storage.binance']),
    license='LICENSE.txt',
    description='Provides persistent storage for cryptocurrencies.',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.11.0",
        "pandas >= 1.0.0",
        "tables >= 3.4.3",
        "tqdm >= 4.59.0",
        "requests >= 2.26.0"
    ],
)
