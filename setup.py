from setuptools import setup, find_packages

setup(
    name='seeclear',
    version='0.0.1',
    description='Transparent-object opacification and depth estimation.',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
