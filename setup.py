from setuptools import setup, find_packages

with open('README.md', 'r') as readf:
    readme = readf.read()

requirements = ['healpy>=1.12.10']

setup(
        name='powspechi',
        version='0.2.9.9',
        author='Meera Vieira Machado',
        author_email='machado.meera@protonmail.com',
        description='A package of angular power spectral analysis for heavy-ions',
        long_description=readme,
        url='https://github.com/m33ra/powspechi/',
        packages=find_packages(),
        include_package_data=True,
        install_requires=requirements,
        classifiers=[
            "Programming Language :: Python :: 3.6",
            "License :: OSI Approved :: MIT License",
        ],
)
