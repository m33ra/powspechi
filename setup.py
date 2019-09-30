from setuptools import setup, find_packages

with open('README.md', 'r') as readf:
    readme = readf.read()

requirements = ['healpy>=1.12.10']

setup(
        name='powspecHI',
        version='0.2.7.3dev',
        author='Meera Machado',
        author_email='machado.meera@gmail.com',
        description='A package angular spectral analysis for heavy-ions',
        long_description=readme,
        url='https://github.com/m33ra/powspecHI/',
        packages=find_packages(),
        include_package_data=True,
        install_requires=requirements,
        classifiers=[
            "Programming Language :: Python :: 3.6",
            "License :: OSI Approved :: MIT License",
        ],
)
