from setuptools import setup, find_packages

setup(
    name='Petrophysic',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'openpyxl Version: 3.1.4',
        'pandas Version: 2.0.3',
        'numpy Version: 1.25.2',
        'statsmodels Version: 0.14.2',
        'scipy Version: 1.11.4',
        'plotly Version: 5.15.0',
        'matplotlib Version: 3.7.1',
        'seaborn Version: 0.13.1',
        'scikit-learn Version: 1.2.2',
    ],
    author='Maurício Gabriel Lacerda Mafra',
    author_email='mauricio.mafra.103@ufrn.edu.br',
    description='Uma biblioteca para processamento e análise de dados de RMN',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mauriciomafra-103/Petrophysic',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
