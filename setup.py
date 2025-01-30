from setuptools import setup, find_packages

setup(
    name='Petrophysic',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'openpyxl',
        'pandas',
        'numpy',
        'statsmodels',
        'scipy',
        'plotly',
        'matplotlib',
        'seaborn',
        'scikit-learn',
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
