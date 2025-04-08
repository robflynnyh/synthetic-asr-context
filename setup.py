from setuptools import setup, find_packages

setup(
    name='synctxasr',
    version='0.1',    
    description='Code for generating synthetic prior context for ASR model',
    url='https://github.com/robflynnyh/synthetic-asr-context',
    author='Rob Flynn',
    author_email='rjflynn2@sheffield.ac.uk',
    license='Apache 2.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['torch',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 2 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',   
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.8',
    ],
)