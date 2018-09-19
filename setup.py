import os
from distutils.core import setup

version = '0.1.0'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.md')).read()
except IOError:
    README = ''

setup(
    name='tf_trainer',
    version=version,
    author='Alexander Susha',
    author_email='isushik94@gmail.com',
    description='A TensorFlow-based model trainer with multi-GPU support',
    long_description=README,
    package_dir={ '': 'src' },
    packages=[ 'tf_trainer' ],
    keywords=[ 'tensorflow', 'gpu', 'multigpu', 'deep learning' ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
