from setuptools import setup, find_packages
import os
import codecs

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

# To release:
# python setup.py sdist
# python -m twine upload --repository testpypi dist/* --verbose  <------ testpypi
#

VERSION = "2.0.1"
DESCRIPTION = "LibEMG - Myoelectric Control Library"
LONG_DESCRIPTION = "A library for designing and exploring real-time and offline myoelectric control systems."

setup(
    name="libemg",
    version=VERSION,
    author="Ethan Eddy, Evan Campbell, Angkoon Phinyomark, Scott Bateman, and Erik Scheme",
    description=DESCRIPTION,
    packages=find_packages(exclude=["*tests*"]),
    long_description_content_type="text/markdown",
    long_description=long_description,
    install_requires=[
        "numpy<2.0",
        "scipy",
        "scikit-learn",
        "pillow",
        "matplotlib",
        "librosa",
        "wfdb",
        "pyserial",
        "PyWavelets",
        "requests",
        "websockets",
        "opencv-python",
        "pythonnet",
        "bleak",
        "dearpygui",
        "h5py",
        "onedrivedownloader",
        "sifi-bridge-py",
        "pygame",
        "mindrove"
    ],
    keywords=[
        "emg",
        "myoelectric_control",
        "pattern_recognition",
        "muscle-based input",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable  ",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)


# In order to push to pypi we first need to build the binaries
# navigate to the project folder in the cmd
# run:
# python setup.py sdist bdist_wheel
# you should have binaries specific to the new version specified in the setup.py file
# if you have other version binaries in the /dist folder, delete them.
# now to actually upload it to pypi, you need twine (pip install twine if you don't have it)
# now run:
# twine upload dist/*
