from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

# with codecs.open(os.path.join(here,"README.md"), encoding="utf-8") as fh:
#     long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'Extract Features'
LONG_DESCRIPTION = 'A package that lets you extract features from time series data (emg based).'

setup(
    name="libemg",
    version=VERSION,
    author="Anonymous",
    description=DESCRIPTION,
    #long_description_content_type="text/markdown",
    #long_description=long_description,
    packages=find_packages(),
    install_requires=["numpy", "scipy", "scikit-learn", "pillow", "matplotlib",
    "librosa", "wfdb"],
    keywords=['emg','feature_extraction','pattern_recognition'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows"
    ]
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
