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
    name="emg_feature_extraction",
    version=VERSION,
    author="Ethan & Evan",
    description=DESCRIPTION,
    #long_description_content_type="text/markdown",
    #long_description=long_description,
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["numpy", "scipy", "sampen"],
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

# setup(
#     name="emg_feature_extraction",
#     version="0.0.1",
#     author="Ethan Eddy",
#     author_email="eeddy@unb.ca",
#     packages=find_packages("src"),
#     package_dir={"": "src"},
#     setup_requires=["numpy", "scipy", "sampen"],
#     tests_require=["pytest", "pytest-cov", "numpy", "scipy", "sampen"],
# )