from setuptools import setup, find_packages

setup(
    name="emg_feature_extraction",
    version="0.0.1",
    author="Ethan Eddy",
    author_email="eeddy@unb.ca",
    packages=find_packages("src"),
    package_dir={"": "src"},
    setup_requires=["numpy", "scipy", "sampen"],
    tests_require=["pytest", "pytest-cov", "numpy", "scipy", "sampen"],
)