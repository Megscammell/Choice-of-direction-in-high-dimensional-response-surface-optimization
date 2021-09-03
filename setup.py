from setuptools import find_packages, setup 


exec(open("src/est_dir/version.py", "r").read())
requirements = ["numpy>=1.19.2", "scipy>=1.5.2", "pytest>=6.2.1",
                "tqdm>=4.50.2", "setuptools>=49.2.1",
                "hypothesis>=6.0.0", "matplotlib>=3.3.2",
                "pytest-cov>=2.10.1", "statsmodels>=0.12.0",
                "seaborn>=0.11.0"]

setup(
    name="est_dir",
    install_requires=requirements,
    author="Meg Scammell",
    author_email="scammellm@cardiff.ac.uk",
    license="MIT",
    keywords=["Regression", "Optimization", "Python"],
    packages=find_packages("src"),
    package_dir={"":"src"}
)


