from setuptools import find_packages, setup 


exec(open("src/est_dir/version.py", "r").read())
requirements = ["numpy>=1.18.1"]

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


