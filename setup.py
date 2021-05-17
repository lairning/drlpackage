from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='lairning-decisions',
    version='0.0.5',
    url='https://github.com/lairning/drlpackage',
    license='MIT License',
    author='md_lairning',
    author_email='md.lairning@gmail.com',
    description='lairning Decisions',
    python_requires=">=3.7",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 4 - Beta"
    ],

)
