# Applied Machine Learning and Systems - UCL 20/21

The repo contains my solution for the AMLS module.


## Warnings

- Code only tested with python 3.6.12. However, therotically it should work
with any python version greater than 3.6
- Code only tested on macOS but it's written in a cross platform manner. So,
it should work on any platform available
- Code Utilises Python Multiprocessing for computing intensive tasks. Make
sure you have permission to fork processes on the machine you are testing


## Libraries Used

- Keras and Kera.preprocessing
- OpenCV
- scikit-learn
- Pillow
- Dlib
- Tensorflow

## Prerequisites

### Using python venv module

- Install cmake and a c++ compiler:

  - On Windows this can be done by installing visual studio
  - On macOS use homebrew or macports to install cmake, clang++ is preinstalled on macs
  - On Linux use your default package manager to install cmake, and gcc/g++

- Create and Activate the python environment

- Use pip to install python dependencies as follows:
  `pip3 install -r requirements.txt`

### Using conda

- Install conda/miniconda on your system

- Create the conda environment as follows:
  `conda env create`

- Activate the newly created conda environment



## Running the Code

After Setting up your environment using the steps described above:

  `python3 main.py`
