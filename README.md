# Entropy-Propagation
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Requirements
- Python 3.8
- Poetry 1.1
- graphviz (optional)
- manim (optional)

## Setup (tested on MacOS and Windows)
1. Download & install the latest python 3.8 release from [here](https://www.python.org/downloads/mac-osx/) 

2. Download and install poetry as explained [here](https://python-poetry.org/docs/)

3. Optional: Install graphviz
    ```
    Mac:
    brew install graphviz
   
    Windows:
    choco install graphviz
    ```

4. Run the following command in the root of the repository to create a virtual environment 
with required dependencies installed
    ```
    poetry install
    ```
   
5. Start the program ...

    5.1. ... using the command line

    ```
    poetry run python main.py
    ```

    5.2. ... through an IDE:

    Select the poetry venv as interpreter & run main.py


## Graphviz
If you have installed Graphviz and would like to see architecture outputs,
you can call the ```save_model_architecture_images()``` method

## Manim
If you want to run the animation scripts in the animations folder, manim is required.
This is not trivial, so make sure to read the manim docs [here](https://docs.manim.community/en/stable/)
to set everything up and learn how to run the manim module.

### network_scene.py
- animates the training process of the Reinforcement Learner (saved as video file)
- manim is required to run the script
- not designed to be too generic
   - might need adjustments when changing parameters

Can be invoked by running the following commands from the project root directory:
```shell script
poetry shell
manim animations/network_scene.py RLScene -p
```
All possible command line arguments can be found [here](https://manimce.readthedocs.io/en/latest/tutorials/configuration.html)

### network_scene_epn.py
- animates the training process of the Entropy Propagation Network (saved as video file)
- manim is required to run the script
- not designed to be too generic
   - might need adjustments when changing parameters

Can be invoked by running the following commands from the project root directory:
```shell script
poetry shell
manim animations/network_scene_epn.py RLScene -p
```
All possible command line arguments can be found [here](https://manimce.readthedocs.io/en/latest/tutorials/configuration.html)

## Troubleshooting
### SSL Certificate issue
Likely on MacOS. Can be fixed when running the following command to install 
necessary certificates:
```bash
/Applications/Python 3.8/Install Certificates.command
```
        
        