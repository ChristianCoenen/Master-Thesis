# Entropy-Propagation
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Requirements
- Python 3.8
- Poetry 1.1
- graphviz (optional)

## Setup (only tested on MacOS)
1. Download & install the latest python 3.8 release from [here](https://www.python.org/downloads/mac-osx/) 

2. Download and install poetry as explained [here](https://python-poetry.org/docs/)

3. Optional: Install graphviz
    ```
    brew install graphviz
    ```

4. Run the following command in the root of the repository to create a virtual environment with needed dependencies installed
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
make sure to set the graphviz_installed parameter to true when calling the EPN constructor
```python
epn = EntropyPropagationNetwork(..., graphviz_installed=True)
```

## Troubleshooting
### SSL Certificate issue
Likely on MacOS. Can be fixed when running the following command to install 
necessary certificates:
```bash
/Applications/Python 3.8/Install Certificates.command
```
        
        