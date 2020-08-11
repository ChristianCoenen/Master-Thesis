# Entropy-Propagation

## Requirements
- Python 3.8
- pipenv 2018.11.26
- graphviz (optional)

## Setup Mac
1. Download & install the latest python 3.8 release from [here](https://www.python.org/downloads/mac-osx/) 

2. Install pipenv
    ```
    brew install pipenv
    ```

3. Optional: Install graphviz
    ```
    brew install graphviz
    ```

4. Run the following command to create a virtual environment with needed dependencies installed
    ```
    pipenv install
    ```
5. Start the program ...

    5.1. ... using the command line

    ```
    pipenv run python main.py
    ```

    5.2. ... through an IDE:

    Select venv as interpreter & run main.py


## Graphviz
If you have installed Graphviz and would like to see architecture outputs,
make sure to set the graphviz flag in the config.py in the root folder
```python
GRAPHVIZ = True
```
        
        