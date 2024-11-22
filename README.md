# Project README

## Overview

This project is a Demo to run computuer use in terminal also as web service


## Prerequisites

- Python 3.11 or higher
- `pip` (Python package installer)

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/hariombangari/computer-use-python-demo.git
    cd computer-use-python-demo
    ```

2. Activate the virtual environment:

    - On Windows:

        ```sh
        .venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```sh
        source .venv/bin/activate
        ```

3. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

## Running the Project

### Running in CLI Mode

To run the project in CLI mode, execute the following command:

```sh
export ANTHROPIC_API_KEY=<ANTHROPIC_API_KEY>
python main.py
```

### Running in Server Mode

To run the project in Server mode, execute the following command:

```sh
export ANTHROPIC_API_KEY=<ANTHROPIC_API_KEY>
python main.py --server
```

## Acknowledgments

This project is based on code from [Anthropic](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo)