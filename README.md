# gg-project
 
CS337 Project 1 -- Tweet Mining & The Golden Globes 

This project uses a variety of natural language processing (NLP) and text-processing libraries for handling various text-related tasks. The following libraries are included in the project:

## Libraries Overview

1. **SpaCy**: A powerful NLP library for advanced text processing such as tokenization, lemmatization, named entity recognition, etc.
    - Model: `en_core_web_sm` (Small English model for NLP tasks).
   
2. **NLTK**: The Natural Language Toolkit (NLTK) is a comprehensive library for working with human language data (text) in Python, including text classification, tokenization, stemming, tagging, parsing, and more.

3. **JupyterLab**: A web-based interactive development environment for Jupyter notebooks, code, and data.

4. **FTFY**: Fixes text encoding issues, like repairing mojibake (encoding errors), and ensures proper display of text.

5. **Unidecode**: Converts Unicode text (e.g., accented characters) into ASCII text, useful for simplifying text handling.

6. **langdetect**: A simple and lightweight language detection library for Python, based on Google's language-detection library, that identifies the language of a given text input.

7. **Inflection**: Provides methods to singularize or pluralize English nouns, and other useful inflection tasks for English grammar.

## Setup Instructions

The project supports three installation methods: Conda with environment.yml (recommended), Conda with pip, or pip with virtual environment. Choose the method that works best for your environment. Create the environment, activate it, then download the SpaCy language model.

### Prerequisites

Make sure you have the following installed on your machine:

- **Python 3.10**: Required Python version.
- **Conda** (for Conda installation) or **Pip** (for pip installation): Package managers.

### Method 1: Conda Installation (Recommended)

1. Clone the project repository or download the project files.
2. Open a terminal in the project directory.
3. Run the following command to create a conda environment called `gg337` with `Python 3.10`. It will also install all required packages from `environment.yml`:

   ```bash
   conda env create -f environment.yml
   ```
4. Once the process completes, activate the environment using:

   ```bash
   conda activate gg337
   ```
5. After activation, download the SpaCy language model `en_core_web_sm`:

    ```bash
    python -m spacy download en_core_web_sm
    ```
6. You are ready to start using the environment for your text processing tasks.

### Method 2: Conda with Pip Installation (usually faster than Method 1)

1. Clone the project repository or download the project files.
2. Open a terminal in the project directory.
3. Create a conda environment with Python 3.10:

   ```bash
   conda create -n gg337 python=3.10
   ```
4. Activate the environment:

   ```bash
   conda activate gg337
   ```
5. Install the required packages from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```
6. Download the SpaCy language model `en_core_web_sm`:

    ```bash
    python -m spacy download en_core_web_sm
    ```
7. You are ready to start using the environment for your text processing tasks.

### Method 3: Pip with Virtual Environment

1. Clone the project repository or download the project files.
2. Open a terminal in the project directory.
3. Create a virtual environment (recommended):

   ```bash
   python -m venv gg337-env
   source gg337-env/bin/activate  # On Windows: gg337-env\Scripts\activate
   ```
4. Install the required packages from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```
5. Download the SpaCy language model `en_core_web_sm`:

    ```bash
    python -m spacy download en_core_web_sm
    ```
6. You are ready to start using the environment for your text processing tasks.

## Using the Autograder

The autograder (`autograder.py`) is provided to help you benchmark your progress and test your implementation against the ground truth data for 2013.

### Running the Autograder

1. **Basic Usage** - Test all components:
   ```bash
   python autograder.py
   ```

2. **Test Specific Components** - Test only certain functions:
   ```bash
   python autograder.py hosts awards
   python autograder.py nominees presenters winner
   ```

3. **Available Components**:
   - `hosts` - Test host identification
   - `awards` - Test award category extraction
   - `nominees` - Test nominee identification
   - `presenters` - Test presenter identification
   - `winner` - Test winner identification

### Understanding Autograder Output

The autograder provides two types of scores for each component:

- **Spelling Score**: Measures how well your results match the ground truth in terms of text similarity (handles typos, variations in formatting)
- **Completeness Score**: Measures how many correct answers you found relative to the total possible answers

Example output:
```
{'2013': {'awards': {'completeness': 0.85, 'spelling': 0.92},
          'hosts': {'completeness': 1.0, 'spelling': 1.0},
          'nominees': {'completeness': 0.78, 'spelling': 0.89},
          'presenters': {'completeness': 0.72, 'spelling': 0.85},
          'winner': {'spelling': 0.91}}}
```

### Expected File Structure

Make sure your project has the following files:
- `gg_api.py` - Your main implementation file
- `gg2013answers.json` - Ground truth data (provided)
- `autograder.py` - The autograder script
- `requirements.txt` or `environment.yml` - Dependencies

### Configuring the Analysis

The `gg_api.py` file includes two important global variables at the top of the file:

1. **`YEAR`** - The year of the Golden Globes ceremony being analyzed (e.g., "2013")
2. **`AWARD_NAMES`** - A list containing the hardcoded award categories that will be used as keys in the dictionaries returned by `get_nominees()`, `get_winner()`, and `get_presenters()` functions

**Important**: You should update both variables for your specific year. The award names list should match exactly with the keys used in your analysis and should be in lowercase format as they appear in the ground truth data.

Example of how to modify:
```python
# At the top of gg_api.py
YEAR = "2013"  # Change this to your year

AWARD_NAMES = [
    "best motion picture - drama",
    "best motion picture - comedy or musical",
    "best performance by an actor in a motion picture - drama",
    # Add or modify categories as needed for your year
    "your custom award category",
    # ... etc
]
```

### Tips for Using the Autograder

1. **Start Early**: Run the autograder frequently during development to catch issues early
2. **Test Incrementally**: Use specific component testing to focus on one area at a time
3. **Check Your Functions**: Make sure your `gg_api.py` functions return the correct data types as specified in the docstrings

### Troubleshooting

- **Import Errors**: Make sure you're in the correct environment and have installed all dependencies
- **Function Not Found**: Ensure your `gg_api.py` functions are properly defined and return the expected data types
- **Low Scores**: Review the ground truth data format and adjust your text processing accordingly

### Additional Information

- **SpaCy Model**: The `en_core_web_sm` model is used for English language processing. You can download other models as needed from the SpaCy documentation.
- **JupyterLab**: You can run Jupyter notebooks with the installed environment by simply running `jupyter lab` in the terminal after activating the environment.

### General Troubleshooting

- If the setup script fails to create the Conda environment or install dependencies, ensure that you have the correct versions of Conda and Python installed.
- For any specific library issues, consult the respective library documentation:
    - [SpaCy Documentation](https://spacy.io/usage)
    - [NLTK Documentation](https://www.nltk.org/)
    - [JupyterLab Documentation](https://jupyter.org/)
    - [FTFY Documentation](https://ftfy.readthedocs.io/en/latest/)
    - [Unidecode Documentation](https://pypi.org/project/Unidecode/)
    - [langdetect Documentation](https://pypi.org/project/langdetect/)
    - [Inflection Documentation](https://pypi.org/project/inflection/)

### License

This project is licensed under the MIT License.



