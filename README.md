# Contesting-Black-Box-Decisions

This repository contains the code and data for the paper "Contesting Black-Box Decisions"

## Data
The data used in the paper is available in the `data/german.csv` file. The data is in the form of a CSV file with the following columns:
- `Sex`: (1.0 for male, 0.0 for female)
- `Single`: (1 for yes, 0 for no)
- `Unemployed`: (1 for yes, 0 for no)
- `Age`
- `Credit amount`: 
- `Loan Duration`: (in months)
- `Purpose Of Loan`: (numeric code)
- `Installment Rate`: (1 for low, 2 for medium, 3 for high, 4 for very high)
- `Housing`: (1 for rent, 2 for own, 3 for free)

The original data is available at https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data) and the preprocessing of the data is described in the repo: https://github.com/alku7660/counterfactual-fairness/tree/main

## Code
The code is diveded into two parts:

### 1. The black-box model

The black-box model can be found in the `black_box` folder. The model is a simple feedforward neural network with 2 hidden layers. The model is trained on the german credit data.

### 2. The interpretable model:

The interpretable model can be found in the `interpretable_model` folder. The model is a Decision Tree trained on the german credit data.


## Running the code

The code can be run using a virtual environment. If you haven't already installed virtualenv, you can do so using pip by running the following command:

``` bash
pip install virtualenv
```

The packages required to run the code are listed in the `requirements.txt` file. The following commands can be used to run the code:

``` bash
virtualenv venv

# On Windows
venv\Scripts\activate

# On Unix or MacOS
source venv/bin/activate

pip install -r requirements.txt
```

Once you're done working on your project, you can deactivate the virtual environment by running:

``` bash
deactivate
```

## Results