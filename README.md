# Contesting-Black-Box-Decisions

This repository contains the code and data for the paper "Contesting Black-Box Decisions"

## Data
The original data is available at [German Credit Data](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) and the preprocessing made on this data is described in the repository [counterfactual-fairness](https://github.com/alku7660/counterfactual-fairness/tree/main) corresponding to the paper [Measuring the Burden of (Un)fairness Using Counterfactuals](https://link.springer.com/chapter/10.1007/978-3-031-23618-1_27#citeas) 
The objective of the data is to predict whether a customer is a good or bad customer based on the following features:

| Name             | Data Type   | Original Values                                                                                          | Transformation                       |
|------------------|-------------|----------------------------------------------------------------------------------------------------------|--------------------------------------|
| PurposeOfLoan    | Categorical | Business, Education, Electronics, Furniture, HomeAppliances, NewCar, Other, Repairs, Retraining, UsedCar | Mapped to numerical values: 1 to 10  |
| Sex              | Binary      | Male, Female                                                                                             | Male mapped to 1, Female mapped to 2 |
| Single           | Binary      | 0 (Not Single), 1 (Single)                                                                               | No transformation                    |
| Unemployed       | Binary      | 0 (Employed), 1 (Unemployed)                                                                             | No transformation                    |
| InstallmentRate  | Numerical   | LoanRateAsPercentOfIncome                                                                                | No transformation                    |
| Housing          | Categorical | OwnsHouse, RentsHouse                                                                                    | Mapped to numerical values: 1 to 3   |
| Age              | Numerical   | Age                                                                                                      | No transformation                    |
| Credit           | Numerical   | Credit                                                                                                   | No transformation                    |
| LoanDuration     | Numerical   | LoanDuration                                                                                             | No transformation                    |
| Label            | Binary      | Good customer : -1 (Bad Customer), 1 (Good Customer)                                                     | -1 mapped to 0, 1 mapped to 1        |

The following table shows the statistics of the data:

|       | Sex         | Single     | Unemployed  | Age         | Credit       | LoanDuration | PurposeOfLoan | InstallmentRate | Housing     | Label       |
|:------|:------------|:-----------|:------------|:------------|:-------------|:-------------|:--------------|:----------------|:------------|:------------|
| count | 1000.000000 | 1000.00000 | 1000.000000 | 1000.000000 | 1000.000000  | 1000.000000  | 1000.000000   | 1000.000000     | 1000.000000 | 1000.000000 |
| mean  | 0.690000    | 0.54800    | 0.062000    | 35.546000   | 3271.258000  | 20.903000    | 4.596000      | 2.973000        | 1.395000    | 0.700000    |
| std   | 0.462725    | 0.49794    | 0.241276    | 11.375469   | 2822.736876  | 12.058814    | 2.518954      | 1.118715        | 0.674856    | 0.458487    |
| min   | 0.000000    | 0.00000    | 0.000000    | 19.000000   | 250.000000   | 4.000000     | 1.000000      | 1.000000        | 1.000000    | 0.000000    |
| 25%   | 0.000000    | 0.00000    | 0.000000    | 27.000000   | 1365.500000  | 12.000000    | 3.000000      | 2.000000        | 1.000000    | 0.000000    |
| 50%   | 1.000000    | 1.00000    | 0.000000    | 33.000000   | 2319.500000  | 18.000000    | 4.000000      | 3.000000        | 1.000000    | 1.000000    |
| 75%   | 1.000000    | 1.00000    | 0.000000    | 42.000000   | 3972.250000  | 24.000000    | 6.000000      | 4.000000        | 2.000000    | 1.000000    |
| max   | 1.000000    | 1.00000    | 1.000000    | 75.000000   | 18424.000000 | 72.000000    | 10.000000     | 4.000000        | 3.000000    | 1.000000    |

The data is divided in three subsets: train, test and validation. The subsets are available in the `data/tra_tst_val` folder along with the data corresponding to the normalisation of the features Age, Credit and LoanDuration necessary to train the black-box model. All this transformation as well as the standardisation of the features is done in the `data_transform.py` file.

## Code
The code is divided into three parts:

### 1. The data 

The folder `data` contains the data and the data transformation. The data is divided into three subsets: train, test and validation. The data is standardised before being used to train the black-box model.

### 2. The black-box model

The black-box model can be found in the `black_box` folder. The model is a simple feedforward neural network with 2 hidden layers. The model is trained on the training set and the validation gives an accuracy of 0.77. The test accuracy is 0.67.
By running the file `bb_app.py` a simple web application is launched. The user can input the features of a customer and the model will predict whether the customer is a good or bad customer. The results of the prediction is saved in the `data/results.csv` file.

### 3. The interpretable model:

The interpretable model can be found in the `interpretable_model` folder. The model is a Decision Tree trained on the results of the black-box model. The results of the black-box model are used as the target variable for the Decision Tree. Acc = 0.7.
By running the file `im_app.py` a simple web application is launched. The user can input the features of a customer and the model will predict whether the customer is a good or bad customer. The interface will also show the features path for the prediction.

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