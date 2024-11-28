# Detection of Large Language Model Contamination with Tabular Data

This repository contains the code and results for the paper "Detection of Large Language Model Contamination with Tabular Data" (under review).


## Files and folders

Files:

- data_loader.py: contains the code to load the data from the different files (csv and others)

- data_processing.py: contains the code to process the data, either as input or as output. The code to serialize the examples is contained in this file.

- enchant_vs_nltk.txt: simple comparison between the pyenchant and the nltk packages

- llm_functions.py: main file, contains the code to use the LLMs and the implementation of the different tests with the functions to run all of them.

- LLM_output_parser.py: contains the code for the decision algorithm described in the paper

- plot_paper.py: code to create the plots and tables from the results (in csv) for the paper

- prompts.py: templates for all the prompts used to run the experiments

- README.md: this file

- requirements.txt: list of the packages and their corresponding version

- google_key.txt: where to put your Google API key (empty by default)

- openai_key.txt: where to put your OpenAI API key (empty by default)


Folders:

- contamination_results: contains all the results in a csv format. Each file is named after the LLM it represents (gpt3.csv contains all the test results for gpt 3.5 for example). The acc_membership_results.csv file contains all the results for the classification tests. The subfolder "serializations" contains the results for the serialization comparison. Each file is named after the model it corresponds to with the serialization number being the order of the serialization templates presented in the paper.

- datasets: contains all the datasets used in this work (as csv files or other)

- plots_paper: contains all the plots displayed in the paper. Each file is named after the LLM it corresponds to.

- plots_serialization: contains the plots from the appendix of the paper for the serialization tests

- results: the raw outputs from the LLMs for the different tests. The files that contain "parsed_tests" have the results for all the contamination tests. The files that contain "seria_tests" have the results for the serialization tests. The files with "df_recent" in their names contain the experiments on the most recent dataset (Thyroid diff). The files with "incomp" contain the results for the last version of the incomplete completion test. The files with "membership" contain the results for the membership test.



## Instructions to run the code

To run the code and perform the experiments shown in the paper:

1) Go to the llm_functions.py file
2) Change the variable llm_name to use the desired LLM. If relevant, the LLM will be downloaded from HuggingFace
3) Depending on the memory available, one may want to delete the downloaded LLM. To do so, use the command line
 ```huggingface-cli delete-cache``` and select the LLM to delete.
4) Launch the code "llm_functions.py" with the command
```python llm_functions.py```
. All the tests will be run automatically with the decision algorithm and the results for the corresponding LLM will be outputted in a csv file in a folder named "contamination_results" with the name of the LLM.

Note: easy addition of new LLMs and datasets will be added later with additional parameters.


## Datasets
Details about the datasets used:

- **Adult Income**: it contains 16 features, including information such as the age, sex, occupation, or marital status of the different individuals. The main goal of this dataset is to predict if a person earns more or less than 50K a year. Concerning the number of examples, it seems that two versions are available, one with 48842 examples (from the UCI repository) and another with 32561 examples (from Kaggle). In this paper, we will mainly use the version from Kaggle. [Access the dataset](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)

- **Bank Marketing**: generally used to determine if a client from the bank applies for a term deposit. It contains 16 features including age, education, and "poutcome" (outcome from the previous marketing campaign). It has 45211 examples and can be found on the UCI repository. [Access the dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

- **California Housing**: used to predict the price of houses in California. It can be found on Kaggle and has features like ocean proximity, housing median age, or total rooms. It contains 20640 examples. [Access the dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

- **German Credit**: labelizes people with good or bad credit risk. It contains 1000 examples but two versions exist for the features: one with 20 features without specific names and a reduced one with 10 features, including age, credit amount, purpose, and duration. [Access the dataset](https://www.kaggle.com/datasets/uciml/german-credit)

- **Iris**: common dataset used in machine learning to classify three types of flowers based on four features: the sepal width and length and the petal width and length. It contains 150 examples. [Access the dataset](https://scikit-learn.org/1.5/auto_examples/datasets/plot_iris_dataset.html)

- **Wine**: or wine quality dataset has 178 examples with 13 features to classify the wines. The features include information such as Alcalinity\_of\_ash, Total\_phenols, or Hue. [Access the dataset](https://archive.ics.uci.edu/dataset/109/wine)

- **Titanic**: dataset from Kaggle whose goal is to determine which person survived the incident. It has 12 features such as PassengerID, Name, SibSp (number of siblings/spouses aboard), and Cabin. It is composed of 891 examples. [Access the dataset](https://www.kaggle.com/competitions/titanic)

- **Spaceship Titanic**: synthetic dataset from Kaggle composed of 8693 examples about the survival of passengers in a spaceship. It has 14 features such as PassengerId, HomePlanet, or Destination. The main reason for considering this dataset is that it was created in 2022, making it the most recent dataset from our list and posterior to many of the LLMs training. [Access the dataset](https://www.kaggle.com/competitions/spaceship-titanic)

- **Differentiated Thyroid Cancer Recurrence**: contains 17 features and 383 examples. It is accessible through Kaggle and the UCI repository. The goal is to predict if the cancer recurred or not. [Access the dataset](http://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence)
