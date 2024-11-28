
# FLAG CONVENTIONS:
#    - dname: the name of the dataset
#    - fname: the name of the feature(s)
#    - serialization: the serialized example(s)
#    - choices: the choices for a multiple choice question


# when giving some examples to the prompt for the values test, should give the first 5 examples (according to paper "elephants...", head is more likely to have been seen during training)

###############################################
#####          KNOWLEDGE PROMPTS          #####
###############################################


### dataset description test ###
prompt_dataset_desc = """
You are an expert in tabular data analysis. Your task is to provide a short description of the {dname} dataset.
The desciption should include: what the dataset represents, the number of examples, the number of features and who collected the data.

Dataset description:"""


### feature description test ###

# uninformed version
prompt_feature_uninf = """
You are an expert in tabular data analysis. Your task is to list all the features from the {dname} dataset.
Provide the feature names as a Python list.

Features:"""


# informed version
prompt_feature_inf = """
You are an expert in tabular data analysis. Your task is to list all the features from the {dname} dataset.
You are provided with some of the features names from the dataset. You then complete with the remaining features names.
Provide the remaining feature names as a Python list.

Subset of feature names from {dname}: {fname}
Remaining features:"""


# quiz version
prompt_feature_quiz = """
This is a five choices quiz. The task is to correctly select the option that exactly lists all the features from the {dname} dataset.

When selecting a choice, the following rules must be followed:
1. Only one choice must be selected: A, B, C, D or E.
2. If the answer is unknown, the option E must be selected.

Hint: even if the different choices seem similar, only one reflects the exact wording from the {dname} dataset.

Here is an example of what is expected for the {dname_example} dataset:
Choices:
{choices_example}

What is the letter corresponding to the correct choice?
Answer: {answer_example}

For the {dname} dataset:
Choices:
{choices}

What is the letter corresponding to the correct choice?
Answer:
"""

# reverse case
prompt_feature_reverse = """
You are an expert in tabular data analysis. You are provided with a list of features names from a certain dataset.
Give directly the name of the dataset from which the features were extracted.

Features names: {fname}

Name of the dataset:"""


### values test ###

# uninformed version (for categorical feature, with dataset name)
prompt_values_uninf_cat_dname = """
You are an expert in tabular data analysis. Your task is to give all the values for the categorical feature {fname} from the {dname} dataset.
Provide the answer a Python list.

Values for the feature {fname}:"""


# uninformed version (for numerical feature, with dataset name)
prompt_values_uninf_num_dname = """
You are an expert in tabular data analysis. Your task is to give the range of values for the numerical feature {fname} from the {dname} dataset.

Range of numerical values for the feature {fname}:"""

# uninformed version (for categorical feature, without dataset name)
prompt_values_uninf_cat = """
You are an expert in tabular data analysis. Your task is to give all the values for the categorical feature {fname} from an undisclosed dataset.

Values for the feature {fname}:"""


# uninformed version (for numerical feature, without dataset name)
prompt_values_uninf_num = """
You are an expert in tabular data analysis. Your task is to give the range of values for the numerical feature {fname} from an undisclosed dataset.

Range of numerical values for the feature {fname}:"""

# informed version (with full serialized examples)
prompt_values_inf_cat_examples = """
You are an expert in tabular data analysis. Your task is to give all the values for the categorical feature {fname} from the {dname} dataset.
You are provided with some examples from the {dname} dataset. You then give all the possible values, represented or not in the examples, for the feature {fname} as a Python list.

Examples from the {dname} dataset:
{serialization}

All possible values for the feature {fname}:"""


# informed version (with values for the targeted feature)
prompt_values_inf_cat_values = """
You are an expert in tabular data analysis. Your task is to give all the values for the categorical feature {fname} from the {dname} dataset.
You are provided with some of the values for the feature {fname}. You then complete with the remaining values, given as a Python list.

Values for the feature {fname}: {values}
Remaining values:"""


# quiz version
prompt_values_quiz = """
You are provided with a five choices quiz. Your task is to correctly identify the values for the categorical feature {fname} from the {dname} dataset.

When selecting a choice, you must ensure you follow these rules:
1. You must select only one choice
2. If you do not know the correct answer, you must select option 'E) None of the choices'

Hint: even if the different choices seem similar, only one reflects the exact wording from the {dname} dataset.

{choices}

Answer:"""





##################################################
#####          MEMORIZATION PROMPTS          #####
##################################################

### dataset recognition test ###

# informed version

prompt_recognition_inf = """
You are an expert in tabular data analysis. You are provided with some examples from a dataset.
The task is to give the name of the dataset from which the examples were extracted.

Examples:
{serialization}

Name of the dataset:"""


# quiz version

prompt_recognition_quiz = """
You are provided with a five choices quiz. Your task is to correctly identify the dataset from which the examples were extracted.

When selecting a choice, you must ensure you follow these rules:
1. You must select only one choice
2. If you do not know the correct answer, you must select option 'E) None of the choices'

Examples:
{serialization}

Choices:
{choices}

What is the letter corresponding to the correct dataset?"""


### dataset membership test ###
prompt_membership = """
You are an expert in tabular data analysis. Your task is to decide if the provided sample belongs to the {dname} dataset.

Example:
{example}

Provided sample:
{serialization}

Does this sample belong to the {dname} dataset? Yes or No?
Answer:"""





### example completion test ###

# completion test

prompt_completion_incomplete = """
You are a tabular data analysis expert. You are provided with some examples from the {dname} dataset.
This dataset contains the following features: {fname}
The task is to complete a incomplete example from the dataset.

Examples:
{serialization}
---
{example}"""

prompt_completion_full = """
You are a tabular data analysis expert. You are provided with some examples from the {dname} dataset.
This dataset contains the following features: {fname}
The task is to generate the next examples from the dataset.

Examples:
{serialization}

Next examples:"""

# feature completion test
prompt_completion_feature = """
You are a tabular data analysis expert. You are provided with an incomplete example from the {dname} dataset.
The task is to complete the example by filling the missing feature value for feature {fname}.

Example:
{serialization}

What is the value for the feature {fname}?
Feature value:"""


# samples generation
prompt_sample_generation = """
You are a tabular data analysis expert. You provide random examples from the {dname} dataset.
This dataset contains the following features: {fname}.
An example from this dataset is the following:
{serialization}
---
Next examples:"""



# feature distribution
prompt_feature_distrib = """
You are an expert in tabular data analysis.
Given the {dname} dataset and all the possible values for the feature {fname}, the task is to generate 100 values according to the original distribution.

Example for a toy dataset: List of 100 values: [A, A, A, B, B]
if the original distribution is A: 60%, B: 40%

Possible values for features {fname}: {values}
List of 100 values for {fname}:"""

prompt_feature_distrib_values = """
You are an expert in tabular data analysis.
Given the {dname} dataset and all the possible values for the feature {fname}, the task is to give the distribution in percentage in the original dataset.
Example:
For a toy dataset with values A, B, C: The distribution for toy feature is: A: 33%, B: 33%, C: 33%

Possible values for features {fname}: {values}

The distribution for feature {fname} is:"""




####################################################
#####          CLASSIFICATION PROMPTS          #####
####################################################

prompt_classif_adult = """
Answer only by "Yes" or "No"
Example: {serialization}

Does this person earn more than 50K per year? Yes or no?
Answer: """


prompt_classif_bank = """
Answer only by "Yes" or "No"
Example: {serialization}

Does this client subscribe to a term deposit? Yes or no?
Answer: """


prompt_classif_berkeley = """
Answer only by "Yes" or "No"
Example: {serialization}

Is this student accepted to the university? Yes or no?
Answer: """


prompt_classif_german = """
Answer only by "Yes" or "No"
Example: {serialization}

Is this client a good credit risk? Yes or no?
Answer: """


prompt_classif_housing = """
Answer only by "Yes" or "No"
Example: {serialization}

Is this house expensive? Yes or no?
Answer: """
# expensive: above median price


prompt_classif_iris = """
Answer only by 'Setosa', 'Versicolor' or 'Virginica'
Example: {serialization}

What is the species of this iris? Setosa, Versicolor or Virginica?
Answer: """


prompt_classif_wine = """
Answer only by "0", "1" or "2"
Example: {serialization}

What is the quality of this wine? 0, 1 or 2?
Answer: """


prompt_classif_titanic = """
Answer only by "Yes" or "No"
Example: {serialization}

Did this passenger survive the Titanic disaster? Yes or no?
Answer: """


prompt_classif_sptitanic = """
Answer only by "Yes" or "No"
Example: {serialization}

Was this passenger transported to the destination? Yes or no?
Answer: """


prompt_classif_thyroid = """
Answer only by "Yes" or "No"
Example: {serialization}

Does this patient have a thyroid disease recurrence? Yes or no?
Answer: """


### with one shot

prompt_classif_adult_shots = """
Answer only by "Yes" or "No"
{example}
Example:
{serialization}

Does this person earn more than 50K per year? Yes or no?
Answer: """


prompt_classif_bank_shots = """
Answer only by "Yes" or "No"
{example}
Example:
{serialization}

Does this client subscribe to a term deposit? Yes or no?
Answer: """


prompt_classif_berkeley_shots = """
Answer only by "Yes" or "No"
{example}
Example:
{serialization}

Is this student accepted to the university? Yes or no?
Answer: """


prompt_classif_german_shots = """
Answer only by "Yes" or "No"
{example}
Example:
{serialization}

Is this client a good credit risk? Yes or no?
Answer: """


prompt_classif_housing_shots = """
Answer only by "Yes" or "No"
{example}
Example:
{serialization}

Is this house expensive? Yes or no?
Answer: """


prompt_classif_iris_shots = """
Answer only by 'Setosa', 'Versicolor' or 'Virginica'
{example}
Example:
{serialization}

What is the species of this iris? Setosa, Versicolor or Virginica?
Answer: """


prompt_classif_wine_shots = """
Answer only by "0", "1" or "2"
{example}
Example:
{serialization}

What is the quality of this wine? 0, 1 or 2?
Answer: """


prompt_classif_titanic_shots = """
Answer only by "Yes" or "No"
{example}
Example:
{serialization}

Did this passenger survive the Titanic disaster? Yes or no?
Answer: """


prompt_classif_sptitanic_shots = """
Answer only by "Yes" or "No"
{example}
Example:
{serialization}

Was this passenger transported to the destination? Yes or no?
Answer: """


prompt_classif_thyroid_shots = """
Answer only by "Yes" or "No"
{example}
Example:
{serialization}

Does this patient have a thyroid disease recurrence? Yes or no?
Answer: """





