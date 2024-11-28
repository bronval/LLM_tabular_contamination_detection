
### Contains the functions to use the LLMs and perform the different tests ###


from prompts import *
from data_processing import *
from data_loader import load_dataset
from LLM_output_parser import *
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import random
import numpy as np
from openai import OpenAI

import google.generativeai as genai
from google.generativeai import GenerationConfig

import warnings
warnings.filterwarnings("ignore")


with open("google_key.txt") as f:
    google_api_key = f.read()
genai.configure(api_key=google_api_key)


with open("openai_key.txt") as f:
    openai_api_key = f.read()


DATASET_NAMES = [
    "adult", "bank",
    #"berkeley",
    "german", "housing", "iris", "wine", "titanic", "spaceship_titanic", 
    "thyroid_diff", #"MathE"
    ]
DATASET_NAMES_LONG = [
    "adult income", "bank marketing",
    #"berkeley admissions",
    "german credit",
    "california housing", "iris", "wine", "titanic", "spaceship_titanic",
    "thyroid disease recurrence", #"MathE"
    ]


LLM_NAMES = {
    "llama2": "meta-llama/Llama-2-7b-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "llama3.1": "meta-llama/Meta-Llama-3.1-8B",
    "gemma": "google/gemma-7b",
    "gemma2": "google/gemma-2-9b",
    "t0": "bigscience/T0",
    "gptj": "EleutherAI/gpt-j-6b",
    "phi2": "microsoft/phi-2",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "gpt3": "gpt-3.5-turbo-0125",
    "gpt4": "gpt-4-0125-preview",
    "gpt4o": "gpt-4o-2024-08-06",
    "gemini1.0": "gemini-1.0-pro",
    "gemini1.5": "gemini-1.5-pro",
    "llama3.2": "meta-llama/Llama-3.2-3B"
}


TARGETS_FEATURES = {
    "adult income": "income",
    "bank marketing": "y",
    "berkeley admissions": "Admission",
    "german credit": "Risk",
    "california housing": "median_house_value",
    "iris": "target",
    "wine": "target",
    "titanic": "Survived",
    "spaceship_titanic": "Transported",
    "thyroid disease recurrence": "Recurred",
    "MathE": None
}


def run_model(prompt, model, tokenizer, max_new_tokens=256, do_sample=False, temperature=0.0):
    """
    Run the given model with the given inputs

    Returns the output as a string
    """

    # check if model is gpt
    if type(model) == str and "gpt-" in model:
        client = OpenAI(api_key=openai_api_key)
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            seed=0,
            max_tokens=max_new_tokens,
            temperature=temperature
        )
        outputs = response.choices[0].message.content
        return outputs
    elif type(model) == str and "gemini" in model:
        model = genai.GenerativeModel(model, generation_config=GenerationConfig(max_output_tokens=max_new_tokens, temperature=temperature))
        answer = model.generate_content(prompt)
        try:
            out = answer.text
        except:
            out = ""
        return out
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    except RuntimeError:
        inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=tokenizer.eos_token_id, temperature=temperature)
    outputs_str = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return outputs_str


###########################################
#####          DATASET TESTS          #####
###########################################

def dataset_description_test(model, tokenizer, dname : str, temperature : float = 0.0, return_full_output : bool = False, crop_output : bool = True) -> str:
    """
    Test the given LLM with the dataset description test
    """
    prompt = get_prompt(prompt_dataset_desc, [dname])
    outputs_str = run_model(prompt, model, tokenizer, do_sample=False, temperature=temperature)

    df = load_dataset(dname)
    size = df.shape
    outputs_str += f"\n\nExpected size: {size}"

    if return_full_output:
        return outputs_str
    if crop_output:
        outputs_str = outputs_str[len(prompt):]
    return outputs_str
    


############################################
#####          FEATURES TESTS          #####
############################################

def features_uninf_test(model,
                        tokenizer,
                        dname : str,
                        temperature : float = 0.0,
                        return_full_output : bool = False,
                        crop_output : bool = True,
                        verbose=False) -> str:
    """
    Features uninformed test

    Parameters:
        - model: the model to use from huggingface
        - tokenizer: the tokenizer to use from huggingface
        - dname: the name of the dataset to test
        - temperature: the temperature to use for the LLM
        - return_full_output: whether to return the full output with the prompt or only the relevant part
        - crop_output: whether to crop the output to remove the prompt
        - verbose: whether to print the output

    Returns:
        - bool, the test is passed or not
        - dict, the matches found in the output
    """
    prompt = get_prompt(prompt_feature_uninf, [dname])
    outputs_str = run_model(prompt, model, tokenizer, max_new_tokens=256, do_sample=False, temperature=temperature)
    
    df = load_dataset(dname)
    features = df.columns.tolist()
    if dname == "spaceship_titanic" and "PassengerId" in features:
        features.remove("PassengerId")
    elif dname == "MathE" and "Student ID" in features:
        features.remove("Student ID")

    if crop_output:
        outputs_str = outputs_str[len(prompt):]

    res, matches = parse_LLM_output_words(outputs_str, features)

    if verbose:
        print(f"output:\n{outputs_str}")
        print(f"Expected:\n{features}")
        print(f"matches:\n{matches}")
        print(f"decision: {res}")

    return res, matches


def features_inf_test(model,
                      tokenizer,
                      dname : str,
                      n_given_features : int = 3,
                      n_repeat : int = 1,
                      temperature : float = 0.0,
                      return_full_output : bool = False,
                      crop_output : bool = True,
                      verbose=False) -> str:
    """
    Features informed test

    Parameters:
        - model: the model to use from huggingface
        - tokenizer: the tokenizer to use from huggingface
        - dname: the name of the dataset to test
        - n_given_features: the number of features to give to the LLM
        - n_repeat: the number of times to repeat the test
        - temperature: the temperature to use for the LLM
        - return_full_output: whether to return the full output with the prompt or only the relevant part
        - crop_output: whether to crop the output to remove the prompt
        - verbose: whether to print the output

    Returns:
        - bool, the test is passed or not
    """
    df = load_dataset(dname)
    features = df.columns.values
    output = ""
    final_decision = False
    for i in range(n_repeat):
        n_given_features = min(n_given_features, len(features)-2) # -2 to have at least two columns to generate
        selected_features = np.random.choice(features, size=n_given_features, replace=False)
        features_str = "[" + ", ".join(selected_features) + "]"
        prompt = get_prompt(prompt_feature_inf, [dname, features_str])
        outputs_str = run_model(prompt, model, tokenizer, max_new_tokens=256, do_sample=False, temperature=temperature)
        
        if crop_output:
            outputs_str = outputs_str[len(prompt):]

        feat_copy = features.copy().tolist()  # not a problem if features given in prompt do not appear in the output
        for feat in selected_features:
            feat_copy.remove(feat)
        if dname == "spaceship_titanic" and "PassengerId" in feat_copy:
            feat_copy.remove("PassengerId")
        elif dname == "MathE" and "Student ID" in feat_copy:
            feat_copy.remove("Student ID")
        
        res, matches = parse_LLM_output_words(outputs_str, feat_copy)

        # if obtain res only once, consider the test as passed
        if res:
            final_decision = True

        if verbose:
            print(f"output:\n{outputs_str}")
            print(f"Expected:\n{feat_copy}")
            print(f"matches:\n{matches}")
            print(f"decision: {res}")
            print(f"final decision: {final_decision}")
    return final_decision


def features_reverse_test(model,
                          tokenizer,
                          dname : str,
                          temperature : float = 0.0,
                          return_full_output : bool = False,
                          crop_output : bool = True,
                          verbose=False) -> str:
    """
    Feature recognition test

    Parameters:
        - model: the model to use from huggingface
        - tokenizer: the tokenizer to use from huggingface
        - dname: the name of the dataset to test
        - temperature: the temperature to use for the LLM
        - return_full_output: whether to return the full output with the prompt or only the relevant part
        - crop_output: whether to crop the output to remove the prompt
        - verbose: whether to print the output

    Returns:
        - bool, the test is passed or not
    """
    df = load_dataset(dname)
    features = df.columns.values
    features_str = "[" + ", ".join(features) + "]"
    prompt = get_prompt(prompt_feature_reverse, [None, features_str])
    outputs_str = run_model(prompt, model, tokenizer, max_new_tokens=16, do_sample=False, temperature=temperature)
    
    if crop_output:
        outputs_str = outputs_str[len(prompt):]
    
    dname_short = DATASET_NAMES[DATASET_NAMES_LONG.index(dname)]
    res, matches = parse_LLM_output_words(outputs_str, [dname])
    if not res:
        res, matches = parse_LLM_output_words(outputs_str, [dname_short])
    
    if verbose:
        print(f"output:\n{outputs_str}")
        print(f"Expected:\n{dname} / {dname_short}")
        print(f"matches:\n{matches}")
        print(f"decision: {res}")

    return res



##########################################
#####          VALUES TESTS          #####
##########################################

def values_uninf_test(model,
                      tokenizer,
                      dname : str,
                      feature_names : list,
                      is_categorical : list,
                      give_dname : bool = True,
                      temperature : float = 0.0,
                      return_full_output : bool = False,
                      crop_output : bool = True,
                      verbose=False) -> str:
    """
    Parameters:
        - model: the model to use from huggingface
        - tokenizer: the tokenizer to use from huggingface
        - dname: the name of the dataset to test
        - feature_names: the names of the features to test
        - is_categorical: whether the features are categorical or not
        - give_dname: whether to give the dataset name in the prompt
        - temperature: the temperature to use for the LLM
        - return_full_output: whether to return the full output with the prompt or only the relevant part
        - crop_output: whether to crop the output to remove the prompt
        - verbose: whether to print the output

    Returns:
        - bool, the test is passed or not
    """
    output = ""
    df = load_dataset(dname)
    final_decision = False
    for i in range(len(feature_names)):
        if is_categorical[i]:
            if give_dname:
                prompt = get_prompt(prompt_values_uninf_cat_dname, [dname, feature_names[i]])
            else:
                prompt = get_prompt(prompt_values_uninf_cat, [None, feature_names[i]])
        else:
            if give_dname:
                prompt = get_prompt(prompt_values_uninf_num_dname, [dname, feature_names[i]])
            else:
                prompt = get_prompt(prompt_values_uninf_num, [None, feature_names[i]])
        outputs_str = run_model(prompt, model, tokenizer, max_new_tokens=128, do_sample=False, temperature=temperature)
        
        if crop_output:
            outputs_str = outputs_str[len(prompt):]
        
        expected_values = df[feature_names[i]].unique().tolist()
        res, matches = parse_LLM_output_words(outputs_str, expected_values)

        if res:     # if one feature gives all the values, consider the test as passed
            final_decision = True

        if verbose:
            print(f"output:\n{outputs_str}")
            print(f"Expected:\n{expected_values}")
            print(f"matches:\n{matches}")
            print(f"decision: {res}")
            print(f"final decision: {final_decision}")

        
    return final_decision

    
def values_inf_examples_test(model,
                             tokenizer,
                             dname : str,
                             feature_names : list,
                             is_categorical : list,
                             n_given_examples : int = 3,
                             random_examples : bool = False,
                             features_sep : str = ", ",
                             value_sep : str = " is ",
                             specifier : str = "The ",
                             temperature : float = 0.0,
                             return_full_output : bool = False,
                             crop_output : bool = True,
                             verbose=False) -> str:
    """
    Values informed test

    Parameters:
        - model: the model to use from huggingface
        - tokenizer: the tokenizer to use from huggingface
        - dname: the name of the dataset to test
        - feature_names: the names of the features to test
        - is_categorical: whether the features are categorical or not
        - n_given_examples: the number of examples to give to the LLM
        - random_examples: whether to take random examples or the first ones
        - features_sep: the separator to use between features
        - value_sep: the separator to use between values
        - specifier: the specifier to use at the beginning of the serialized example
        - temperature: the temperature to use for the LLM
        - return_full_output: whether to return the full output with the prompt or only the relevant part
        - crop_output: whether to crop the output to remove the prompt
        - verbose: whether to print the output

    Returns:
        - bool, the test is passed or not
    """
    df = load_dataset(dname)
    output = ""
    final_decision = False
    
    
    for i in range(len(feature_names)):

        if dname == "spaceship_titanic" and (feature_names[i] == "Transported" or feature_names[i] == "HomePlanet"):
            continue  # ignore as it is only True or False, once given a value it is trivial to complete it

        expected_values = df[feature_names[i]].unique().tolist()
        n_given_examples = min(n_given_examples, len(expected_values)-1) # want at least 1 feature to be found by the llm

        if random_examples:
            serialization = serialize_samples(df, n_samples=n_given_examples, random_sampling=True, seed=0,
                                                features_sep=features_sep, value_sep=value_sep, specifier=specifier, remove_last_sep=True)
        else:
            serialization = serialize_samples(df, n_samples=n_given_examples, random_sampling=False,
                                            features_sep=features_sep, value_sep=value_sep, specifier=specifier, remove_last_sep=True)

        prompt = get_prompt(prompt_values_inf_cat_examples, [dname, feature_names[i], serialization])
        outputs_str = run_model(prompt, model, tokenizer, max_new_tokens=128, do_sample=False, temperature=temperature)
        
        if crop_output:
            outputs_str = outputs_str[len(prompt):]

        # expected_values = df[feature_names[i]].unique().tolist()
        given_examples = df.head(n_given_examples)
        values_known = given_examples[feature_names[i]].unique().tolist()
        for val in values_known:
            expected_values.remove(val)
        res, matches = parse_LLM_output_words(outputs_str, expected_values)

        if res:
            final_decision = True
        
        if verbose:
            print(prompt)
            print(f"output:\n{outputs_str}")
            print(f"Expected:\n{expected_values}")
            print(f"matches:\n{matches}")
            print(f"decision: {res}")
            print(f"final decision: {final_decision}")

    return final_decision



###############################################
#####          RECOGNITION TESTS          #####
###############################################

def recognition_dataset_test(model,
                             tokenizer,
                             dname : str,
                             n_given_examples : int = 3,
                             random_examples : bool = False,
                             features_sep : str = ", ",
                             value_sep : str = " is ",
                             specifier : str = "The ",
                             return_full_output : bool = False,
                             crop_output : bool = True,
                             verbose=False) -> str:
    """
    Example recognition test

    Parameters:
        - model: the model to use from huggingface
        - tokenizer: the tokenizer to use from huggingface
        - dname: the name of the dataset to test
        - n_given_examples: the number of examples to give to the LLM
        - random_examples: whether to take random examples or the first ones
        - features_sep: the separator to use between features
        - value_sep: the separator to use between values
        - specifier: the specifier to use at the beginning of the serialized example
        - return_full_output: whether to return the full output with the prompt or only the relevant part
        - crop_output: whether to crop the output to remove the prompt
        - verbose: whether to print the output

    Returns:
        - bool, the test is passed or not
    """
    df = load_dataset(dname)
    if random_examples:
        serialization = serialize_samples(df, n_samples=n_given_examples, random_sampling=True, seed=0,
                                            features_sep=features_sep, value_sep=value_sep, specifier=specifier, remove_last_sep=True)
    else:
        serialization = serialize_samples(df, n_samples=n_given_examples, random_sampling=False,
                                            features_sep=features_sep, value_sep=value_sep, specifier=specifier, remove_last_sep=True)
    prompt = get_prompt(prompt_recognition_inf, [None, None, serialization])
    outputs_str = run_model(prompt, model, tokenizer, max_new_tokens=16, do_sample=False)

    if crop_output:
        outputs_str = outputs_str[len(prompt):]

    if dname == "MathE":
        uppercase_important = True
    else:
        uppercase_important = False
    dname_short = DATASET_NAMES[DATASET_NAMES_LONG.index(dname)]
    res, matches = parse_LLM_output_words(outputs_str, [dname], uppercase_important=uppercase_important)
    if not res:
        res, matches = parse_LLM_output_words(outputs_str, [dname_short], uppercase_important=uppercase_important)
    
    if verbose:
        print(f"output:\n{outputs_str}")
        print(f"Expected:\n{dname} / {dname_short}")
        print(f"matches:\n{matches}")
        print(f"decision: {res}")

    return res


def membership_dataset_test(model,
                            tokenizer,
                            dname : str,
                            n_tests : int = 10,
                            features_sep : str = ", ",
                            value_sep : str = " is ",
                            specifier : str = "The ",
                            return_full_output : bool = False,
                            crop_output : bool = True,
                            verbose=False) -> str:
    """
    Membership test

    Parameters:
        - model: the model to use from huggingface
        - tokenizer: the tokenizer to use from huggingface
        - dname: the name of the dataset to test
        - n_tests: the number of tests to perform
        - features_sep: the separator to use between features
        - value_sep: the separator to use between values
        - specifier: the specifier to use at the beginning of the serialized example
        - return_full_output: whether to return the full output with the prompt or only the relevant part
        - crop_output: whether to crop the output to remove the prompt
        - verbose: whether to print the output

    Returns the score of the model
    """
    df = load_dataset(dname)
    score = 0
    n_yes = 0
    n_no = 0
    outputs_str = ""
    expected = ["No", "Yes"]
    for i in range(n_tests):
        sample = df.iloc[i]
        if i % 2 == 0:
            # sample = shuffle_sample(sample, df.columns, n_swapped=2)
            sample = shuffle_example_value(df, sample, n_swapped=len(df.columns)-3)
        
        # create example for 1-shot prompt
        example = serialize_sample(df.iloc[len(df)-1-i], features_sep=features_sep, value_sep=value_sep, specifier=specifier, remove_last_sep=True)
        example += "\nAnswer: Yes\n"
        example += serialize_sample(shuffle_example_value(df, df.iloc[len(df)-1-i], n_swapped=len(df.columns)-3), features_sep=features_sep, value_sep=value_sep, specifier=specifier, remove_last_sep=True)
        # example += serialize_sample(shuffle_sample(df.iloc[len(df)-1-i], df.columns, n_swapped=3), features_sep=features_sep, value_sep=value_sep, specifier=specifier, remove_last_sep=True)
        example += "\nAnswer: No\n"

        serialization = serialize_sample(sample,
                                         features_sep=features_sep,
                                         value_sep=value_sep,
                                         specifier=specifier,
                                         remove_last_sep=True)
        prompt = get_prompt(prompt_membership, [dname, None, serialization, example])
        out = run_model(prompt, model, tokenizer, max_new_tokens=4, do_sample=False)
        response = out[len(prompt):]
        if not crop_output:
            response = out
        if expected[i % 2] in response and expected[(i+1) % 2] not in response:
            score += 1
        if "Yes" in response:
            n_yes += 1
        elif "No" in response:
            n_no += 1
        if i == 0 and return_full_output:
            outputs_str += out
            outputs_str += f"\nExpected: {expected[i % 2]}"
    outputs_str += f"\n\nScore: {score}/{n_tests}  (n_Yes: {n_yes}, n_no: {n_no})"
    
    if verbose:
        print(outputs_str)

    return score



##############################################
#####          COMPLETION TESTS          #####
##############################################

def completion_incomplete_test(model,
                               tokenizer,
                               dname : str,
                               n_removed_features : int = 3,
                               n_given_examples : int = 3,
                               random_examples : bool = False,
                               features_sep : str = ", ",
                               value_sep : str = " is ",
                               specifier : str = "The ",
                               return_full_output : bool = False,
                               crop_output : bool = True,
                               verbose=False) -> str:
    """
    Completion incomplete test

    Parameters:
        - model: the model to use from huggingface
        - tokenizer: the tokenizer to use from huggingface
        - dname: the name of the dataset to test
        - n_removed_features: the number of features to remove from the dataset
        - n_given_examples: the number of examples to give to the LLM
        - random_examples: whether to take random examples or the first ones
        - features_sep: the separator to use between features
        - value_sep: the separator to use between values
        - specifier: the specifier to use at the beginning of the serialized example
        - return_full_output: whether to return the full output with the prompt or only the relevant part
        - crop_output: whether to crop the output to remove the prompt
        - verbose: whether to print the output

    Returns:
        - bool, the test is passed or not
        - dict, the matches found in the output
    """
    df = load_dataset(dname)
    features = df.columns.values
    n_removed_features = min(n_removed_features, len(features)-1)
    features_removed = features[len(features)-n_removed_features:]
    features_str = "[" + ", ".join(features_removed) + "]"
    if random_examples:
        serialization = serialize_samples(df, n_samples=n_given_examples, random_sampling=True, seed=0,
                                            features_sep=features_sep, value_sep=value_sep, specifier=specifier, remove_last_sep=True)
        idx = random.randint(0, len(df)-1) # np.random.randint(0, len(df))
        # complete_ex = serialize_sample(df.iloc[idx],
        #                                features_sep=features_sep, value_sep=value_sep, specifier=specifier, remove_last_sep=True)
        incomplete_ex = serialize_sample(df.iloc[idx], columns_ignored=features_removed,
                                         features_sep=features_sep, value_sep=value_sep, specifier=specifier, remove_last_sep=True)
    else:
        serialization = serialize_samples(df, n_samples=n_given_examples, random_sampling=False,
                                            features_sep=features_sep, value_sep=value_sep, specifier=specifier, remove_last_sep=True)
        # complete_ex = serialize_sample(df.iloc[n_given_examples],
        #                                  features_sep=features_sep, value_sep=value_sep, specifier=specifier, remove_last_sep=True)
        incomplete_ex = serialize_sample(df.iloc[n_given_examples], columns_ignored=features_removed,
                                         features_sep=features_sep, value_sep=value_sep, specifier=specifier, remove_last_sep=True)
    prompt = get_prompt(prompt_completion_incomplete, [dname, features_str, serialization, incomplete_ex])
    outputs_str = run_model(prompt, model, tokenizer, max_new_tokens=256, do_sample=False)

    if crop_output:
        outputs_str = outputs_str[len(prompt):]
    if features_sep != "\n":
        outputs_str = outputs_str.split("\n")[0]

    expected_example = df.iloc[idx]

    res, matches = parse_LLM_output_example(outputs_str, expected_example, features_removed)

    if verbose:
        print(f"output:\n{outputs_str}")
        print(f"Expected:\n{expected_example}")
        print(f"matches:\n{matches}")
        print(f"decision: {res}")

    return res, matches


def completion_full_test(model,
                         tokenizer,
                         dname : str,
                         n_given_examples : int = 3,
                         random_examples : bool = False,
                         features_sep : str = ", ",
                         value_sep : str = " is ",
                         specifier : str = "The ",
                         return_full_output : bool = False,
                         crop_output : bool = True,
                         verbose=False) -> str:
    """
    Completion full test

    Parameters:
        - model: the model to use from huggingface
        - tokenizer: the tokenizer to use from huggingface
        - dname: the name of the dataset to test
        - n_given_examples: the number of examples to give to the LLM
        - random_examples: whether to take random examples or the first ones
        - features_sep: the separator to use between features
        - value_sep: the separator to use between values
        - specifier: the specifier to use at the beginning of the serialized example
        - return_full_output: whether to return the full output with the prompt or only the relevant part
        - crop_output: whether to crop the output to remove the prompt
        - verbose: whether to print the output

    Returns:
        - bool, the test is passed or not
        - dict, the matches found in the output
    """
    df = load_dataset(dname)
    features = df.columns.values
    features_str = "[" + ", ".join(features) + "]"
    serialization = serialize_samples(df, n_samples=n_given_examples, random_sampling=random_examples, seed=0,
                                      features_sep=features_sep, value_sep=value_sep, specifier=specifier, remove_last_sep=True)
    prompt = get_prompt(prompt_completion_full, [dname, features_str, serialization])
    outputs_str = run_model(prompt, model, tokenizer, max_new_tokens=512, do_sample=False)

    if crop_output:
        outputs_str = outputs_str.split("Next examples:")[1]
        # outputs_str = outputs_str[len(prompt):]
    if features_sep != "\n":
        outputs_str = outputs_str.split("\n")
        if len(outputs_str[0].split(" ")) < 5:  # try to avoid the case where the first line is not the completion (example: Next examples: ...)
            outputs_str = outputs_str[1]
        else:
            outputs_str = outputs_str[0]
            

    expected_example = df.iloc[n_given_examples]

    res, matches = parse_LLM_output_example(outputs_str, expected_example, df.columns.tolist())

    if verbose:
        print(f"output:\n{outputs_str}")
        print(f"Expected:\n{expected_example}")
        print(f"matches:\n{matches}")
        print(f"decision: {res}")

    return res, matches


def completion_feature_test(model,
                            tokenizer,
                            dname : str,
                            feature_name : str,
                            n_tests : int = 3,
                            random_example : bool = False,
                            features_sep : str = ", ",
                            value_sep : str = " is ",
                            specifier : str = "The ",
                            return_full_output : bool = False,
                            crop_output : bool = True,
                            verbose=False) -> str:
    """
    Feature completion test

    Parameters:
        - model: the model to use from huggingface
        - tokenizer: the tokenizer to use from huggingface
        - dname: the name of the dataset to test
        - feature_name: the name of the feature to complete
        - n_tests: the number of tests to perform
        - random_example: whether to take a random example or the first one
        - features_sep: the separator to use between features
        - value_sep: the separator to use between values
        - specifier: the specifier to use at the beginning of the serialized example
        - return_full_output: whether to return the full output with the prompt or only the relevant part
        - crop_output: whether to crop the output to remove the prompt
        - verbose: whether to print the output

    Returns:
        - bool, the test is passed or not
    """
    df = load_dataset(dname)
    output = ""
    final_decision = False

    for i in range(n_tests):
        if random_example:
            sample = df.iloc[np.random.randint(0, len(df))]
        else:
            sample = df.iloc[i]
        serialization = serialize_sample(sample, columns_ignored=[feature_name],
                                        features_sep=features_sep, value_sep=value_sep, specifier=specifier, remove_last_sep=True)
        prompt = get_prompt(prompt_completion_feature, [dname, feature_name, serialization])
        outputs_str = run_model(prompt, model, tokenizer, max_new_tokens=64, do_sample=False)
        
        if crop_output:
            outputs_str = outputs_str[len(prompt):]
        if features_sep != "\n":
            outputs_str = outputs_str.split("\n")[0]

        res, matches = parse_LLM_output_example(outputs_str, sample, [feature_name])

        if res:
            final_decision = True
        
        if verbose:
            print(f"output:\n{outputs_str}")
            print(f"Expected:\n{sample}")
            print(f"matches:\n{matches}")
            print(f"decision: {res}")
            print(f"final decision: {final_decision}")
        
    return final_decision



##################################################
#####          CLASSIFICATION TESTS          #####
##################################################


def classification_test(model, tokenizer, dname, n_tests=100, temperature=0.0, return_full_output=False, crop_output=True,
                        features_sep=", ", value_sep=" is ", specifier="The ", shots=False):
    """
    Classification test

    Parameters:
        - model: the model to use from huggingface
        - tokenizer: the tokenizer to use from huggingface
        - dname: the name of the dataset to test
        - n_tests: the number of tests to perform
        - temperature: the temperature to use for the LLM
        - return_full_output: whether to return the full output with the prompt or only the relevant part
        - crop_output: whether to crop the output to remove the prompt
        - features_sep: the separator to use between features
        - value_sep: the separator to use between values
        - specifier: the specifier to use at the beginning of the serialized example
        - shots: whether to use few-shots or not

    Returns the accuracy score of the model
    """
    df = load_dataset(dname)
    df = df.sample(n=n_tests, random_state=42)
    target = TARGETS_FEATURES[dname]

    score = 0
    n_yes_classif = 0
    n_no_classif = 0
    others = [0, 0, 0]
    outs = {}
    for i in range(n_tests):
        sample = df.iloc[i]
        y = sample[target]
        sample = sample.drop(target)
        serialization = serialize_sample(sample, features_sep=features_sep, value_sep=value_sep, specifier=specifier, remove_last_sep=True)
        if shots:
            prompt = get_prompt_classif_shots(dname, serialization)
        else:
            prompt = get_prompt_classif(dname, serialization)
        out = run_model(prompt, model, tokenizer, max_new_tokens=4, do_sample=False, temperature=temperature)
        if i == 0:
            if not crop_output:
                print(prompt)
            print(out)
            print(f"Expected: {y}", flush=True)
        
        if crop_output:
            out = out[len(prompt)-1:]

        out = out.strip().strip("<b>").strip("\n").strip("</b>")
        if evaluate_output_classif(dname, out, y):
            score += 1

        if dname != "iris":
            out = out[:3]
        if out not in outs:
            outs[out] = 1
        else:
            outs[out] += 1
    outs = {k: v for k, v in sorted(outs.items(), key=lambda item: item[1], reverse=True)}
    print(f"{dname}  outputs: {outs}")
    return score / n_tests * 100



#######################################
#####          ALL TESTS          #####
#######################################


def run_all_with_parsing(llm_hf, tokenizer, temperature=0.0, crop_output=True,
                         features_sep : str = ", ",
                         value_sep : str = " is ",
                         specifier : str = "The "):
    """
    Launches all the contamination tests described in the paper

    Parameters:
        - llm_hf: the model to use from huggingface
        - tokenizer: the tokenizer to use from huggingface
        - temperature: the temperature to use for the LLM
        - crop_output: whether to crop the output to remove the prompt
        - features_sep: the separator to use between features
        - value_sep: the separator to use between values
        - specifier: the specifier to use at the beginning of the serialized example

    Returns a dictionary with the results of the tests for each dataset
    """
    features_selected = {
        "adult income" : ["income", "marital.status"],
        "bank marketing" : ["y", "job"],
        "berkeley admissions" : ["Admission", "Major"],
        "german credit" : ["Risk", "Saving accounts"],
        "california housing" : ["ocean_proximity"],
        "iris" : ["target"], # species
        "titanic" : ["Embarked"], # Sex
        "spaceship_titanic" : ["Transported", "Destination"],  # "HomePlanet"
        "thyroid disease recurrence": ["Adenopathy", "Response"],
        "MathE": ["Question Level", "Topic"],
    }
    answers = {}

    # feature list uninformed
    print("\n### FEATURES UNINFORMED TEST ###\n\n", flush=True)
    answers["features_list_uninf"] = {}
    for dname in DATASET_NAMES_LONG:
        print(f"---for dataset {dname}")
        res, matches = features_uninf_test(llm_hf, tokenizer, dname, temperature, return_full_output=True, crop_output=crop_output, verbose=True)
        print("\n\n\n", flush=True)
        answers["features_list_uninf"][dname] = res

    # feature list informed
    print("\n### FEATURES INFORMED TEST ###\n\n", flush=True)
    n_repeat = 3
    n_given_features = 5
    answers["features_list_inf"] = {}
    for dname in DATASET_NAMES_LONG:
        print(f"---for dataset {dname}")
        res = features_inf_test(llm_hf, tokenizer, dname, n_given_features, n_repeat, temperature, return_full_output=True, crop_output=crop_output, verbose=True)
        print("\n\n\n", flush=True)
        answers["features_list_inf"][dname] = res

    # feature values uninformed
    print("\n### VALUES UNINFORMED TEST ###\n\n", flush=True)
    answers["feature_values_uninf"] = {}
    for dname in DATASET_NAMES_LONG:
        if dname == "wine":
            continue
        print(f"---for dataset {dname}")
        features = features_selected[dname]
        res = values_uninf_test(llm_hf, tokenizer, dname, features, [True, True, True], temperature=temperature, return_full_output=True, crop_output=crop_output, verbose=True)
        print("\n\n\n", flush=True)
        answers["feature_values_uninf"][dname] = res

    # # feature values informed
    print("\n### VALUES INFORMED EXAMPLES TEST ###\n\n", flush=True)
    answers["feature_values_inf"] = {}
    for dname in DATASET_NAMES_LONG:
        if dname == "wine":
            continue
        print(f"---for dataset {dname}")
        features = features_selected[dname]
        res = values_inf_examples_test(llm_hf, tokenizer, dname, features, [True, True, True], temperature=temperature, return_full_output=True, crop_output=crop_output, verbose=True, random_examples=True)
        print("\n\n\n", flush=True)
        answers["feature_values_inf"][dname] = res

    # incomplete example completion
    print("\n### COMPLETION INCOMPLETE TEST (random) ###\n\n", flush=True)
    answers["incomplete_completion_random"] = {}
    threshold_incomp_test = 0.5
    n_tests_incomp = 5
    for dname in DATASET_NAMES_LONG:
        print(f"---for dataset {dname}")
        score_incomp = 0
        for i in range(n_tests_incomp):
            res, matches = completion_incomplete_test(llm_hf, tokenizer, dname, n_removed_features=3, n_given_examples=4, random_examples=True, return_full_output=True, crop_output=crop_output, verbose=True)
            if res:
                score_incomp += 1
        score_incomp /= n_tests_incomp
        answers["incomplete_completion_random"][dname] = score_incomp >= threshold_incomp_test
        print(f"Final decision: {answers['incomplete_completion_random'][dname]}    (score: {score_incomp}, threshold: {threshold_incomp_test})")
        print("\n\n\n", flush=True)

    # full example completion
    print("\n### COMPLETION FULL TEST ###\n\n", flush=True)
    answers["full_completion"] = {}
    for dname in DATASET_NAMES_LONG:
        print(f"---for dataset {dname}")
        res, matches = completion_full_test(llm_hf, tokenizer, dname, n_given_examples=4, random_examples=False, return_full_output=True, crop_output=crop_output, verbose=True)
        print("\n\n\n", flush=True)
        answers["full_completion"][dname] = res

    # feature completion
    print("\n### COMPLETION FEATURE TEST ###\n\n", flush=True)
    answers["feature_completion"] = {}
    completion_features = {
        "adult income" : "fnlwgt",
        "bank marketing" : "balance",
        "german credit" : "Credit amount",
        "california housing" : "median_house_value",
        "wine" : "target",
        "iris" : "sepal length (cm)",
        "titanic" : "PassengerId",
        "spaceship_titanic" : "PassengerId",
        "thyroid disease recurrence": "Age",
        "MathE": "Student ID",
    }    # do not consider: berkeley admissions,
    for dname in completion_features.keys():
        print(f"---for dataset {dname}")
        feature = completion_features[dname]
        res = completion_feature_test(llm_hf, tokenizer, dname, feature, n_tests=5, random_example=False, return_full_output=True, crop_output=crop_output, verbose=True)
        print("\n\n\n", flush=True)
        answers["feature_completion"][dname] = res
    
    # recognition features test
    print("\n### FEATURES REVERSE TEST ###\n\n", flush=True)
    answers["recognition_feat"] = {}
    for dname in DATASET_NAMES_LONG:
        print(f"---for dataset {dname}")
        res = features_reverse_test(llm_hf, tokenizer, dname, temperature=temperature, return_full_output=True, crop_output=crop_output, verbose=True)
        print("\n\n\n", flush=True)
        answers["recognition_feat"][dname] = res

    # recognition example test
    print("\n### RECOGNITION DATASET TEST ###\n\n", flush=True)
    answers["recognition_ex"] = {}
    for dname in DATASET_NAMES_LONG:
        print(f"---for dataset {dname}")
        res = recognition_dataset_test(llm_hf, tokenizer, dname, n_given_examples=5, random_examples=False, return_full_output=True, crop_output=crop_output, verbose=True)
        print("\n\n\n", flush=True)
        answers["recognition_ex"][dname] = res
    
    # membership test
    print("\n### MEMBERSHIP TEST ###\n\n", flush=True)
    answers["membership"] = {}
    for dname in DATASET_NAMES_LONG:
        print(f"---for dataset {dname}")
        res = membership_dataset_test(llm_hf, tokenizer, dname, n_tests=100, return_full_output=True, crop_output=crop_output, verbose=True)
        print("\n\n\n", flush=True)
        answers["membership"][dname] = res

    return answers



def run_all_parsing_serialization(llm_hf, tokenizer, temperature=0.0, crop_output=True,
                            features_sep : str = ", ",
                            value_sep : str = " is ",
                            specifier : str = "The "):
    """
    Runs the tests from the paper with the serialization

    Parameters:
        - llm_hf: the model to use from huggingface
        - tokenizer: the tokenizer to use from huggingface
        - temperature: the temperature to use for the LLM
        - crop_output: whether to crop the output to remove the prompt
        - features_sep: the separator to use between features
        - value_sep: the separator to use between values
        - specifier: the specifier to use at the beginning of the serialized example

    Returns a dictionary with the results of the tests for each dataset
    """
    features_selected = {
        "adult income" : ["income", "marital.status"],
        "bank marketing" : ["y", "job"],
        "berkeley admissions" : ["Admission", "Major"],
        "german credit" : ["Risk", "Saving accounts"],
        "california housing" : ["ocean_proximity"],
        "iris" : ["target"], # species
        "titanic" : ["Embarked"], # Sex
        "spaceship_titanic" : ["Transported", "Destination"],  # "HomePlanet"
        "thyroid disease recurrence": ["Adenopathy", "Response"],
        "MathE": ["Question Level", "Topic"],
    }
    answers = {}

    # feature values informed
    print("\n### VALUES INFORMED EXAMPLES TEST ###\n\n", flush=True)
    answers["feature_values_inf"] = {}
    for dname in DATASET_NAMES_LONG:
        if dname == "wine":
            continue
        print(f"---for dataset {dname}")
        features = features_selected[dname]
        res = values_inf_examples_test(llm_hf, tokenizer, dname, features, [True, True, True], temperature=temperature, return_full_output=True, crop_output=crop_output, verbose=True,
                                       features_sep=features_sep, value_sep=value_sep, specifier=specifier)
        print("\n\n\n", flush=True)
        answers["feature_values_inf"][dname] = res

    # incomplete example completion
    print("\n### COMPLETION INCOMPLETE TEST ###\n\n", flush=True)
    answers["incomplete_completion"] = {}
    for dname in DATASET_NAMES_LONG:
        print(f"---for dataset {dname}")
        res, matches = completion_incomplete_test(llm_hf, tokenizer, dname, n_removed_features=3, n_given_examples=4, random_examples=True, return_full_output=True, crop_output=crop_output, verbose=True,
                                                  features_sep=features_sep, value_sep=value_sep, specifier=specifier)
        print("\n\n\n", flush=True)
        answers["incomplete_completion"][dname] = res

    # full example completion
    print("\n### COMPLETION FULL TEST ###\n\n", flush=True)
    answers["full_completion"] = {}
    for dname in DATASET_NAMES_LONG:
        print(f"---for dataset {dname}")
        res, matches = completion_full_test(llm_hf, tokenizer, dname, n_given_examples=4, random_examples=False, return_full_output=True, crop_output=crop_output, verbose=True,
                                            features_sep=features_sep, value_sep=value_sep, specifier=specifier)
        print("\n\n\n", flush=True)
        answers["full_completion"][dname] = res

    # feature completion
    print("\n### COMPLETION FEATURE TEST ###\n\n", flush=True)
    answers["feature_completion"] = {}
    completion_features = {
        "adult income" : "fnlwgt",
        "bank marketing" : "balance",
        "german credit" : "Credit amount",
        "california housing" : "median_house_value",
        "wine" : "target",
        "iris" : "sepal length (cm)",
        "titanic" : "PassengerId",
        "spaceship_titanic" : "PassengerId",
        "thyroid disease recurrence": "Age",
        "MathE": "Student ID",
    }    # do not consider: berkeley admissions,
    for dname in completion_features.keys():
        print(f"---for dataset {dname}")
        feature = completion_features[dname]
        res = completion_feature_test(llm_hf, tokenizer, dname, feature, n_tests=5, random_example=False, return_full_output=True, crop_output=crop_output, verbose=True,
                                      features_sep=features_sep, value_sep=value_sep, specifier=specifier)
        print("\n\n\n", flush=True)
        answers["feature_completion"][dname] = res

    # recognition example test
    print("\n### RECOGNITION DATASET TEST ###\n\n", flush=True)
    answers["recognition_ex"] = {}
    for dname in DATASET_NAMES_LONG:
        print(f"---for dataset {dname}")
        res = recognition_dataset_test(llm_hf, tokenizer, dname, n_given_examples=5, random_examples=False, return_full_output=True, crop_output=crop_output, verbose=True,
                                       features_sep=features_sep, value_sep=value_sep, specifier=specifier)
        print("\n\n\n", flush=True)
        answers["recognition_ex"][dname] = res

    return answers





if __name__ == "__main__":


    ### CHANGE THE MODEL NAME HERE ###
    ### CHOICES: gpt3, gpt4, gpt4o, gemini1.0, gemini1.5, llama2, llama3, llama3.1, gemma, gemma2, phi2, phi3, mistral, t0, gptj
    
    llm_name = "llama3.1"

    ##################################




    print(f"---LLM: {llm_name}---", flush=True)
    llm_hf = LLM_NAMES[llm_name]
    temperature = 0.0


    # for serializaton tests, {specifier}{feature_name}{value_sep}{feature_value}{features_sep}
    serializations = [
        (", ", " is ", "The "),
        (", ", " = ", ""),
        (", ", ": ", ""),
        ("; ", " is equal to ", "The ")
    ]
    # default serialization
    features_sep = ", "
    value_sep = " is "
    specifier = "The "


    # load model and tokenizer
    if llm_name == "gpt3" or llm_name == "gpt4" or llm_name == "gpt4o" or "gemini" in llm_name:
        tokenizer = None
    elif llm_name != "t0":
        tokenizer = AutoTokenizer.from_pretrained(llm_hf, device_map="auto", use_auth_token=True, trust_remote_code=True)
        llm_hf = AutoModelForCausalLM.from_pretrained(llm_hf, device_map="auto", use_auth_token=True, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(llm_hf, device_map="auto", use_auth_token=True)
        llm_hf = AutoModelForSeq2SeqLM.from_pretrained(llm_hf, device_map="auto", use_auth_token=True)

    # to output the results
    crop_output = True
    if llm_name == 't0' or llm_name == "gpt3" or llm_name == "gpt4" or llm_name == "gpt4o" or "gemini" in llm_name:
        crop_output = False    


    ## launch all the tests
    answers = run_all_with_parsing(llm_hf, tokenizer, temperature=temperature, crop_output=crop_output,
                                     features_sep=features_sep, value_sep=value_sep, specifier=specifier)
    print(answers)

    ## save the results to a csv file
    df = convert_results_csv(answers)
    df.to_csv(f"contamination_results/{llm_name}.csv", index=False)


    ## launch the classification tests
    for dname in DATASET_NAMES_LONG:
        if dname == "MathE":  # not a classification task
            continue
        print(f"---for dataset {dname}")
        score = classification_test(llm_hf, tokenizer, dname, n_tests=100, temperature=0.0, return_full_output=True, crop_output=crop_output,
                                  features_sep=features_sep, value_sep=value_sep, specifier=specifier, shots=False)
        print(f"Score: {score} / 100 (0-shot)")
        score = classification_test(llm_hf, tokenizer, dname, n_tests=100, temperature=0.0, return_full_output=True, crop_output=crop_output,
                                  features_sep=features_sep, value_sep=value_sep, specifier=specifier, shots=True)
        print(f"Score: {score} / 100 (1-shot)")
        print("\n\n\n", flush=True)


    ## launch the serialization tests for relevant LLMs
    if llm_name in ["mistral", "gemma2", "gpt4o", "llama3"]:
        i = 0
        for features_sep, value_sep, specifier in serializations:
            print(f"@@@@@ SERIALIZATION: {specifier}{{feature_name}}{value_sep}{{feature_value}}{features_sep}", flush=True)

            answers = run_all_parsing_serialization(llm_hf, tokenizer, temperature=temperature, crop_output=crop_output,
                                                    features_sep=features_sep, value_sep=value_sep, specifier=specifier)
            
            df = convert_results_csv(answers)
            df.to_csv(f"contamination_results/serializations/{llm_name}_seria_{i}_recent_incomp.csv", index=False)
            i += 1
            print("---"*30)


    print("\n\n---DONE---")




