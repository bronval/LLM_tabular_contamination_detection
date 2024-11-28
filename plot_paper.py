
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd


tests_knowledge = ["Features list (inf.)", "Features list (uninf.)", "Recognition (feat.)"]  #, "Feature distrib."] "Dataset desc.", 
tests_memorization = ["Recognition (ex.)", "Feature values (inf.)", "Feature values (uninf.)", "Membership", "Incomplete completion", "Full completion", "Feature completion"]

LLMs = ["GPT-3.5", "GPT-4", "Llama 2", "Llama 3", "Phi-2", "Phi-3", "Gemma", "Mistral", "T0", "GPT-J", "Gemini-1.0", "Gemini-1.5"]

datasets = ["Adult", "Bank", "Berkeley", "Housing", "German", "Iris", "Titanic", "Sp. Titanic", "Wine"]

green = (0, 255, 0)
red = (255, 0, 0)


test_mapping = {"dataset_desc": "Dataset desc.",
                "features_list_inf": "Features list (inf.)",
                "features_list_uninf": "Features list (uninf.)",
                "recognition_feat": "Recognition (feat.)",
                "recognition_ex": "Recognition (ex.)",
                "feature_values_inf": "Feature values (inf.)",
                "feature_values_uninf": "Feature values (uninf.)",
                "membership": "Membership",
                # "incomplete_completion": "Incomplete completion",
                "full_completion": "Full completion",
                "feature_completion": "Feature completion",
                "incomplete_completion_random": "Incomplete completion"
                }


def reverse_test_mapping():
    return {v: k for k, v in test_mapping.items()}


def get_test_category(results, category):
    mapping = reverse_test_mapping()
    res = pd.DataFrame()
    if category == "knowledge":
        for test in tests_knowledge:
            res = res._append(results.loc[mapping[test]])
        return res
    elif category == "memorization":
        for test in tests_memorization:
            if test != "Membership":
                res = res._append(results.loc[mapping[test]])
        return res
    else:
        raise ValueError("Category not found")


def plot_heatmap(results, model_name, ignore_cols=[]):
    """
    Given the results of the contamination tests, plot and save the heatmap

    Parameters:
        - results (pd.DataFrame): the results of the contamination tests
        - model_name (str): the name of the model
        - ignore_cols (list): the columns to ignore (columns correspond to datasets in this case)
    """
    if len(ignore_cols) > 0:
        results.drop(columns=ignore_cols, inplace=True)


    results_knowledge = get_test_category(results, "knowledge")
    results_memorization = get_test_category(results, "memorization")
    results = pd.concat([results_knowledge, results_memorization])
    results.rename(index=test_mapping, inplace=True)

    # compute the sum over all the results
    total = 0
    for i in range(len(results)):
        total += results.iloc[i].sum()
    print(model_name, total)

    only_up = ["gpt4", "llama3", "t0", "gemini1.0", "gemini1.5", "gpt4o", "llama3.1", "mistral"]

    if model_name in only_up:
        plt.figure(figsize=(5, 6))
    elif model_name in ["gemma", "gemma2", "phi3"]:
        plt.figure(figsize=(8, 8))
    elif model_name == "phi2":
        plt.figure(figsize=(8, 5))
    else:
        plt.figure(figsize=(8, 6))

    my_colors = ['white', 'green']
    my_cmap = ListedColormap(my_colors)
    bounds = [0, 1]
    my_norm = BoundaryNorm(bounds, ncolors=len(my_colors))

    annot = [['' for _ in range(results.shape[1])] for _ in range(results.shape[0])]
    cols = [5, 6, 7, 0, 4, 8]
    lines = [1, 5, 8, 7, 9]
    lines = [0, 4, 7, 6, 8]
    passes_eleph_gpt3 = [[1, 1, 1, 0, 0],
                         [1, 1, 1, 0, 1],
                         [1, 1, 1, 1, 1],
                         [1, 1, 1, 0, 0],
                         [1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0]]
    
    passes_eleph_gpt4 = [[1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1],
                         [1, 1, 1, 0, 0],
                         [1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0]]

    if model_name == "gpt3":
        eleph = passes_eleph_gpt3
    elif model_name == "gpt4":
        eleph = passes_eleph_gpt4
    if model_name in ["gpt3", "gpt4"]:
        for i in range(len(eleph)):
            for j in range(len(eleph[0])):
                col = cols[i]
                line = lines[j]
                if eleph[i][j] == 1:
                    annot[line][col] = "$\u25CF$"
                else:
                    annot[line][col] = "$\u2716$"

    if model_name not in ["gpt3", "gpt4"]:
        annot = False

    ax = sns.heatmap(results, cmap=my_cmap, norm=my_norm, vmin=0, vmax=1, annot=annot, annot_kws={'size':20, 'color': 'black'}, fmt='s', cbar=False, linecolor="black", linewidths=0.5)
    
    if model_name in only_up:
        ax.tick_params(right=False, top=True, labelright=False, labeltop=True, labelbottom=False, bottom=False, left=False, labelleft=False)
    elif model_name in ["gemma", "gemma2", "phi3"]:#["Phi-3", "Mistral"]:
        ax.tick_params(right=False, top=False, labelright=False, labeltop=False, labelbottom=False, bottom=False, left=False, labelleft=False)
    elif model_name == "phi2":
        ax.tick_params(right=False, top=False, labelright=False, labeltop=False, labelbottom=False, bottom=False, left=True, labelleft=True)
    else:
        ax.tick_params(right=False, top=True, labelright=False, labeltop=True, labelbottom=False, bottom=False, left=True, labelleft=True)

    ax.axhline(len(tests_knowledge), color='black', linewidth=7, linestyle='-')
    plt.yticks(rotation=0, fontsize=20)
    if model_name in only_up:
        plt.xticks(rotation=65, fontsize=18)
    else:
        plt.xticks(rotation=65, fontsize=18)
    plt.tight_layout()

    plt.savefig(f"plots_paper/{model_name}_conta_results.pdf", bbox_inches='tight')

    # plt.show()


def plot_heatmap_squeezed(dfs, model_names, filename, ignore_cols=[]):
    """
    results: list of dataframes with the results for this plot
    model_names: list of strings with the model names
    ignore_cols: list of columns to ignore in the results dataframes (corresponding to the datasets)
    """
    results = pd.DataFrame()
    for df in dfs:
        if len(ignore_cols) > 0:
            df.drop(columns=ignore_cols, inplace=True)
        
        res_knowledge = get_test_category(df, "knowledge")
        res_memorization = get_test_category(df, "memorization")
        res = pd.concat([res_knowledge, res_memorization], axis=0)
        res.rename(index=test_mapping, inplace=True)

        results = pd.concat([results, res], axis=1)
    
    my_colors = ['white', 'green']
    my_cmap = ListedColormap(my_colors)
    bounds = [0, 1]
    my_norm = BoundaryNorm(bounds, ncolors=len(my_colors))

    annot = [['' for _ in range(results.shape[1])] for _ in range(results.shape[0])]
    annot_gpt3 = [[1, 0, 0, 1, 1, 1, 1, -1, 0],
                  [],
                  [],
                  [],
                  [1, 0, 0, 1, 1, 1, 1, -1, 0],
                  [],
                  [-1, 0, 0, -1, -1, -1, 1, -1, 0],
                  [1, 0, 0, 1, 1, 1, 1, -1, 0],
                  [-1, 0, 0, -1, -1, 1, 1, -1, 0]
                  ]
    annot_gpt4 = [[1, 0, 0, 1, 1, 1, 1, -1, 0],
                  [],
                  [],
                  [],
                  [1, 0, 0, 1, 1, 1, 1, -1, 0],
                  [],
                  [-1, 0, 0, -1, 1, 1, 1, -1, 0],
                  [1, 0, 0, 1, 1, 1, 1, -1, 0],
                  [-1, 0, 0, -1, -1, 1, 1, -1, 0]
                  ]
    # add annot for gpt3
    for i in range(len(annot_gpt3)):
        for j in range(len(annot_gpt3[i])):
            if annot_gpt3[i][j] == 1:
                annot[i][j] = "$\u25CF$"
            elif annot_gpt3[i][j] == -1:
                annot[i][j] = "$\u2716$"
    # add annot for gpt4
    for i in range(len(annot_gpt4)):
        for j in range(len(annot_gpt4[i])):
            if annot_gpt4[i][j] == 1:
                annot[i][j+dfs[0].shape[1]] = "$\u25CF$"
            elif annot_gpt4[i][j] == -1:
                annot[i][j+dfs[0].shape[1]] = "$\u2716$"
    
    
    
    
    
    
    
    plt.figure(figsize=(6*len(model_names), 6))

    ax = sns.heatmap(results, cmap=my_cmap, norm=my_norm, vmin=0, vmax=1,
                     annot=annot, annot_kws={'size':20, 'color': 'black'},
                     fmt='s', cbar=False, linecolor="grey", linewidths=0.2,) #square=True, rasterized=True
    
    # hard horizontal line to separate knowledge and memorization tests
    ax.axhline(len(tests_knowledge), color='gray', linewidth=8, linestyle='-')

    # hard vertical line between each model
    n_cols = dfs[0].shape[1]
    for i in range(1, len(dfs)):
        ax.axvline(i * n_cols, color='black', linewidth=10, linestyle='-')
    
    ax.tick_params(right=True, left=True, bottom=True, labelright=True)

    plt.xticks(rotation=90, fontsize=25)
    plt.yticks(rotation=0, fontsize=25)
    ax.set_yticklabels(range(1, results.shape[0]+1))

    sec = ax.secondary_xaxis(location="top", rasterized=True)
    sec.set_xticks([n_cols/2 + i * n_cols for i in range(len(dfs))], labels=model_names, fontsize=35)
    sec.tick_params('x', length=0)
    
    plt.savefig(f"plots_paper/{filename}.pdf", bbox_inches='tight')

    # plt.show()



def plot_heatmap_squeezed_double(dfs, model_names, filename, ignore_cols=[]):
    """
    results: list of dataframes with the results for this plot
    model_names: list of strings with the model names
    ignore_cols: list of columns to ignore in the results dataframes (corresponding to the datasets)
    """
    results = pd.DataFrame()
    for df in dfs[:len(dfs)//2]:
        if len(ignore_cols) > 0:
            df.drop(columns=ignore_cols, inplace=True)
        
        res_knowledge = get_test_category(df, "knowledge")
        res_memorization = get_test_category(df, "memorization")
        res = pd.concat([res_knowledge, res_memorization], axis=0)
        res.rename(index=test_mapping, inplace=True)

        results = pd.concat([results, res], axis=1)

    results2 = pd.DataFrame()
    for df in dfs[len(dfs)//2:]:
        if len(ignore_cols) > 0:
            df.drop(columns=ignore_cols, inplace=True)
        
        res_knowledge = get_test_category(df, "knowledge")
        res_memorization = get_test_category(df, "memorization")
        res = pd.concat([res_knowledge, res_memorization], axis=0)
        res.rename(index=test_mapping, inplace=True)

        results2 = pd.concat([results2, res], axis=1)

    
    my_colors = ['white', 'green']
    my_cmap = ListedColormap(my_colors)
    bounds = [0, 1]
    my_norm = BoundaryNorm(bounds, ncolors=len(my_colors))

    annot = False

    plt.figure(figsize=(6*len(model_names)//2, 6*2))

    plt.subplot(211)

    # plt.figure(figsize=(6*len(model_names), 6))

    ax = sns.heatmap(results, cmap=my_cmap, norm=my_norm, vmin=0, vmax=1,
                     annot=annot, annot_kws={'size':20, 'color': 'black'},
                     fmt='s', cbar=False, linecolor="grey", linewidths=0.2,) #square=True, rasterized=True
    
    # hard horizontal line to separate knowledge and memorization tests
    ax.axhline(len(tests_knowledge), color='gray', linewidth=8, linestyle='-')

    # hard vertical line between each model
    n_cols = dfs[0].shape[1]
    for i in range(1, len(dfs)//2):
        ax.axvline(i * n_cols, color='black', linewidth=10, linestyle='-')
    
    ax.tick_params(right=True, left=True, bottom=False, labelright=True, labelbottom=False)

    plt.xticks(rotation=90, fontsize=25)
    plt.yticks(rotation=0, fontsize=25)
    ax.set_yticklabels(range(1, results.shape[0]+1))

    sec = ax.secondary_xaxis(location="top", rasterized=True)
    sec.set_xticks([n_cols/2 + i * n_cols for i in range(len(dfs)//2)], labels=model_names[:5], fontsize=35)
    sec.tick_params('x', length=0)



    plt.subplot(212)
    # plt.figure(figsize=(6*len(model_names), 6))

    ax = sns.heatmap(results2, cmap=my_cmap, norm=my_norm, vmin=0, vmax=1,
                     annot=annot, annot_kws={'size':20, 'color': 'black'},
                     fmt='s', cbar=False, linecolor="grey", linewidths=0.2,) #square=True, rasterized=True
    
    # hard horizontal line to separate knowledge and memorization tests
    ax.axhline(len(tests_knowledge), color='gray', linewidth=8, linestyle='-')

    # hard vertical line between each model
    n_cols = dfs[0].shape[1]
    for i in range(1, len(dfs)//2):
        ax.axvline(i * n_cols, color='black', linewidth=10, linestyle='-')
    
    ax.tick_params(right=True, left=True, bottom=True, labelright=True)

    plt.xticks(rotation=90, fontsize=25)
    plt.yticks(rotation=0, fontsize=25)
    ax.set_yticklabels(range(1, results2.shape[0]+1))

    sec = ax.secondary_xaxis(location="top", rasterized=True)
    sec.set_xticks([n_cols/2 + i * n_cols for i in range(len(dfs)//2)], labels=model_names[5:], fontsize=35)
    sec.tick_params('x', length=0)



    
    plt.savefig(f"plots_paper/{filename}.pdf", bbox_inches='tight')

    # plt.show()










def plot_serialization(results, serialization, model_name):
    """
    Plot the heatmap for the serialization results

    Parameters:
        - results (pd.DataFrame): the results of all the serialization tests
        - serialization (int): the serialization number
        - model_name (str): the name of the model
    """
    results = results[results["serialization"] == serialization]
    results.rename(index=test_mapping, inplace=True)
    results.drop(columns=["serialization"], inplace=True)

    if serialization != 0:
        plt.figure(figsize=(5, 6))
    else:
        plt.figure(figsize=(8, 6))

    my_colors = ['white', 'green']
    my_cmap = ListedColormap(my_colors)
    bounds = [0, 1]
    my_norm = BoundaryNorm(bounds, ncolors=len(my_colors))

    ax = sns.heatmap(results, cmap=my_cmap, norm=my_norm, vmin=0, vmax=1, cbar=False, linecolor="black", linewidths=0.5)
    
    if serialization == 0:
        ax.tick_params(right=False, top=True, labelright=False, labeltop=True, labelbottom=False, bottom=False, left=True, labelleft=True)
    else:
        ax.tick_params(right=False, top=True, labelright=False, labeltop=True, labelbottom=False, bottom=False, left=False, labelleft=False)

    plt.yticks(rotation=0, fontsize=20)
    plt.xticks(rotation=65, fontsize=18)
    plt.tight_layout()

    plt.savefig(f"plots_serialization/{model_name}_{serialization}.pdf", bbox_inches='tight')

    # plt.show()


def plot_heatmap_diff(old, new):
    df_old = load_results(old)
    df_new = load_results(new)

    results_knowledge_old = get_test_category(df_old, "knowledge")
    results_memorization_old = get_test_category(df_old, "memorization")
    results_old = pd.concat([results_knowledge_old, results_memorization_old])
    results_old.rename(index=test_mapping, inplace=True)

    results_knowledge_new = get_test_category(df_new, "knowledge")
    results_memorization_new = get_test_category(df_new, "memorization")
    results_new = pd.concat([results_knowledge_new, results_memorization_new])
    results_new.rename(index=test_mapping, inplace=True)

    old_data = results_old.to_numpy()
    new_data = results_new.to_numpy()

    only_new = new_data - old_data
    both = old_data + new_data

    full = np.zeros((len(both), len(both[0])))
    for i in range(len(both)):
        for j in range(len(both[0])):
            full[i,j] = only_new[i, j]
            if both[i, j] == 2:
                full[i, j] = 2

    full = pd.DataFrame(data=full,
                        index=results_new.index,
                        columns=results_new.columns)

    full_legend = ["GPT-3.5", "GPT-4", "Llama 2", "Llama 3", "T0", "GPT-J"]


    plt.figure(figsize=(8, 6))

    my_colors = ['grey', 'white', 'limegreen', 'darkgreen', 'darkgreen']
    my_cmap = ListedColormap(my_colors)
    bounds = [-1, 0, 1, 2, 3]
    my_norm = BoundaryNorm(bounds, ncolors=len(my_colors))

    ax = sns.heatmap(full, cmap=my_cmap, norm=my_norm, vmin=-1, vmax=2, annot=False, cbar=False, linecolor="black", linewidths=0.5)
    
    ax.tick_params(right=False, top=True, labelright=False, labeltop=True, labelbottom=False, bottom=False)
    ax.axhline(len(tests_knowledge), color='black', linewidth=7, linestyle='-')
    plt.yticks(rotation=0, fontsize=20)
    plt.xticks(rotation=65, fontsize=18)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"plots/{old}_{new}_conta_results.pdf", bbox_inches='tight')


def build_table_acc_membership(details=True, ignore_cols=[], ignore_llms=[], show_mean=False):
    """
    Creates the latex table for the paper
    """

    scores_0 = {}
    scores_1 = {}
    scores_mem = {}


    df = pd.read_csv("contamination_results/acc_membership_results.csv")
    res = "\\begin[tabular][cc|".replace("[", "{").replace("]", "}")
    for i in range(len(df.columns) - len(ignore_cols) - 2):
        res += "|c"
    if show_mean:
        res += "||c"
    res += "}\n"
    res += "LLM &"
    for col in df.columns[2:]:
        if col not in ignore_cols:
            res += f" & {col}"
    if show_mean:
        res += " & \\textbf[Mean]".replace("[", "{").replace("]", "}")
    res += " \\\\ \n\hline\n"
    for i in range(len(df)):
        sample = df.iloc[i]
        mean_value = 0.0
        model = sample["LLM"]
        if model not in ignore_llms:
            if i % 3 == 0:
                res += "\hline\n"
                res += f"\multicolumn[1][c|][\multirow[3][*][{model}]]".replace("[", "{").replace("]", "}")
            else:
                res += "\multicolumn[1][c|][]".replace("[", "{").replace("]", "}")
            cat = sample["type"]
            res += f" & {cat}"
            for col in df.columns[2:]:
                if show_mean:
                    value = sample[col]
                    score = float(value.split(" ")[0])
                    mean_value += score
                if col not in ignore_cols:
                    value = sample[col]
                    if not details:
                        value = value.split(" ")[0]
                        if value == "0.0" or value == "0":
                            value = "0.00"
                        elif value == "1.0" or value == "1":
                            value = "1.00"
                        elif value == "0.5":
                            value = "0.50"
                    res += f" & {value}"
            if show_mean:
                mean_value /= len(df.columns) - len(ignore_cols)
                res += f" & {mean_value:.2f}"
                if cat == "0-shot":
                    scores_0[model] = mean_value
                elif cat == "1-shot":
                    scores_1[model] = mean_value
                elif cat == "member.":
                    scores_mem[model] = mean_value
            res += " \\\\ \n"
    res += "\end{tabular}"
    print(res)
    return scores_0, scores_1, scores_mem


def build_ranking(scores_0, scores_1, scores_mem, ignore_llms=[], ignore_cols=[]):
    """
    Build the latex table with the results for 0-shot, 1-shot and membership with the contamination results
    """
    llms_short = ["llama2", "llama3", "gemma", "t0", "gptj", "phi2", "phi3", "mistral", "gpt3", "gpt4", "gemini1.0", "gemini1.5", "llama3.1", "gpt4o", "gemma2"]
    llms = ["Llama 2", "Llama 3", "Gemma", "T0", "GPT-J", "Phi 2", "Phi 3", "Mistral", "GPT-3.5", "GPT-4", "Gemini 1.0", "Gemini 1.5", "Llama 3.1", "GPT-4o", "Gemma 2"]


    conta_results = {}
    for i, llm in enumerate(llms_short):
        df = pd.read_csv(f"contamination_results/{llm}.csv", index_col=0)
        df.drop(columns=ignore_cols, inplace=True)
        df.drop(index=["membership", "incomplete_completion"], inplace=True)
        total = 0
        for j in range(len(df)):
            total += df.iloc[j].sum()
        conta_results[llms[i]] = total    

    n_conta_tests = len(df) * len(df.columns)

    conta_results = {k: v for k, v in sorted(conta_results.items(), key=lambda item: item[1], reverse=True)}
    for k, v in conta_results.items():
        conta_results[k] = conta_results[k] / n_conta_tests
    
    scores_0 = {k: v for k, v in sorted(scores_0.items(), key=lambda item: item[1], reverse=True)}
    scores_1 = {k: v for k, v in sorted(scores_1.items(), key=lambda item: item[1], reverse=True)}
    scores_mem = {k: v for k, v in sorted(scores_mem.items(), key=lambda item: item[1], reverse=True)}

    res = ""
    res += "\\begin{tabular}{c|"
    for _ in range(len(scores_0) - len(ignore_llms)):
        res += "c"
    res += "}\n"
    for k in conta_results.keys():
        if k not in ignore_llms:
            res += f" & {k}"
    res += " \\\\ \n\hline\n\hline\n"
    all_scores = {"Contamination": conta_results, "Membership": scores_mem, "0-shot": scores_0, "1-shot": scores_1}
    for cat in all_scores.keys():
        res += cat
        for k in conta_results.keys():
            if k not in ignore_llms:
                scores = all_scores[cat]
                val = scores[k]
                if val == max(scores.values()):
                    res += f" & \\textbf[{val:.2f}]".replace("[", "{").replace("]", "}")
                else:
                    res += f" & {val:.2f}"
        res += " \\\\ \n"


    res += "\\end{tabular}"
    print(res)




def merge_recent(llm_name):
    df = pd.read_csv(f"contamination_results/{llm_name}_old.csv", index_col=0)
    df_recent = pd.read_csv(f"contamination_results/{llm_name}_recent.csv", index_col=0)

    res = pd.concat([df, df_recent], axis=1)
    return res






if __name__ == "__main__":

    llms = ["llama2", "llama3", "gemma", "t0", "gptj", "phi2", "phi3", "mistral", "gpt3", "gpt4", "gemini1.0", "gemini1.5", "llama3.1", "gpt4o", "gemma2"]

    llms_short = ["llama2", "llama3", "gemma", "t0", "gptj", "phi2", "phi3", "mistral", "gpt3", "gpt4", "gemini1.0", "gemini1.5", "llama3.1", "gpt4o", "gemma2"]
    llms = ["Llama 2", "Llama 3", "Gemma", "T0", "GPT-J", "Phi 2", "Phi 3", "Mistral", "GPT-3.5", "GPT-4", "Gemini 1.0", "Gemini 1.5", "Llama 3.1", "GPT-4o", "Gemma 2"]



    # dfs = []
    # llms = ["gpt3", "gpt4", "gpt4o", "gemini1.0", "gemini1.5"]
    # llms = ["llama2", "llama3", "llama3.1", "gemma", "gemma2", "mistral", "phi2", "phi3", "gptj", "t0"]
    # # llms = ["t0", "gptj"]
    # for model in llms:
    #     dfs.append(pd.read_csv(f"contamination_results/{model}.csv", index_col=0))

    # plot_heatmap_squeezed(dfs, ["GPT-3.5", "GPT-4", "GPT-4o", "Gemini 1.0", "Gemini 1.5"], "closed_source", ignore_cols=["Berkeley", "MathE"])
    # plot_heatmap_squeezed(dfs, ["GPT-3.5", "GPT-4", "GPT-4o", "Gemini 1.0", "Gemini 1.5"], "closed_source.pdf", ignore_cols=["Berkeley", "MathE"])
    # plot_heatmap_squeezed_double(dfs, ["Llama 2", "Llama 3", "Llama 3.1", "Gemma", "Gemma 2", "Mistral", "Phi 2", "Phi 3", "GPT-J", "T0"], "open_source_llama3.2", ignore_cols=["Berkeley", "MathE"])


    # for model_name in llms:
    #     # df = merge_recent(model_name)
    #     # df.to_csv(f"contamination_results/{model_name}.csv", index=True)

    #     df = pd.read_csv(f"contamination_results/{model_name}.csv", index_col=0)
    #     plot_heatmap(df, model_name, ignore_cols=["Berkeley", "MathE"])




    # ### BUILD THE TABLE WITH ALL THE RESULTS ###

    scores_0, scores_1, scores_mem = build_table_acc_membership(ignore_llms=[],
                               ignore_cols=["Berkeley", "MathE"],
                               details=False,
                               show_mean=True)
    
    # get the new membership scores
    scores_mem = {}
    for i, name in enumerate(llms_short):
        df = pd.read_csv(f"contamination_results/{name}_membership.csv", index_col=0)
        score = df["value"].mean()/100
        scores_mem[llms[i]] = score

    scores_0 = {k: v for k, v in sorted(scores_0.items(), key=lambda item: item[1], reverse=True)}
    scores_1 = {k: v for k, v in sorted(scores_1.items(), key=lambda item: item[1], reverse=True)}
    scores_mem = {k: v for k, v in sorted(scores_mem.items(), key=lambda item: item[1], reverse=True)}

    print()
    print()
    print("scores 0-shot:", scores_0.keys())
    print("scores 1-shot:", scores_1.keys())
    print("scores member:", scores_mem.keys())



    # ### BUILD THE TABLE WITH THE MEAN RESULTS FROM THE MAIN PAPER ###

    build_ranking(scores_0, scores_1, scores_mem,
                  ignore_llms=[], ignore_cols=["Berkeley", "MathE"])



    # ### PLOT THE HEATMAPS FOR THE SERIALIZATION COMPARISON ###

    # for model_name in ["gemma2", "gpt4o", "llama3", "mistral"]:
    #     for i in range(5):
    #         df = pd.read_csv(f"contamination_results/serializations/{model_name}_serialization.csv", index_col=0)
    #         plot_serialization(df, i, model_name)



    # ### PLOT THE HEATMAPS FOR THE CONTAMINATION RESULTS ###

    # llms = ["llama2", "llama3", "gemma", "t0", "gptj", "phi2", "phi3", "mistral", "gpt3", "gpt4", "gemini1.0", "gemini1.5", "llama3.1", "gpt4o", "gemma2"]

    # for model in llms:
    #     df = pd.read_csv(f"contamination_results/{model}.csv", index_col=0)
    #     plot_heatmap(df, model)





