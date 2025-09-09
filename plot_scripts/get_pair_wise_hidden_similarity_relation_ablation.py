import os
import sys
import re
import random
import pickle

import torch
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(".")
sys.path.append("../")
from results_explore_fewshot import load_file

model_languages = {
    "bloom": ["ar", "ca", "en", "es", "fr", "vi", "zh"],
    "llama2": ["ca", "en", "es", "fr", "hu", "ja", "ko", "nl", "ru", "uk", "vi", "zh"],
}
model_num_layers = {"bloom": 24 * 2 + 1, "llama2": 32 * 2 + 1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama2")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="llm-transparency-tool-bak/backend_results_<MODEL>_fewshot_demo_new",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_dev/hidden_consistency_sets_<MODEL>_fewshot_demo_new",
    )
    parser.add_argument("--similarity_type", type=str, default="cosine_similarity")
    parser.add_argument("--select_top_k_neurons", type=int, default=-1)
    parser.add_argument("--set_neurons_threshold", type=float, default=-1)
    parser.add_argument("--binary", type=bool, default=False)
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--plot_figures", type=bool, default=False)
    args = parser.parse_args()

    model_name = args.model_name
    results_dir = args.results_dir.replace("<MODEL>", model_name)
    output_dir = args.output_dir.replace("<MODEL>", model_name)
    similarity_type = args.similarity_type
    output_dir += "_" + args.similarity_type.split("_")[0]
    select_top_k_neurons = args.select_top_k_neurons
    set_neurons_threshold = args.set_neurons_threshold
    binary = args.binary
    language_list = sorted(model_languages[model_name])
    num_layers = model_num_layers[model_name]
    os.makedirs(output_dir, exist_ok=True)

    ### Section 1 - Get the correctness statistics for each sample ###

    # Initialize a dictionary to collect the data
    data = {
        "relation": [],
        "sample_index": [],
    }

    # Collect all languages as columns
    for lang in language_list:
        data[lang] = []

    # Initialize sample index dictionary for tracking unique samples
    sample_status = {}
    # Traverse each relation_language folder in th
    # e base directory
    for folder in sorted(os.listdir(results_dir)):
        folder_path = os.path.join(results_dir, folder)

        # Skip if it's not a directory or not in the format relation_language
        if (
            not os.path.isdir(folder_path)
            or "_" not in folder
            or folder == "neuron_contributions_tensor"
            or folder == "plots_all_relations"
        ):
            continue

        # Extract relation and language
        relation, language = folder.rsplit("_", 1)

        # Ensure the language column is initialized
        if language not in data:
            data[language] = []

        # Check for correct and incorrect folders
        for status in ["correct", "incorrect", "other"]:
            status_folder = os.path.join(folder_path, status)

            if not os.path.isdir(status_folder):
                continue  # Skip if the status folder does not exist

            # Traverse each file in the correct/incorrect subfolder
            for file in os.listdir(status_folder):
                if file.endswith("_high_rank.json"):
                    continue
                # Extract the sample index from the file name using a regular expression
                # match = re.search(r"_(\d+)", file)
                match = re.search(r"_(\d+)\.json$", file)
                if match:
                    sample_index = int(match.group(1))

                    # Initialize sample index if not already in sample_status
                    if sample_index not in sample_status:
                        sample_status[sample_index] = {
                            lang: 0 for lang in language_list
                        }
                        sample_status[sample_index]["relation"] = relation
                        sample_status[sample_index]["sample_index"] = sample_index

                    # Set the language status: 1 if correct, 0 if incorrect
                    sample_status[sample_index][language] = (
                        1 if status == "correct" else 0
                    )

    # Append each sample's data to the main data dictionary
    for sample in sample_status.values():
        for key in sample:
            data[key].append(sample[key])

    # Create a DataFrame from the data dictionary
    df = pd.DataFrame(data)

    # Sort the DataFrame by relation alphabetically, then by sample_index numerically
    df = df.sort_values(
        by=["relation", "sample_index"], ascending=[True, True]
    ).reset_index(drop=True)

    # Add a "n_languages" column with the sum of language-specific columns
    df["n_languages"] = df[language_list].sum(axis=1)

    # Save to CSV
    output_csv_path = f"{output_dir}/sample_stats.csv"
    df.to_csv(output_csv_path, index=False)
    print("Sample consistency statistics done!")
    print(f"Results saved in {output_dir}/sample_stats.csv")

    ### Section 2 - Ablation 1: Construct the relation-wise sample stats
    ### Section 2 - & do layer-wise stats and calculate unparallel similarity

    relations = df["relation"].unique().tolist()
    if model_name == "llama2" and "language_of_work_or_name_2" in relations:
        relations.remove("language_of_work_or_name_2")

    num_samples_for_each_relation = 50 if model_name == "bloom" else 25
    sample_stats = {
        relation: df[df["relation"] == relation]["sample_index"].tolist()
        for relation in relations
    }

    # Specify layer_names
    layer_names = [
        f"{idx//2}_pre" if idx % 2 == 0 else f"{idx//2}_mid"
        for idx in range(num_layers - 1)
    ]
    layer_names.append("final_post")

    # Specify langauge pairs (English as pivot)
    language_pairs_en = []
    for i in range(len(language_list)):
        for j in range(i + 1, len(language_list)):
            pair = f"{language_list[i]}_{language_list[j]}"
            if "en" in pair:
                language_pairs_en.append(pair)

    # Initialize dictionaries
    similarity_intra_relation_layers = (
        {}
    )  # Same relation, similarity between different samples

    # if os.path.exists(f'{output_dir}/language_pair_stats_ablation.pkl'):
    #     with open(f'{output_dir}/language_pair_stats_ablation.pkl', "rb") as f:
    #         similarity_intra_relation_layers = pickle.load(f)
    if 1 == 0:
        pass
    else:
        for layer in tqdm(range(num_layers)):
            print(f"Processing layer - {layer_names[layer]}")
            hidden_states = {relation: {} for relation in relations}
            hidden_states = {
                relation: {lang: {} for lang in language_list} for relation in relations
            }

            print("Loading sample hidden states...")
            for relation, sample_list in sample_stats.items():
                sample_list = random.sample(
                    sample_list, min(num_samples_for_each_relation, len(sample_list))
                )
                for sample_idx in tqdm(sample_list):
                    for lang in language_list:
                        source_filename = f"{results_dir}/{relation}_{lang}/hidden_states/{relation}_{lang}_{sample_idx}.pkl"
                        data = load_file(source_filename)
                        data = torch.stack([v.cpu() for v in data.values()])[layer]
                        hidden_states[relation][lang][sample_idx] = data

            if (
                "language_of_work_or_name_1" in hidden_states
                and "language_of_work_or_name_2" in hidden_states
            ):
                hidden_states["language_of_work_or_name"] = {}

                # Merge the keys
                for key in hidden_states["language_of_work_or_name_1"].keys():
                    hidden_states["language_of_work_or_name"][key] = {
                        **hidden_states["language_of_work_or_name_1"].get(key, {}),
                        **hidden_states["language_of_work_or_name_2"].get(key, {}),
                    }

                # Remove the old keys
                del hidden_states["language_of_work_or_name_1"]
                del hidden_states["language_of_work_or_name_2"]

            relations_new = list(hidden_states.keys())
            # Ablation 1 - Calculate similarity with each relation, with different samples
            print("Calculating pair-wise similarities...")
            for relation in relations_new:
                # English as the "pivot" language
                sample_indices = list(hidden_states[relation]["en"].keys())
                if relation not in similarity_intra_relation_layers:
                    similarity_intra_relation_layers[relation] = {
                        pair: {layer: [] for layer in range(num_layers)}
                        for pair in language_pairs_en
                    }

                for sample_idx_en, layer_vector_en in tqdm(
                    hidden_states[relation]["en"].items()
                ):
                    for language in language_list:
                        if language == "en":
                            continue
                        sample_idx_xx = random.choice(
                            [idx for idx in sample_indices if idx != sample_idx_en]
                        )
                        layer_vector_xx = hidden_states[relation][language][
                            sample_idx_xx
                        ]
                        layer_similarity = F.cosine_similarity(
                            layer_vector_en.unsqueeze(0), layer_vector_xx.unsqueeze(0)
                        ).item()
                        if (
                            f"{language}_en"
                            in similarity_intra_relation_layers[relation].keys()
                        ):
                            similarity_intra_relation_layers[relation][
                                f"{language}_en"
                            ][layer].append(layer_similarity)
                        else:
                            similarity_intra_relation_layers[relation][
                                f"en_{language}"
                            ][layer].append(layer_similarity)

                for pair in similarity_intra_relation_layers[relation].keys():
                    average_similarity = sum(
                        similarity_intra_relation_layers[relation][pair][layer]
                    ) / len(similarity_intra_relation_layers[relation][pair][layer])
                    similarity_intra_relation_layers[relation][pair][layer] = round(
                        average_similarity, 4
                    )

        for relation, pair_wise_data in similarity_intra_relation_layers.items():
            for pair in pair_wise_data.keys():
                similarity_intra_relation_layers[relation][pair] = list(
                    similarity_intra_relation_layers[relation][pair].values()
                )

        with open(f"{output_dir}/language_pair_stats_ablation.pkl", "wb") as f:
            pickle.dump(similarity_intra_relation_layers, f)

    print(similarity_intra_relation_layers.keys())

    ### Section 3 - Ablation 2: Calculate similarities across different relations ###
    similarity_inter_relation_layers = {
        pair: {layer: [] for layer in range(num_layers)} for pair in language_pairs_en
    }  # Similarity between samples from different relations

    # if os.path.exists(f'{output_dir}/language_pair_stats_ablation_2.pkl'):
    #     with open(f'{output_dir}/language_pair_stats_ablation_2.pkl', "rb") as f:
    #         similarity_inter_relation_layers = pickle.load(f)
    if 1 == 0:
        pass
    else:
        for layer in tqdm(range(num_layers)):
            print(f"Processing layer - {layer_names[layer]}")
            hidden_states = {relation: {} for relation in relations}
            hidden_states = {
                relation: {lang: {} for lang in language_list} for relation in relations
            }

            print("Loading sample hidden states...")
            for relation, sample_list in sample_stats.items():
                sample_list = random.sample(
                    sample_list, min(num_samples_for_each_relation, len(sample_list))
                )
                for sample_idx in tqdm(sample_list):
                    for lang in language_list:
                        source_filename = f"{results_dir}/{relation}_{lang}/hidden_states/{relation}_{lang}_{sample_idx}.pkl"
                        data = load_file(source_filename)
                        data = torch.stack([v.cpu() for v in data.values()])[layer]
                        hidden_states[relation][lang][sample_idx] = data

            if (
                "language_of_work_or_name_1" in hidden_states
                and "language_of_work_or_name_2" in hidden_states
            ):
                hidden_states["language_of_work_or_name"] = {}

                # Merge the keys
                for key in hidden_states["language_of_work_or_name_1"].keys():
                    hidden_states["language_of_work_or_name"][key] = {
                        **hidden_states["language_of_work_or_name_1"].get(key, {}),
                        **hidden_states["language_of_work_or_name_2"].get(key, {}),
                    }

                # Remove the old keys
                del hidden_states["language_of_work_or_name_1"]
                del hidden_states["language_of_work_or_name_2"]

            relations_new = list(hidden_states.keys())
            # Ablation 2 - Calculate similarity with different relations
            print("Calculating pair-wise similarities...")
            for relation_en in relations_new:
                # English as the "pivot" language
                sample_indices = list(hidden_states[relation_en]["en"].keys())

                for sample_idx_en, layer_vector_en in tqdm(
                    hidden_states[relation_en]["en"].items()
                ):
                    for language in language_list:
                        if language == "en":
                            continue
                        relation_xx = random.choice(
                            [r for r in relations_new if r != relation_en]
                        )
                        sample_indices_xx = list(
                            hidden_states[relation_xx][language].keys()
                        )
                        sample_idx_xx = random.choice(
                            [idx for idx in sample_indices_xx]
                        )

                        layer_vector_xx = hidden_states[relation_xx][language][
                            sample_idx_xx
                        ]
                        layer_similarity = F.cosine_similarity(
                            layer_vector_en.unsqueeze(0), layer_vector_xx.unsqueeze(0)
                        ).item()
                        if f"{language}_en" in similarity_inter_relation_layers.keys():
                            similarity_inter_relation_layers[f"{language}_en"][
                                layer
                            ].append(layer_similarity)
                        else:
                            similarity_inter_relation_layers[f"en_{language}"][
                                layer
                            ].append(layer_similarity)

            for pair in similarity_inter_relation_layers.keys():
                average_similarity = sum(
                    similarity_inter_relation_layers[pair][layer]
                ) / len(similarity_inter_relation_layers[pair][layer])
                similarity_inter_relation_layers[pair][layer] = round(
                    average_similarity, 4
                )

        with open(f"{output_dir}/language_pair_stats_ablation_2.pkl", "wb") as f:
            pickle.dump(similarity_inter_relation_layers, f)

    ### Section 4 - Ablation 1: Generate the pair-wise plots ###

    # First merge all relations together
    # Initialize a dictionary to store the merged data
    merged_all_relations = {}
    similarity_intra_relation_layers["all_relations"] = {}
    # Iterate through all relations in the data
    for relation, subkeys in similarity_intra_relation_layers.items():
        for subkey, values in subkeys.items():
            # Ensure the subkey exists in the merged_data
            if subkey not in merged_all_relations:
                merged_all_relations[subkey] = []

            # Append the current values to the subkey
            merged_all_relations[subkey].append(values)

    # Calculate the average value for each subkey across all relations
    for subkey, values_list in merged_all_relations.items():
        # Stack the values and calculate the mean along the first axis
        similarity_intra_relation_layers["all_relations"][subkey] = np.mean(
            np.stack(values_list), axis=0
        )

    # Merge all language pairs together
    all_languages_values = []

    # Iterate through all language pairs in "all_relations"
    for subkey, values in similarity_intra_relation_layers["all_relations"].items():
        if subkey != "all_languages":  # Exclude any existing "all_languages" key
            all_languages_values.append(values)

    # Calculate the average values across all language pairs
    similarity_intra_relation_layers["all_relations"]["all_languages"] = np.mean(
        np.stack(all_languages_values), axis=0
    )

    for relation, similarity_data in similarity_intra_relation_layers.items():
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette("hls", len(similarity_data))
        for i, pair in enumerate(similarity_data.keys()):

            layers = np.arange(len(similarity_data[pair]))
            plt.plot(layers, similarity_data[pair], label=f"{pair}", color=colors[i])

        # Add labels and title
        plt.xlabel("Layers", fontsize=12)
        plt.ylabel(similarity_type, fontsize=12)
        plt.title(
            f"Ablation 1 - Hidden states {similarity_type} (same relation, different samples)\nAll relations - *Language pair: {pair}*",
            fontsize=14,
        )
        plt.tight_layout()

        # Add grid and legend
        plt.grid(alpha=0.3)
        plt.legend(loc="best")

        plot_path = f"{output_dir}/plots/ablation1"
        os.makedirs(plot_path, exist_ok=True)
        plt.savefig(
            f"{plot_path}/similarity_ablation1_{relation}_all_english_pairs.png"
        )
        plt.close()

    ### Section 5 - Ablation 2: Generate the pair-wise plots ###

    # Merge all language pairs together
    # Iterate through all language pairs in "all_relations"
    try:
        all_languages_values = []
        for subkey, values in similarity_inter_relation_layers.items():
            if subkey != "all_languages":  # Exclude any existing "all_languages" key
                all_languages_values.append(values)

        # Calculate the average values across all language pairs
        similarity_inter_relation_layers["all_languages"] = np.mean(
            np.stack(all_languages_values), axis=0
        )
    except:
        all_languages_values = []
        for pair, pair_wise_data in similarity_inter_relation_layers.items():
            similarity_inter_relation_layers[pair] = list(
                similarity_inter_relation_layers[pair].values()
            )
        for subkey, values in similarity_inter_relation_layers.items():
            if subkey != "all_languages":  # Exclude any existing "all_languages" key
                all_languages_values.append(values)

        # Calculate the average values across all language pairs
        similarity_inter_relation_layers["all_languages"] = np.mean(
            np.stack(all_languages_values), axis=0
        )

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("hls", len(similarity_inter_relation_layers))
    for i, pair in enumerate(similarity_inter_relation_layers.keys()):

        layers = np.arange(len(similarity_inter_relation_layers[pair]))
        plt.plot(
            layers,
            similarity_inter_relation_layers[pair],
            label=f"{pair}",
            color=colors[i],
        )

    x_labels = [
        f"{idx//2}_pre" if idx % 2 == 0 else f"{idx//2}_mid"
        for idx in range(len(layers) - 1)
    ]
    x_labels.append("final_post")

    # Add labels and title
    plt.xlabel("Layers", fontsize=12)
    plt.ylabel(similarity_type, fontsize=12)
    plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=45, fontsize=8)
    plt.title(
        f"Ablation 2 - Hidden states {similarity_type} (different relations)\nAll relations - *Language pair: {pair}*",
        fontsize=14,
    )
    plt.tight_layout()

    # Add grid and legend
    plt.grid(alpha=0.3)
    plt.legend(loc="best")

    plot_path = f"{output_dir}/plots/ablation2"
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(f"{plot_path}/similarity_ablation2_all_english_pairs.png")
    plt.close()
