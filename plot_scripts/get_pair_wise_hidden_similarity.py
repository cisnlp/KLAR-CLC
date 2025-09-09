import os
import sys
import re
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
    parser.add_argument("--model_name", type=str, default="bloom")
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
    model_print_name = "LLaMA2" if model_name == "llama2" else "BLOOM"
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

    ### Section 2 - Structure the result in the dictionary ###
    """
    The new dictionary should be like:
    language_pair_stats = {
        "relation": {
            "lang1_lang2": {
                "correct_both": {
                    "sample_idx: [],
                    "hidden_similarity": [],
                    },
                "correct_1": {
                    "sample_idx: [],
                    "hidden_similarity": [],
                    },
                "correct_2": {
                    "sample_idx: [],
                    "hidden_similarity": [],
                    },
                "incorrect_both": {
                    "sample_idx: [],
                    "hidden_similarity": [],
                    },
                "all": {
                    "sample_idx: [],
                    "hidden_similarity": [],
                    },
            },
            # ... other relations
        },
            # ... other language pairs
    }
    """
    # Initialize the language_pair_stats dictionary

    # Generate all unique pairs of languages
    language_pairs = []
    for i in range(len(language_list)):
        for j in range(i + 1, len(language_list)):
            pair = f"{language_list[i]}_{language_list[j]}"
            language_pairs.append(pair)

    if (
        os.path.exists(f"{output_dir}/language_pair_stats.pkl")
        and select_top_k_neurons == -1
        and set_neurons_threshold == -1
    ):
        with open(f"{output_dir}/language_pair_stats.pkl", "rb") as f:
            language_pair_stats = pickle.load(f)
    # if 1 == 0:
    #     pass
    else:
        language_pair_stats = {}
        language_pair_stats_all = {
            pair: {
                "correct_both": {
                    "sample_idx": [],
                    "hidden_similarity": [],
                },
                "correct_1": {
                    "sample_idx": [],
                    "hidden_similarity": [],
                },
                "correct_2": {
                    "sample_idx": [],
                    "hidden_similarity": [],
                },
                "incorrect_both": {
                    "sample_idx": [],
                    "hidden_similarity": [],
                },
                "all": {
                    "sample_idx": [],
                    "hidden_similarity": [],
                },
            }
            for pair in language_pairs
        }
        # Iterate over each row in the DataFrame
        for _, row in tqdm(df.iterrows(), total=len(df)):
            # Get the sample index and relation
            sample_idx = row["sample_index"]
            relation = row["relation"]

            if relation not in language_pair_stats:
                language_pair_stats[relation] = {
                    pair: {
                        "correct_both": {
                            "sample_idx": [],
                            "hidden_similarity": [],
                        },
                        "correct_1": {
                            "sample_idx": [],
                            "hidden_similarity": [],
                        },
                        "correct_2": {
                            "sample_idx": [],
                            "hidden_similarity": [],
                        },
                        "incorrect_both": {
                            "sample_idx": [],
                            "hidden_similarity": [],
                        },
                        "all": {
                            "sample_idx": [],
                            "hidden_similarity": [],
                        },
                    }
                    for pair in language_pairs
                }

            language_correctness = {lang: row[lang] for lang in language_list}
            hidden_states = {lang: None for lang in language_list}
            for lang in language_list:
                source_filename = f"{results_dir}/{relation}_{lang}/hidden_states/{relation}_{lang}_{sample_idx}.pkl"
                data = load_file(source_filename)
                data = {key: tensor.cpu() for key, tensor in data.items()}
                hidden_states[lang] = torch.stack(list(data.values()))

            hidden_states_all = torch.stack(
                [hidden_states[lang] for lang in language_list], dim=0
            )

            similarity_stats_layers = {
                pair: [None for _ in range(num_layers)] for pair in language_pairs
            }

            for layer in range(num_layers):
                layer_data = hidden_states_all[:, layer, :]

                n_languages = len(language_list)
                # Initialize an empty similarity matrix
                similarity_matrix = np.ones((n_languages, n_languages))

                # Calculate similaritys between each pair
                for i in range(n_languages):
                    for j in range(
                        i + 1, n_languages
                    ):  # Compute only for upper triangle (i <= j)
                        # Calculate similarity
                        if similarity_type == "cosine_similarity":
                            similarity = F.cosine_similarity(
                                layer_data[i].unsqueeze(0), layer_data[j].unsqueeze(0)
                            ).item()
                        elif similarity_type == "euclidean_distance":
                            similarity = torch.dist(layer_data[i], layer_data[j], p=2)
                            similarity = similarity / (
                                torch.sqrt(
                                    layer_data[i].norm(p=2) * layer_data[j].norm(p=2)
                                )
                                + 1e-6
                            )
                            similarity = similarity.item()
                        similarity_matrix[i, j] = round(
                            similarity, 4
                        )  # Assign to (i, j)
                        similarity_matrix[j, i] = round(
                            similarity, 4
                        )  # Mirror to (j, i)

                        lang1 = language_list[i]
                        lang2 = language_list[j]
                        pair = f"{lang1}_{lang2}"
                        similarity_stats_layers[pair][layer] = similarity_matrix[i, j]

            for i in range(n_languages):
                for j in range(
                    i + 1, n_languages
                ):  # Compute only for upper triangle (i <= j)
                    lang1 = language_list[i]
                    lang2 = language_list[j]
                    pair = f"{lang1}_{lang2}"

                    if (
                        language_correctness[lang1] == 1
                        and language_correctness[lang2] == 1
                    ):
                        active_key = "correct_both"
                    elif language_correctness[lang1] == 1:
                        active_key = "correct_1"
                    elif language_correctness[lang2] == 1:
                        active_key = "correct_2"
                    else:
                        active_key = "incorrect_both"

                    language_pair_stats[relation][pair][active_key][
                        "sample_idx"
                    ].append(sample_idx)
                    language_pair_stats[relation][pair]["all"]["sample_idx"].append(
                        sample_idx
                    )

                    language_pair_stats[relation][pair][active_key][
                        "hidden_similarity"
                    ].append(similarity_stats_layers[pair])
                    language_pair_stats[relation][pair]["all"][
                        "hidden_similarity"
                    ].append(similarity_stats_layers[pair])

        print("language_pair_stats ready!")
        stats_filepath = f"{output_dir}/language_pair_stats.pkl"
        if select_top_k_neurons != -1:
            stats_filepath = (
                stats_filepath.replace(".pkl", f"_top{select_top_k_neurons}_binary.pkl")
                if binary
                else stats_filepath.replace(".pkl", f"_top{select_top_k_neurons}.pkl")
            )
        if set_neurons_threshold != -1:
            stats_filepath = (
                stats_filepath.replace(
                    ".pkl", f"_ths{set_neurons_threshold}_binary.pkl"
                )
                if binary
                else stats_filepath.replace(
                    ".pkl", f"_ths{set_neurons_threshold}_binary.pkl"
                )
            )

        with open(stats_filepath, "wb") as f:
            pickle.dump(language_pair_stats, f)

    ### Section 3 - Generate the pair-wise plots ###
    colors = {
        "correct_both": "tab:green",
        "correct_1": "tab:blue",
        "correct_2": "tab:organge",
        "incorrect_both": "tab:red",
        "inconsistent": "tab:red",
        "all": "tab:gray",
    }
    colors_ablation = {
        "parallel": "#4a8ee1",
        # "ablation 1": "#40997b",
        # "ablation 1": "#FFDE7D",
        "ablation 1": "#FF6F3C",
        # "ablation 2": "tab:orange",
        "ablation 2": "#F6416C",
        # "ablation 2": "#FF9A3C",
        # "ablation 2": "#d3687c",
    }
    font_sizes = {
        "title": 28,
        "ticks": 18,
        "label": 26,
        "legend": 18,
    }
    keys_to_display = [
        "correct_both",
        "inconsistent",
        "all",
    ]  # or ["correct_both", "correct_1", "correct_2", "incorrect_both", "all"]
    hidden_similarity_avg_all = {}
    hidden_similarity_avg_inconsistent_all = {}

    def process_legends(legend_labels):
        # Preprocess labels to be in the format <en, lang>
        formatted_labels = []
        for label in legend_labels:
            if "_en" in label:
                lang1, lang2 = label.split("_")
                formatted_labels.append(f"{lang2}-{lang1}")
            elif "en_" in label:
                lang1, lang2 = label.split("_")
                formatted_labels.append(f"{lang1}-{lang2}")
            else:
                formatted_labels.append(label)

        return formatted_labels

    for relation in language_pair_stats.keys():
        print(f"Plotting relation: {relation}")
        for pair in tqdm(language_pair_stats[relation].keys()):
            hidden_similarity_avg_all[pair] = {key: [] for key in keys_to_display}
            hidden_similarity_avg_inconsistent_all[pair] = []

            plt.figure(figsize=(10, 6))
            plot_sets = list(language_pair_stats[relation][pair].keys())
            y_min = 0
            for i, plot_key in enumerate(plot_sets):
                if (
                    "hidden_similarity_avg"
                    not in language_pair_stats[relation][pair][plot_key]
                ):
                    hidden_similarity = language_pair_stats[relation][pair][plot_key][
                        "hidden_similarity"
                    ]
                    num_samples = len(hidden_similarity)
                    if num_samples == 0:
                        language_pair_stats[relation][pair][plot_key][
                            "num_samples"
                        ] = num_samples
                        language_pair_stats[relation][pair][plot_key][
                            "hidden_similarity_avg"
                        ] = None
                        continue

                    hidden_similarity_avg = np.mean(np.array(hidden_similarity), axis=0)

                    language_pair_stats[relation][pair][plot_key][
                        "num_samples"
                    ] = num_samples
                    language_pair_stats[relation][pair][plot_key][
                        "hidden_similarity_avg"
                    ] = hidden_similarity_avg
                else:
                    num_samples = language_pair_stats[relation][pair][plot_key][
                        "num_samples"
                    ]
                    if num_samples == 0:
                        continue
                    hidden_similarity_avg = language_pair_stats[relation][pair][
                        plot_key
                    ]["hidden_similarity_avg"]

                if min(hidden_similarity_avg) < 0:
                    y_min = -1

                layers = np.arange(len(hidden_similarity_avg))
                if plot_key in keys_to_display:
                    plt.plot(
                        layers,
                        hidden_similarity_avg,
                        label=f"{plot_key} (N={num_samples})",
                        color=colors[plot_key],
                    )
                    hidden_similarity_avg_all[pair][plot_key].append(
                        hidden_similarity_avg
                    )

            if "inconsistent" in keys_to_display and "inconsistent" not in plot_sets:
                plot_key = "inconsistent"
                num_samples_inconsistent = sum(
                    [
                        language_pair_stats[relation][pair][key]["num_samples"]
                        for key in ["correct_1", "correct_2", "incorrect_both"]
                    ]
                )
                avg_keys = [
                    key
                    for key in ["correct_1", "correct_2", "incorrect_both"]
                    if language_pair_stats[relation][pair][key]["hidden_similarity_avg"]
                    is not None
                ]
                hidden_similarity_avg_inconsistent = np.mean(
                    [
                        language_pair_stats[relation][pair][key][
                            "hidden_similarity_avg"
                        ]
                        for key in avg_keys
                    ],
                    axis=0,
                )

                if min(hidden_similarity_avg_inconsistent) < 0:
                    y_min = -1

                language_pair_stats[relation][pair]["inconsistent"] = {}

                language_pair_stats[relation][pair]["inconsistent"][
                    "num_samples"
                ] = num_samples_inconsistent
                language_pair_stats[relation][pair]["inconsistent"][
                    "hidden_similarity_avg"
                ] = hidden_similarity_avg_inconsistent

                plt.plot(
                    layers,
                    hidden_similarity_avg_inconsistent,
                    label=f"inconsistent (N={num_samples_inconsistent})",
                    color=colors["inconsistent"],
                )
                hidden_similarity_avg_inconsistent_all[pair].append(
                    hidden_similarity_avg_inconsistent
                )
                hidden_similarity_avg_all[pair][plot_key].append(
                    hidden_similarity_avg_inconsistent
                )

            # Add labels and title
            if similarity_type == "cosine_similarity":
                # plt.gca().set_ylim(y_min, 1)
                plt.gca().set_ylim(-0.05, 1.05)
            plt.xlabel("Layers", fontsize=font_sizes["label"])
            plt.ylabel(similarity_type, fontsize=font_sizes["label"])
            plt.title(
                f"{model_print_name} - Hidden states {similarity_type.split('_')[-1]}: {relation} - <{pair.replace('_', ', ')}>",
                fontsize=font_sizes["title"],
            )
            plt.tight_layout()

            # Add grid and legend
            plt.grid(alpha=0.1)
            plt.legend(loc="best", fontsize=font_sizes["legend"])

            plot_path = f"{output_dir}/plots/individual_pairs"

            if select_top_k_neurons != -1:
                plot_path = (
                    plot_path.replace(
                        "plots/", f"plots_top{select_top_k_neurons}_binary/"
                    )
                    if binary
                    else plot_path.replace(
                        "plots/", f"plots_top{select_top_k_neurons}/"
                    )
                )
            if set_neurons_threshold != -1:
                plot_path = (
                    plot_path.replace(
                        "plots/", f"plots_ths{set_neurons_threshold}_binary/"
                    )
                    if binary
                    else plot_path.replace(
                        "plots/", f"plots_ths{set_neurons_threshold}/"
                    )
                )

            os.makedirs(plot_path, exist_ok=True)
            plt.savefig(f"{plot_path}/{model_name}_similarity_{relation}_{pair}.png")
            plt.savefig(
                f"{plot_path}/{model_name}_similarity_{relation}_{pair}.pdf", dpi=200
            )
            plt.close()

    ### Section x: Plot curves for all relations for each langauge pair

    for pair in tqdm(hidden_similarity_avg_all.keys()):
        plt.figure(figsize=(10, 6))
        plot_sets = list(hidden_similarity_avg_all[pair].keys())
        y_min = 0
        for i, plot_key in enumerate(plot_sets):
            hidden_similarity_avg = np.array(
                hidden_similarity_avg_all[pair][plot_key]
            ).mean(axis=0)
            if min(hidden_similarity_avg_inconsistent) < 0:
                y_min = -1

            layers = np.arange(len(hidden_similarity_avg))
            if plot_key in keys_to_display:
                plt.plot(
                    layers,
                    hidden_similarity_avg,
                    label=f"{plot_key}",
                    color=colors[plot_key],
                )

        if "inconsistent" in keys_to_display and "inconsistent" not in plot_sets:
            plot_key = "inconsistent"
            hidden_similarity_avg_inconsistent = np.array(
                hidden_similarity_avg_inconsistent_all[pair]
            ).mean(axis=0)
            if min(hidden_similarity_avg_inconsistent) < 0:
                y_min = -1

            plt.plot(
                layers,
                hidden_similarity_avg_inconsistent,
                label=f"inconsistent",
                color=colors["inconsistent"],
            )

        # Add labels and title
        if similarity_type == "cosine_similarity":
            plt.gca().set_ylim(-0.05, 1.05)
        plt.xlabel("Layers", fontsize=font_sizes["label"])
        plt.ylabel(similarity_type, fontsize=font_sizes["label"])
        plt.title(
            f"{model_print_name} - Hidden states {similarity_type.split('_')[-1]}: All relations - <{pair.replace('_', ', ')}>",
            fontsize=font_sizes["title"],
        )
        plt.grid(True, linestyle="--")
        plt.tight_layout()

        # Add grid and legend
        plt.grid(alpha=0.1)
        plt.legend(loc="best", fontsize=font_sizes["legend"])

        plot_path = f"{output_dir}/plots/individual_pairs"

        if select_top_k_neurons != -1:
            plot_path = (
                plot_path.replace("plots/", f"plots_top{select_top_k_neurons}_binary/")
                if binary
                else plot_path.replace("plots/", f"plots_top{select_top_k_neurons}/")
            )
        if set_neurons_threshold != -1:
            plot_path = (
                plot_path.replace("plots/", f"plots_ths{set_neurons_threshold}_binary/")
                if binary
                else plot_path.replace("plots/", f"plots_ths{set_neurons_threshold}/")
            )

        os.makedirs(plot_path, exist_ok=True)
        plt.savefig(f"{plot_path}/{model_name}_similarity_all_relations_{pair}.png")
        plt.savefig(
            f"{plot_path}/{model_name}_similarity_all_relations_{pair}.pdf", dpi=200
        )
        plt.close()

    if "inconsistent" in keys_to_display and "inconsistent" not in plot_sets:
        print("Update language_pair_stats!")
        with open(stats_filepath, "wb") as f:
            pickle.dump(language_pair_stats, f)

    ### Section x: Plot curves for all relations and all languages together and compare different sets
    summarize_keys = ["correct_both", "inconsistent", "all"]
    average_vectors_all = []
    variance_vectors_all = []

    for summerize_key in summarize_keys:
        all_vectors = {}

        for key in hidden_similarity_avg_all.keys():
            if "all" in hidden_similarity_avg_all[key]:
                all_vectors[key] = np.array(
                    hidden_similarity_avg_all[key][summerize_key]
                ).mean(axis=0)

        # Determine the length of the vectors
        layers_length = len(list(all_vectors.values())[0])

        # Plot all 'all' vectors
        if model_name == "bloom":
            plt.figure(figsize=(10, 8))
        elif model_name == "llama2":
            plt.figure(figsize=(10, 12))
        colors = sns.color_palette("hls", len(all_vectors))
        for i, (key, vector) in enumerate(all_vectors.items()):
            plt.plot(
                range(layers_length), vector, label=key, color=colors[i], linestyle="--"
            )

        # Calculate the average and variance vectors
        average_vector = np.mean(list(all_vectors.values()), axis=0)
        variance_vector = np.std(list(all_vectors.values()), axis=0)

        average_vectors_all.append(average_vector)
        variance_vectors_all.append(variance_vector)

        # Plot the average vector
        plt.plot(
            range(layers_length),
            average_vector,
            label="Avg",
            color="#4a8ee1",
            linewidth=2.5,
        )

        # Plot the variance as a shaded area
        plt.fill_between(
            range(layers_length),
            average_vector - variance_vector,
            average_vector + variance_vector,
            color="#7AA9BB",
            alpha=0.1,
        )

        # Example custom names for each x-axis grid
        num_layers = layers_length // 2
        x_labels = [
            f"{idx//2}_pre" if idx % 2 == 0 else f"{idx//2}_mid"
            for idx in range(layers_length - 1)
        ]
        x_labels.append("final_post")

        # Add legend, labels, and title
        if similarity_type == "cosine_similarity":
            plt.gca().set_ylim(-0.05, 1.05)
        plt.legend(
            loc="upper center",  # Place the legend above the figure
            bbox_to_anchor=(0.5, -0.25),  # Position it outside the plot area
            ncol=5,  # Number of columns
            fontsize=font_sizes["legend"],
        )
        plt.xlabel("Layers", fontsize=font_sizes["label"])
        plt.ylabel(similarity_type, fontsize=font_sizes["label"])
        if model_name == "bloom":
            plt.xticks(
                ticks=list(range(len(x_labels)))[::2],
                labels=[v.split("_")[0] for v in x_labels[::2]],
                rotation=45,
            )
        elif model_name == "llama2":
            plt.xticks(
                ticks=list(range(len(x_labels)))[::4],
                labels=[v.split("_")[0] for v in x_labels[::4]],
                rotation=45,
            )
        plt.tick_params(labelsize=font_sizes["ticks"])
        plt.title(model_print_name, fontsize=font_sizes["title"])
        plt.grid(True, linestyle="--")

        if summerize_key == "all":
            if model_name == "bloom":
                # plt.axvline(x=18, color='tab:orange')
                plt.axvline(x=30, color="gray")
                plt.axvline(x=38, color="gray")
            elif model_name == "llama2":
                # plt.axvline(x=18, color='tab:orange')
                plt.axvline(x=24, color="gray")
                plt.axvline(x=56, color="gray")

        plt.tight_layout()

        if model_name == "bloom":
            plt.subplots_adjust(bottom=0.4)  # Minimal adjustment to fit the legend
        elif model_name == "llama2":
            plt.subplots_adjust(bottom=0.6)  # Minimal adjustment to fit the legend

        plot_path = f"{output_dir}/plots/"

        if select_top_k_neurons != -1:
            plot_path = (
                plot_path.replace("plots/", f"plots_top{select_top_k_neurons}_binary/")
                if binary
                else plot_path.replace("plots/", f"plots_top{select_top_k_neurons}/")
            )
        if set_neurons_threshold != -1:
            plot_path = (
                plot_path.replace("plots/", f"plots_ths{set_neurons_threshold}_binary/")
                if binary
                else plot_path.replace("plots/", f"plots_ths{set_neurons_threshold}/")
            )

        os.makedirs(plot_path, exist_ok=True)
        plt.savefig(
            f"{plot_path}/{model_name}_similarity_all_relations_all_pairs_{summerize_key}.png"
        )
        plt.savefig(
            f"{plot_path}/{model_name}_similarity_all_relations_all_pairs_{summerize_key}.pdf",
            dpi=200,
        )
        plt.close()

        if summerize_key == "all":
            with open(f"{output_dir}/language_pair_similarity.pkl", "wb") as f:
                pickle.dump(all_vectors, f)

    average_vectors_all = np.array(average_vectors_all)
    variance_vectors_all = np.array(variance_vectors_all)

    plt.figure(figsize=(10, 6))
    for i in range(average_vectors_all.shape[0]):
        if summarize_keys[i] == "all":
            continue
        average_vector = average_vectors_all[i]
        variance_vector = variance_vectors_all[i]
        # Plot the average vector
        plt.plot(range(layers_length), average_vector, label=summarize_keys[i])

        # Plot the variance as a shaded area
        plt.fill_between(
            range(layers_length),
            average_vector - variance_vector,
            average_vector + variance_vector,
            alpha=0.1,
        )
    # Add legend, labels, and title
    # Extract the current legend handles and labels
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    # Add the legend back with formatted labels
    plt.legend(
        handles=handles,
        labels=legend_labels,
        loc="lower left",
        ncol=2 if model_name == "bloom" else 4,
        fontsize=font_sizes["legend"],
    )
    plt.xlabel("Layers", fontsize=font_sizes["label"])
    plt.ylabel(similarity_type, fontsize=font_sizes["label"])
    if model_name == "bloom":
        plt.xticks(
            ticks=list(range(len(x_labels)))[::2],
            labels=[v.split("_")[0] for v in x_labels[::2]],
            rotation=45,
        )
    elif model_name == "llama2":
        plt.xticks(
            ticks=list(range(len(x_labels)))[::4],
            labels=[v.split("_")[0] for v in x_labels[::4]],
            rotation=45,
        )
    plt.tick_params(labelsize=font_sizes["ticks"])
    plt.title(
        f"{model_print_name} - Hidden states {similarity_type.split('_')[-1]}",
        fontsize=font_sizes["title"],
    )
    plt.grid(True, linestyle="--")
    plt.tight_layout()

    plot_path = f"{output_dir}/plots/"

    if select_top_k_neurons != -1:
        plot_path = (
            plot_path.replace("plots/", f"plots_top{select_top_k_neurons}_binary/")
            if binary
            else plot_path.replace("plots/", f"plots_top{select_top_k_neurons}/")
        )
    if set_neurons_threshold != -1:
        plot_path = (
            plot_path.replace("plots/", f"plots_ths{set_neurons_threshold}_binary/")
            if binary
            else plot_path.replace("plots/", f"plots_ths{set_neurons_threshold}/")
        )

    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(
        f"{plot_path}/{model_name}_similarity_all_relations_all_pairs_compare_average.png"
    )
    plt.savefig(
        f"{plot_path}/{model_name}_similarity_all_relations_all_pairs_compare_average.pdf",
        dpi=200,
    )
    plt.close()

    ### Section xxx: Plot English-centric similarity curves
    average_vectors_all = []
    variance_vectors_all = []

    for summerize_key in summarize_keys:
        all_vectors = {}

        for key in hidden_similarity_avg_all.keys():
            if "all" in hidden_similarity_avg_all[key] and "en" in key:
                all_vectors[key] = np.array(
                    hidden_similarity_avg_all[key][summerize_key]
                ).mean(axis=0)

        # Determine the length of the vectors
        layers_length = len(list(all_vectors.values())[0])

        # Plot all 'all' vectors
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette("hls", len(all_vectors))
        for i, (key, vector) in enumerate(all_vectors.items()):
            plt.plot(
                range(layers_length), vector, label=key, color=colors[i], linestyle="--"
            )

        # Calculate the average and variance vectors
        average_vector = np.mean(list(all_vectors.values()), axis=0)
        variance_vector = np.std(list(all_vectors.values()), axis=0)

        average_vectors_all.append(average_vector)
        variance_vectors_all.append(variance_vector)

        # Plot the average vector
        plt.plot(
            range(layers_length),
            average_vector,
            label="Avg",
            color="#4a8ee1",
            linewidth=2.5,
        )

        # Plot the variance as a shaded area
        plt.fill_between(
            range(layers_length),
            average_vector - variance_vector,
            average_vector + variance_vector,
            color="#7AA9BB",
            alpha=0.1,
        )

        # Example custom names for each x-axis grid
        num_layers = layers_length // 2
        x_labels = [
            f"{idx//2}_pre" if idx % 2 == 0 else f"{idx//2}_mid"
            for idx in range(layers_length - 1)
        ]
        x_labels.append("final_post")

        # Add legend, labels, and title
        # Extract the current legend handles and labels
        handles, legend_labels = plt.gca().get_legend_handles_labels()

        # Preprocess labels to be in the format <en, lang>
        formatted_labels = process_legends(legend_labels)
        # Add the legend back with formatted labels
        plt.legend(
            handles=handles,
            labels=formatted_labels,
            loc="lower left",
            ncol=2 if model_name == "bloom" else 4,
            fontsize=font_sizes["legend"],
        )
        plt.xlabel("Layers", fontsize=font_sizes["label"])
        plt.ylabel(similarity_type, fontsize=font_sizes["label"])
        if model_name == "bloom":
            plt.xticks(
                ticks=list(range(len(x_labels)))[::2],
                labels=[v.split("_")[0] for v in x_labels[::2]],
                rotation=45,
            )
        elif model_name == "llama2":
            plt.xticks(
                ticks=list(range(len(x_labels)))[::4],
                labels=[v.split("_")[0] for v in x_labels[::4]],
                rotation=45,
            )
        plt.tick_params(labelsize=font_sizes["ticks"])
        if summerize_key != "all":
            plt.title(
                f"{model_print_name} - Hidden states {similarity_type.split('_')[-1]}",
                fontsize=font_sizes["title"],
            )
        if summerize_key == "all":
            if model_name == "bloom":
                # plt.axvline(x=18, color='tab:orange')
                plt.axvline(x=30, color="gray")
                plt.axvline(x=38, color="gray")
                plt.text(
                    34,
                    0.325,
                    "Object Extraction\nin Latent Language",
                    color="black",
                    fontsize=17,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="lightgray", alpha=0.5, edgecolor="none"),
                )
                plt.text(
                    43,
                    0.45,
                    "Language Transition",
                    color="black",
                    fontsize=17,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="lightgray", alpha=0.5, edgecolor="none"),
                )
            elif model_name == "llama2":
                # plt.axvline(x=18, color='tab:orange')
                plt.axvline(x=24, color="gray")
                plt.axvline(x=56, color="gray")
                plt.text(
                    40,
                    0.9,
                    "Object Extraction\nin Latent Language",
                    color="black",
                    fontsize=17,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="lightgray", alpha=0.5, edgecolor="none"),
                )
                plt.text(
                    60,
                    0.9,
                    "Language\nTransition",
                    color="black",
                    fontsize=17,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="lightgray", alpha=0.5, edgecolor="none"),
                )

        if similarity_type == "cosine_similarity":
            plt.gca().set_ylim(-0.05, 1.05)
        plt.grid(True, linestyle="--")
        plt.tight_layout()
        plot_path = f"{output_dir}/plots/"

        if select_top_k_neurons != -1:
            plot_path = (
                plot_path.replace("plots/", f"plots_top{select_top_k_neurons}_binary/")
                if binary
                else plot_path.replace("plots/", f"plots_top{select_top_k_neurons}/")
            )
        if set_neurons_threshold != -1:
            plot_path = (
                plot_path.replace("plots/", f"plots_ths{set_neurons_threshold}_binary/")
                if binary
                else plot_path.replace("plots/", f"plots_ths{set_neurons_threshold}/")
            )

        os.makedirs(plot_path, exist_ok=True)
        plt.savefig(
            f"{plot_path}/{model_name}_similarity_all_relations_english_pairs_{summerize_key}.png"
        )
        plt.savefig(
            f"{plot_path}/{model_name}_similarity_all_relations_english_pairs_{summerize_key}.pdf",
            dpi=200,
        )
        plt.close()

    average_vectors_all = np.array(average_vectors_all)
    variance_vectors_all = np.array(variance_vectors_all)

    plt.figure(figsize=(10, 6))
    for i in range(average_vectors_all.shape[0]):
        if summarize_keys[i] == "all":
            continue
        average_vector = average_vectors_all[i]
        variance_vector = variance_vectors_all[i]
        # Plot the average vector
        plt.plot(range(layers_length), average_vector, label=summarize_keys[i])

        # Plot the variance as a shaded area
        plt.fill_between(
            range(layers_length),
            average_vector - variance_vector,
            average_vector + variance_vector,
            alpha=0.1,
        )

    # Add legend, labels, and title
    # Extract the current legend handles and labels
    handles, legend_labels = plt.gca().get_legend_handles_labels()

    # Preprocess labels to be in the format <en, lang>
    formatted_labels = process_legends(legend_labels)
    # Add the legend back with formatted labels
    plt.legend(
        handles=handles,
        labels=formatted_labels,
        loc="lower left",
        ncol=2 if model_name == "bloom" else 4,
        fontsize=font_sizes["legend"],
    )
    plt.xlabel("Layers", fontsize=font_sizes["label"])
    plt.ylabel(similarity_type, fontsize=font_sizes["label"])
    if model_name == "bloom":
        plt.xticks(
            ticks=list(range(len(x_labels)))[::2],
            labels=[v.split("_")[0] for v in x_labels[::2]],
            rotation=45,
        )
    elif model_name == "llama2":
        plt.xticks(
            ticks=list(range(len(x_labels)))[::4],
            labels=[v.split("_")[0] for v in x_labels[::4]],
            rotation=45,
        )
    plt.tick_params(labelsize=font_sizes["ticks"])
    plt.title(
        f"{model_print_name} - Hidden states {similarity_type.split('_')[-1]}",
        fontsize=font_sizes["title"],
    )
    plt.grid(True, linestyle="--")
    plt.tight_layout()

    plot_path = f"{output_dir}/plots/"

    if select_top_k_neurons != -1:
        plot_path = (
            plot_path.replace("plots/", f"plots_top{select_top_k_neurons}_binary/")
            if binary
            else plot_path.replace("plots/", f"plots_top{select_top_k_neurons}/")
        )
    if set_neurons_threshold != -1:
        plot_path = (
            plot_path.replace("plots/", f"plots_ths{set_neurons_threshold}_binary/")
            if binary
            else plot_path.replace("plots/", f"plots_ths{set_neurons_threshold}/")
        )

    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(
        f"{plot_path}/{model_name}_similarity_all_relations_all_english_pairs_compare_average.png"
    )
    plt.savefig(
        f"{plot_path}/{model_name}_similarity_all_relations_all_english_pairs_compare_average.pdf",
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    ### Section xxx: Plot similarity together with ablation(s) (for all relations, all samples)
    average_vector = average_vectors_all[-1]
    variance_vector = variance_vectors_all[-1]
    # Plot the average vector
    plt.plot(
        range(layers_length),
        average_vector,
        label="same relation, same object (Parallel)",
        color=colors_ablation["parallel"],
        linewidth=2.5,
    )

    # Plot the variance as a shaded area
    plt.fill_between(
        range(layers_length),
        average_vector - variance_vector,
        average_vector + variance_vector,
        alpha=0.1,
        color=colors_ablation["parallel"],
    )

    # Add the ablation 1 curve
    if os.path.exists(f"{output_dir}/language_pair_stats_ablation.pkl"):
        with open(f"{output_dir}/language_pair_stats_ablation.pkl", "rb") as f:
            similarity_intra_relation_layers = pickle.load(f)

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
        all_languages_variance = np.std(np.stack(all_languages_values), axis=0)

        plt.plot(
            range(layers_length),
            similarity_intra_relation_layers["all_relations"]["all_languages"],
            label="same relation, different objects (Dissection 1)",
            color=colors_ablation["ablation 1"],
            linewidth=2.5,
        )

        # Plot the variance as a shaded area
        plt.fill_between(
            range(layers_length),
            similarity_intra_relation_layers["all_relations"]["all_languages"]
            - all_languages_variance,
            similarity_intra_relation_layers["all_relations"]["all_languages"]
            + all_languages_variance,
            alpha=0.1,
            color=colors_ablation["ablation 1"],
        )

    # Add the ablation 2 curve
    if os.path.exists(f"{output_dir}/language_pair_stats_ablation_2.pkl"):
        with open(f"{output_dir}/language_pair_stats_ablation_2.pkl", "rb") as f:
            similarity_inter_relation_layers = pickle.load(f)

        # Merge all language pairs together
        # Iterate through all language pairs in "all_relations"
        try:
            all_languages_values = []
            for subkey, values in similarity_inter_relation_layers.items():
                if (
                    subkey != "all_languages"
                ):  # Exclude any existing "all_languages" key
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
                if (
                    subkey != "all_languages"
                ):  # Exclude any existing "all_languages" key
                    all_languages_values.append(values)

            # Calculate the average values across all language pairs
            similarity_inter_relation_layers["all_languages"] = np.mean(
                np.stack(all_languages_values), axis=0
            )

        all_languages_variance = np.std(np.stack(all_languages_values), axis=0)

        plt.plot(
            range(layers_length),
            similarity_inter_relation_layers["all_languages"],
            label="different relations, different objects (Dissection 2)",
            color=colors_ablation["ablation 2"],
            linewidth=2.5,
        )

        # Plot the variance as a shaded area
        plt.fill_between(
            range(layers_length),
            similarity_inter_relation_layers["all_languages"] - all_languages_variance,
            similarity_inter_relation_layers["all_languages"] + all_languages_variance,
            alpha=0.1,
            color=colors_ablation["ablation 2"],
        )

    # Add legend, labels, and title
    plt.legend(loc="lower left", fontsize=font_sizes["legend"], framealpha=0.5)
    plt.xlabel("Layers", fontsize=font_sizes["label"])
    plt.ylabel(similarity_type, fontsize=font_sizes["label"])
    if model_name == "bloom":
        plt.xticks(
            ticks=list(range(len(x_labels)))[::2],
            labels=[v.split("_")[0] for v in x_labels[::2]],
            rotation=45,
        )
    elif model_name == "llama2":
        plt.xticks(
            ticks=list(range(len(x_labels)))[::4],
            labels=[v.split("_")[0] for v in x_labels[::4]],
            rotation=45,
        )
    plt.tick_params(labelsize=font_sizes["ticks"])
    if similarity_type == "cosine_similarity":
        plt.gca().set_ylim(-0.05, 1.05)
    # plt.title(f"{model_print_name} - Hidden states {similarity_type.split('_')[-1]} - Ablations", fontsize=font_sizes['title'])
    if model_name == "bloom":
        # plt.axvline(x=18, color=colors_ablation["ablation 2"])
        # plt.axvline(x=30, color=colors_ablation["ablation 1"])
        # plt.axvline(x=38, color=colors_ablation["parallel"])
        plt.axvline(x=18, color="gray")
        plt.axvline(x=30, color="gray")
        plt.axvline(x=38, color="gray")
        plt.text(
            24,
            0.625,
            "Relation Processing\nin Latent Language",
            color="black",
            fontsize=17,
            ha="center",
            va="center",
            bbox=dict(facecolor="lightgray", alpha=0.5, edgecolor="none"),
        )
        plt.text(
            34,
            0.325,
            "Object Extraction\nin Latent Language",
            color="black",
            fontsize=17,
            ha="center",
            va="center",
            bbox=dict(facecolor="lightgray", alpha=0.5, edgecolor="none"),
        )
        plt.text(
            43,
            0.45,
            "Language Transition",
            color="black",
            fontsize=17,
            ha="center",
            va="center",
            bbox=dict(facecolor="lightgray", alpha=0.5, edgecolor="none"),
        )

    elif model_name == "llama2":
        # plt.axvline(x=18, color=colors_ablation["ablation 2"])
        # plt.axvline(x=24, color=colors_ablation["ablation 1"])
        # plt.axvline(x=56, color=colors_ablation["parallel"])
        plt.axvline(x=18, color="gray")
        plt.axvline(x=24, color="gray")
        plt.axvline(x=56, color="gray")
        plt.text(
            21,
            0.4,
            "Relation Processing\nin Latent Language",
            color="black",
            fontsize=17,
            ha="center",
            va="center",
            bbox=dict(facecolor="lightgray", alpha=0.5, edgecolor="none"),
        )
        plt.text(
            40,
            0.9,
            "Object Extraction\nin Latent Language",
            color="black",
            fontsize=17,
            ha="center",
            va="center",
            bbox=dict(facecolor="lightgray", alpha=0.5, edgecolor="none"),
        )
        plt.text(
            60,
            0.9,
            "Language\nTransition",
            color="black",
            fontsize=17,
            ha="center",
            va="center",
            bbox=dict(facecolor="lightgray", alpha=0.5, edgecolor="none"),
        )

    plt.grid(True, linestyle="--")
    plt.tight_layout()

    plot_path = f"{output_dir}/plots/"
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(
        f"{plot_path}/{model_name}_similarity_all_relations_all_english_pairs_w_ablation.png"
    )
    plt.savefig(
        f"{plot_path}/{model_name}_similarity_all_relations_all_english_pairs_w_ablation.pdf",
        dpi=200,
    )
    plt.close()
