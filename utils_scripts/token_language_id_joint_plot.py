import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import cycle
from collections import defaultdict

# Directories for LLaMA2 and BLOOM model results
llama2_dir = "results_dev/language_id_llama2"
bloom_dir = "results_dev/language_id_bloom"

# Option to remove "UNK" and "OTHERS" from plots
remove_unknown = True  # Set to False to include "UNK" and "OTHERS"

frequent_languages = ["zh", "pt", "vi", "ja", "en", "ca", "es", "fr", "de", "ar"]
frequent_languages_2 = ["it", "hu", "nl", "cs", "war", "sr", "ro", "no", "oc"]
frequent_colors = sns.color_palette(
    "Set3", len(frequent_languages)
) + sns.color_palette("husl", len(frequent_languages_2))


# Define a custom color palette
language_colors = {
    language: color
    for language, color in zip(
        frequent_languages + frequent_languages_2, frequent_colors
    )
}
language_colors["UNK"] = "black"
language_colors["OTHERS"] = "gray"

# # Define a custom color palette
default_colors = cycle(sns.color_palette("tab20c") + sns.color_palette("tab20")[::2])

languages_code2name = {
    "ar": "Arabic",
    "ca": "Catalan",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hu": "Hungarian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "ru": "Russian",
    "uk": "Ukrainian",
    "vi": "Vitnamese",
    "zh": "Chinese",
}

font_sizes = {
    "title": 28,
    "ticks": 18,
    "label": 26,
    "legend": 18,
}


# Function to ensure consistent colors for all languages
def get_language_color(language):
    if language not in language_colors:
        language_colors[language] = next(default_colors)
    return language_colors[language]


# Load all files for LLaMA2 and BLOOM
llama2_files = [f for f in os.listdir(llama2_dir) if f.endswith(".json")]
bloom_files = [f for f in os.listdir(bloom_dir) if f.endswith(".json")]


def compute_aggregated_proportions(model_dir, files, remove_unknown=True):
    aggregated_stats = defaultdict(lambda: defaultdict(list))

    for file in files:
        with open(os.path.join(model_dir, file), "r") as f:
            data = json.load(f)
            for layer, languages in data.items():
                for lang, proportion in languages.items():
                    if remove_unknown and lang in {"UNK", "OTHERS"}:
                        continue
                    aggregated_stats[layer][lang].append(proportion)

    # Aggregate and normalize proportions
    aggregated_results = {}
    for layer, lang_data in aggregated_stats.items():
        aggregated_layer = {
            lang: np.sum(proportions) / len(files)
            for lang, proportions in lang_data.items()
        }
        # Normalize values to sum up to 1.0
        total = sum(aggregated_layer.values())
        normalized_layer = {
            lang: value / total for lang, value in aggregated_layer.items()
        }
        aggregated_results[layer] = normalized_layer

    return aggregated_results


# Prepare data for plotting
def prepare_data(aggregated_stats):
    layers = list(aggregated_stats.keys())
    all_languages = set(
        lang for layer_data in aggregated_stats.values() for lang in layer_data.keys()
    )

    proportions = {}
    for language in all_languages:
        proportions[language] = [
            aggregated_stats[layer].get(language, 0) for layer in layers
        ]
    return layers, sorted(all_languages), proportions


# Plot individual language comparison
def plot_language_comparison(shared_lang, llama2_path, bloom_path):
    # Load data for both models
    with open(llama2_path, "r") as f:
        llama2_data = json.load(f)
    with open(bloom_path, "r") as f:
        bloom_data = json.load(f)

    # Prepare data for plotting
    def prepare_data(data):
        layers = list(data.keys())
        all_languages = set(
            lang for layer_data in data.values() for lang in layer_data.keys()
        )

        if remove_unknown:
            all_languages -= {"UNK", "OTHERS"}

        proportions = {}
        for language in all_languages:
            proportions[language] = [data[layer].get(language, 0) for layer in layers]

        # Normalize proportions if removing "UNK" and "OTHERS"
        if remove_unknown:
            for layer_idx in range(len(layers)):
                total = sum(proportions[lang][layer_idx] for lang in proportions)
                for lang in proportions:
                    proportions[lang][layer_idx] = (
                        proportions[lang][layer_idx] / total if total > 0 else 0
                    )

        sorted_languages = sorted(all_languages)
        if not remove_unknown:
            sorted_languages = sorted(all_languages - {"OTHERS", "UNK"}) + [
                "UNK",
                "OTHERS",
            ]
        return layers, sorted_languages, proportions

    llama2_layers, llama2_sorted_languages, llama2_proportions = prepare_data(
        llama2_data
    )
    bloom_layers, bloom_sorted_languages, bloom_proportions = prepare_data(bloom_data)

    # Create subplots
    x_llama2 = np.arange(len(llama2_layers))
    x_bloom = np.arange(len(bloom_layers))
    # fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 14), sharex=False)  # 2 rows, 1 column
    # fig.suptitle(f"Language Composition Comparison: {shared_lang}", fontsize=16)

    # Dictionaries to collect legend elements
    legend_handles = {}
    legend_labels = {}

    # Plot LLaMA2 (left subplot)
    bottom = np.zeros(len(llama2_layers))
    for language in llama2_sorted_languages:
        bar = axes[0].bar(
            x_llama2,
            llama2_proportions[language],
            label=language,
            bottom=bottom,
            color=get_language_color(language),
        )
        bottom += llama2_proportions[language]

        legend_handles[language] = bar[0]  # Collect unique handles

    axes[0].set_title(
        f"LLaMA2 - {languages_code2name[shared_lang]} ({shared_lang})",
        fontsize=font_sizes["title"],
    )
    axes[0].set_xticks(x_llama2[::2])
    axes[0].set_xticklabels(
        [l.split("_")[0] for l in llama2_layers][::2], rotation=45, ha="right"
    )
    axes[0].tick_params(labelsize=font_sizes["ticks"])
    # axes[0].set_xlabel("Layers", fontsize=font_sizes["label"])
    axes[0].set_ylabel("Proportion", fontsize=font_sizes["label"])

    # Plot BLOOM (right subplot)
    bottom = np.zeros(len(bloom_layers))
    for language in bloom_sorted_languages:
        bar = axes[1].bar(
            x_bloom,
            bloom_proportions[language],
            label=language,
            bottom=bottom,
            color=get_language_color(language),
        )
        bottom += bloom_proportions[language]
        legend_handles[language] = bar[0]  # Collect unique handles

    axes[1].set_title(
        f"BLOOM - {languages_code2name[shared_lang]} ({shared_lang})",
        fontsize=font_sizes["title"],
    )
    axes[1].set_xticks(x_bloom)
    axes[1].set_xticklabels(
        [l.split("_")[0] for l in bloom_layers], rotation=45, ha="right"
    )
    axes[1].tick_params(labelsize=font_sizes["ticks"])
    axes[1].set_xlabel("Layers", fontsize=font_sizes["label"])
    axes[1].set_ylabel("Proportion", fontsize=font_sizes["label"])

    # Extract unique languages and handles for legend
    unique_languages = sorted(legend_handles.keys())
    unique_handles = [legend_handles[lang] for lang in unique_languages]

    # # Adjust layout for legend
    # plt.tight_layout(rect=[0, 0, 0.88, 1])  # Reserve space for legend
    # legend = fig.legend(
    #     handles=unique_handles, labels=unique_languages,
    #     loc="center right", title="Languages", fontsize=10, title_fontsize=12, ncol=2
    # )
    # legend.set_bbox_to_anchor((1, 0.5))  # Move legend to the right of the subplots

    fig.legend(
        handles=unique_handles,
        labels=unique_languages,
        loc="lower center",
        ncol=7,
        fontsize=font_sizes["legend"],
    )
    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0.14, 1, 1])  # Reserve space at the bottom for the legend
    # plt.subplots_adjust(hspace=0.2)  # Add vertical space between subplots

    # Save the figure
    output_filename = (
        f"results_dev/language_id_llama2/{shared_lang}_llama2_bloom_comparison.png"
    )
    plt.savefig(output_filename)
    plt.savefig(output_filename.replace(".png", ".pdf"), dpi=200)
    plt.close(fig)
    print(f"Saved plot: {output_filename}")


# Plot aggregated comparison
def plot_aggregated_comparison(llama2_aggregated, bloom_aggregated):
    llama2_layers, llama2_languages, llama2_proportions = prepare_data(
        llama2_aggregated
    )
    bloom_layers, bloom_languages, bloom_proportions = prepare_data(bloom_aggregated)

    # fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 14), sharex=False)  # 2 rows, 1 column
    # fig.suptitle("Aggregated Language Composition Comparison", fontsize=16)

    # Plot LLaMA2 aggregated statistics
    x_llama2 = np.arange(len(llama2_layers))
    bottom = np.zeros(len(llama2_layers))
    for language in llama2_languages:
        axes[0].bar(
            x_llama2,
            llama2_proportions[language],
            label=language,
            bottom=bottom,
            color=get_language_color(language),
        )
        bottom += llama2_proportions[language]
    axes[0].set_title("LLaMA2 - All languages", fontsize=font_sizes["title"])
    axes[0].set_xticks(x_llama2[::2])
    axes[0].set_xticklabels(
        [l.split("_")[0] for l in llama2_layers][::2], rotation=45, ha="right"
    )
    axes[0].tick_params(labelsize=font_sizes["ticks"])
    # axes[0].set_xlabel("Layers", fontsize=font_sizes["label"])
    axes[0].set_ylabel("Proportion", fontsize=font_sizes["label"])

    # Plot BLOOM aggregated statistics
    x_bloom = np.arange(len(bloom_layers))
    bottom = np.zeros(len(bloom_layers))
    for language in bloom_languages:
        axes[1].bar(
            x_bloom,
            bloom_proportions[language],
            label=language,
            bottom=bottom,
            color=get_language_color(language),
        )
        bottom += bloom_proportions[language]
    axes[1].set_title("BLOOM - All languages", fontsize=font_sizes["title"])
    axes[1].set_xticks(x_bloom)
    axes[1].set_xticklabels(
        [l.split("_")[0] for l in bloom_layers], rotation=45, ha="right"
    )
    axes[1].tick_params(labelsize=font_sizes["ticks"])
    axes[1].set_xlabel("Layers", fontsize=font_sizes["label"])
    axes[1].set_ylabel("Proportion", fontsize=font_sizes["label"])

    # Extract unique languages and handles for legend
    all_languages = sorted(set(llama2_languages) | set(bloom_languages))
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=get_language_color(lang))
        for lang in all_languages
    ]

    # # Add a shared legend
    # fig.legend(
    #     handles=legend_handles, labels=all_languages,
    #     loc="center right", title="Languages", fontsize=10, title_fontsize=12, ncol=2
    # )

    # # Adjust layout and save the figure
    # plt.tight_layout(rect=[0, 0, 0.85, 1])

    fig.legend(
        handles=legend_handles,
        labels=all_languages,
        loc="lower center",
        ncol=7,
        fontsize=font_sizes["legend"],
    )
    # Adjust layout to make space for the legend
    plt.tight_layout(
        rect=[0, 0.225, 1, 1]
    )  # Reserve space at the bottom for the legend
    # plt.subplots_adjust(hspace=0.2)  # Add vertical space between subplots

    output_filename = (
        "results_dev/language_id_llama2/aggregated_language_composition_comparison.png"
    )
    plt.savefig(output_filename)
    plt.savefig(output_filename.replace(".png", ".pdf"), dpi=200)
    plt.close(fig)
    print(f"Saved aggregated plot: {output_filename}")


# Main Execution
llama2_aggregated = compute_aggregated_proportions(
    llama2_dir, llama2_files, remove_unknown
)
bloom_aggregated = compute_aggregated_proportions(
    bloom_dir, bloom_files, remove_unknown
)

# Plot individual languages
shared_languages = set(llama2_files) & set(bloom_files)
for lang_file in sorted(shared_languages):
    shared_lang = lang_file.split("_")[0]
    plot_language_comparison(
        shared_lang,
        os.path.join(llama2_dir, lang_file),
        os.path.join(bloom_dir, lang_file),
    )

# Plot aggregated comparison
plot_aggregated_comparison(llama2_aggregated, bloom_aggregated)
