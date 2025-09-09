import os
import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

# Define colors for languages (extend if necessary)
default_colors = cycle(
    plt.cm.Spectral(np.linspace(0, 1, 50))
)  # 50 colors from Spectral colormap
language_colors = {}


# Function to ensure consistent colors for all languages
def get_language_color(language):
    if language not in language_colors:
        language_colors[language] = next(default_colors)
    return language_colors[language]


# Function to load and process JSON files, with optional removal of "UNK" and "OTHERS"
def load_and_process_json(file_path, remove_unknown):
    with open(file_path, "r") as f:
        data = json.load(f)

    if remove_unknown:
        processed_data = {}
        for layer, languages in data.items():
            filtered_layer = {
                lang: prop
                for lang, prop in languages.items()
                if lang not in {"UNK", "OTHERS"}
            }
            total = sum(filtered_layer.values())
            normalized_layer = (
                {lang: value / total for lang, value in filtered_layer.items()}
                if total > 0
                else filtered_layer
            )
            processed_data[layer] = normalized_layer
        return processed_data
    else:
        return data


# Function to plot a single language composition bar chart
def plot_single_language(model_name, language, language_data):
    layers = list(language_data.keys())
    languages = sorted(
        set(lang for layer in language_data.values() for lang in layer.keys())
    )  # Unique langs in all layers

    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = np.zeros(len(layers))
    for lang in languages:
        values = [language_data[layer].get(lang, 0) for layer in layers]
        ax.bar(
            layers, values, label=lang, bottom=bottom, color=get_language_color(lang)
        )
        bottom += np.array(values)

    ax.set_xlabel("Layers")
    ax.set_ylabel("Proportion")
    ax.set_title(f"Language Composition: {language} ({model_name})")
    ax.set_xticklabels(layers, rotation=45, ha="right")
    ax.legend(loc="upper right", ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(f"language_composition_{language}_{model_name}.png")
    plt.savefig(f"language_composition_{language}_{model_name}.pdf", dpi=200)
    plt.close()


# Define directories containing JSON files
llama2_dir = "results_dev/language_id_llama2"
bloom_dir = "results_dev/language_id_bloom"

# Get all JSON files
llama2_files = [f for f in os.listdir(llama2_dir) if f.endswith(".json")]
bloom_files = [f for f in os.listdir(bloom_dir) if f.endswith(".json")]

# Extract common and unique languages
llama2_languages = set(f.split("_")[0] for f in llama2_files)
bloom_languages = set(f.split("_")[0] for f in bloom_files)

common_languages = llama2_languages & bloom_languages
llama2_unique_languages = llama2_languages - bloom_languages
bloom_unique_languages = bloom_languages - llama2_languages

remove_unknown = True  # Set this flag to remove "UNK" and "OTHERS" if needed

# Plot for common languages (side-by-side comparison)
for lang in sorted(common_languages):
    llama2_data = load_and_process_json(
        os.path.join(llama2_dir, f"{lang}_language_composition_stats.json"),
        remove_unknown,
    )
    bloom_data = load_and_process_json(
        os.path.join(bloom_dir, f"{lang}_language_composition_stats.json"),
        remove_unknown,
    )

    layers = list(llama2_data.keys())  # Assuming layers are the same in both models
    languages = sorted(
        set(lang for layer in llama2_data.values() for lang in layer.keys())
        | set(lang for layer in bloom_data.values() for lang in layer.keys())
    )  # Unique langs in both models

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 10), sharex=True
    )  # Two subplots: one above another

    for ax, model_name, model_data in zip(
        axes, ["LLaMA2", "BLOOM"], [llama2_data, bloom_data]
    ):
        layers = list(model_data.keys())
        bottom = np.zeros(len(layers))
        for language in languages:
            values = [model_data[layer].get(language, 0) for layer in layers]
            ax.bar(
                layers,
                values,
                label=language,
                bottom=bottom,
                color=get_language_color(language),
            )
            bottom += np.array(values)

        ax.set_ylabel("Proportion")
        ax.set_title(f"{model_name}")

    axes[-1].set_xticklabels(layers, rotation=45, ha="right")
    fig.legend(loc="lower center", ncol=10, fontsize=8, bbox_to_anchor=(0.5, -0.1))

    plt.tight_layout()
    plt.savefig(f"language_composition_comparison_{lang}.png")
    plt.savefig(f"language_composition_comparison_{lang}.pdf", dpi=200)
    plt.close()

# Plot unique languages for LLaMA2
for lang_file in sorted(llama2_unique_languages):
    unique_lang = lang_file.split("_")[0]
    if unique_lang == "language":
        continue
    processed_data = load_and_process_json(
        os.path.join(llama2_dir, f"{unique_lang}_language_composition_stats.json"),
        remove_unknown,
    )
    plot_single_language("LLaMA2", unique_lang, processed_data)

# Plot unique languages for BLOOM
for lang_file in sorted(bloom_unique_languages):
    unique_lang = lang_file.split("_")[0]
    processed_data = load_and_process_json(
        os.path.join(bloom_dir, f"{unique_lang}_language_composition_stats.json"),
        remove_unknown,
    )
    plot_single_language("BLOOM", unique_lang, processed_data)
