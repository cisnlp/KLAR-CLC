import os
import json
import fasttext
import argparse
from tqdm import tqdm
from collections import defaultdict

import sys
sys.path.append(".")
sys.path.append("..")
from plot_scripts.rank_plot_en_pivot import extract_relations_languages

# Load pre-trained language identification model
model = fasttext.load_model("utils_scripts/langid/lid.176.bin")


def identify_language_fasttext(token, model_name=None, language=None):
    token = token.strip().strip("▁")
    # Check if the token is purely non-alphanumeric (e.g., punctuation, whitespace)
    if not any(char.isalnum() for char in token):
        return "UNK", 1.0  # Mark as UNK with confidence 1.0

    prediction = model.predict([token])
    lang, confidence = prediction[0][0][0].split("_")[-1], prediction[1][0].item()
    return lang, confidence


def process_language_identification(filepath):
    """Process a single JSON file for token-level language identification."""
    with open(filepath, "r") as file:
        data = json.load(file)

    # Step 1: Get the last token index
    last_token_index = data["answer_token_span_test"][0] - 1

    # Step 2: Filter relevant keys in "logit_lens_result" and extract predicted tokens
    resid_keys = {
        key: value
        for key, value in data["logit_lens_result"]["resid"].items()
        if key.endswith("_pre") or key == "final_post"
    }

    results_dict = {}
    for key, values in resid_keys.items():
        if last_token_index < len(values):  # Check if the index exists
            top_tokens = values[last_token_index]["top_token_strings"]

            # Identify languages and confidence
            languages = [identify_language_fasttext(token)[0] for token in top_tokens]
            confidences = [identify_language_fasttext(token)[1] for token in top_tokens]

            # Save results for the current key
            results_dict[key] = {
                "top_tokens": top_tokens,
                "language_list": languages,
                "langid_confidence": confidences,
            }

    # Step 5: Save results to a JSON file
    output_dir = f"{os.path.dirname(filepath)}/language_id"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(filepath))

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(results_dict, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama2")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="llm-transparency-tool-bak/backend_results_<MODEL>_fewshot_demo_new",
    )
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--process", type=str, default="False")
    parser.add_argument("--do_statistics", type=str, default="False")
    parser.add_argument("--plot", type=str, default="True")
    parser.add_argument("--without_unknown", type=str, default="True")
    args = parser.parse_args()

    # model_languages = {
    #     "bloom": ['ar', 'ca', 'en', 'es', 'fr', 'vi', 'zh'],
    #     "llama2": ['ca', 'en', 'es', 'fr', 'hu', 'ja', 'ko', 'nl', 'ru', 'uk', 'vi', 'zh']
    # }
    results_dir = args.results_dir.replace("<MODEL>", args.model_name)
    relations, languages = extract_relations_languages(results_dir)
    correctness_list = ["correct", "incorrect", "others"]

    model_print_name = "LLaMA2" if args.model_name == "llama2" else "BLOOM"

    if args.process == "True":
        """
        ##################################################################
        Batch process all JSON files in the specified directory structure.
        ##################################################################
        """
        for relation in tqdm(relations):
            for language in languages:
                results_path = os.path.join(results_dir, f"{relation}_{language}")

                for correctness in correctness_list:
                    root = os.path.join(results_path, correctness)
                    if not os.path.exists(root):
                        continue

                    files = [
                        file
                        for file in os.listdir(root)
                        if file.endswith(".json")
                        and file.split("_")[-1].replace(".json", "").isdigit()
                    ]
                    for file in tqdm(
                        sorted(files), desc=f"{relation}_{language}/{correctness}"
                    ):
                        filepath = os.path.join(root, file)
                        process_language_identification(filepath)

    if args.do_statistics == "True":
        """
        ##################################################################
        Language composition statistics
        ##################################################################
        """
        # Confidence threshold
        CONFIDENCE_THRESHOLD = args.confidence_threshold

        for language in languages:
            # Dictionary to store the statistics
            stats = defaultdict(lambda: defaultdict(float))
            # Process each file in the directory
            file_count = 0
            for relation in tqdm(relations):
                for correctness in correctness_list:
                    results_path = os.path.join(
                        results_dir,
                        f"{relation}_{language}",
                        correctness,
                        "language_id",
                    )
                    if not os.path.exists(results_path):
                        print(f"{results_path} doesn't exist! Skipping...")
                        continue

                    print(f"Processing {relation}_{language}/{correctness}...")
                    for filename in sorted(os.listdir(results_path)):
                        if filename.endswith(".json"):
                            file_path = os.path.join(results_path, filename)
                            with open(file_path, "r") as file:
                                data = json.load(file)
                                for layer, layer_data in data.items():
                                    language_counts = defaultdict(int)
                                    total_tokens = 0

                                    # Count the languages with confidence handling
                                    for lang, conf in zip(
                                        layer_data["language_list"],
                                        layer_data["langid_confidence"],
                                    ):
                                        if conf < CONFIDENCE_THRESHOLD:
                                            lang = "UNK"
                                        language_counts[lang] += 1
                                        total_tokens += 1

                                    # Additional processing for Llama2
                                    for token, lang in zip(
                                        layer_data["top_tokens"],
                                        layer_data["language_list"],
                                    ):
                                        if (
                                            args.model_name == "llama2"
                                            and language in ["ja", "ko", "zh"]
                                        ):
                                            if layer == "final_post":
                                                if token.startswith("<0x"):
                                                    lang = language
                                            elif (
                                                int(layer.split("_")[0]) < 5
                                                or int(layer.split("_")[0])
                                                > len(data) - 5
                                            ):
                                                if token.startswith("<0x"):
                                                    lang = language

                                        language_counts[lang] += 1
                                        total_tokens += 1

                                    # Calculate percentages for each language
                                    for lang, count in language_counts.items():
                                        stats[layer][lang] += count / total_tokens

                            file_count += 1

            # Average the results across files
            for layer, lang_stats in stats.items():
                for lang in lang_stats:
                    stats[layer][lang] /= file_count

            sorted_stats = {}
            for layer, languages in stats.items():
                others_sum = 0
                to_remove = []

                # Identify languages to merge into "OTHERS"
                for lang, proportion in languages.items():
                    if lang != "UNK" and proportion < 0.04:  # threshold as 0.04
                        others_sum += proportion
                        to_remove.append(lang)

                # Remove low-proportion languages and add "OTHERS"
                for lang in to_remove:
                    del languages[lang]
                if others_sum > 0:
                    languages["OTHERS"] = others_sum

                # Reorder to ensure "UNK" is second last and "OTHERS" is last
                if "UNK" in languages:
                    unk_value = languages.pop("UNK")
                    if "OTHERS" in languages:
                        others_value = languages.pop("OTHERS")
                        sorted_languages = dict(sorted(languages.items()))
                        sorted_languages["UNK"] = unk_value
                        sorted_languages["OTHERS"] = others_value
                    else:
                        sorted_languages = dict(sorted(languages.items()))
                        sorted_languages["UNK"] = unk_value
                else:
                    sorted_languages = dict(sorted(languages.items()))
                    if "OTHERS" in languages:
                        others_value = languages.pop("OTHERS")
                        sorted_languages["OTHERS"] = others_value

                # Update the layer with the sorted dictionary
                sorted_stats[layer] = sorted_languages

                # sorted_subdict = {
                #     k: subdict[k] for k in sorted(
                #         subdict.keys(),
                #         key=lambda x: (x == "UNK", x)  # Sort alphabetically, UNK at the end
                #     )
                # }
                # sorted_stats[key] = sorted_subdict

            # Save the statistics to a JSON file
            output_file = f"results_dev/language_id_{args.model_name}/{language}_language_composition_stats.json"
            with open(output_file, "w") as f:
                json.dump(sorted_stats, f, indent=4)

    if args.plot == "True":
        """
        ##################################################################
        Plot stacked bar charts to show language composition
        ##################################################################
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from itertools import cycle

        frequent_languages = [
            "zh",
            "pt",
            "vi",
            "ja",
            "en",
            "ca",
            "es",
            "fr",
            "de",
            "ar",
        ]
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

        default_colors = cycle(
            sns.color_palette("tab20c") + sns.color_palette("tab20")[::2]
        )

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
        def get_language_color(lang):
            if lang not in language_colors:
                language_colors[lang] = next(default_colors)
            return language_colors[lang]

        json_files = [
            f"results_dev/language_id_{args.model_name}/{file}"
            for file in os.listdir(f"results_dev/language_id_{args.model_name}")
            if file.endswith(".json")
        ]
        # Process each file in the directory
        for file in sorted(json_files):
            # Load JSON data
            with open(file, "r") as f:
                data = json.load(f)
            current_language = os.path.basename(file).split("_")[0]
            if current_language == "language":
                continue

            # Extract layers and languages
            layers = list(data.keys())
            languages = set(lang for layer in data.values() for lang in layer.keys())

            # Sort languages alphabetically with "OTHERS" at the bottom and "UNK" second to last
            sorted_languages = sorted(languages - {"OTHERS", "UNK"}) + ["UNK", "OTHERS"]

            # Prepare data for plotting
            proportions = {}
            for lang in sorted_languages:
                proportions[lang] = [data[layer].get(lang, 0) for layer in layers]

            if args.without_unknown == "True":
                # Remove "UNK" and "OTHERS" and normalize
                filtered_proportions = {}
                for lang, values in proportions.items():
                    if lang not in ["UNK", "OTHERS"]:
                        filtered_proportions[lang] = values

                # Normalize each layer
                normalized_proportions = {}
                for layer_idx in range(len(layers)):
                    total = sum(
                        filtered_proportions[lang][layer_idx]
                        for lang in filtered_proportions
                    )
                    for lang in filtered_proportions:
                        if lang not in normalized_proportions:
                            normalized_proportions[lang] = []
                        normalized_proportions[lang].append(
                            filtered_proportions[lang][layer_idx] / total
                            if total > 0
                            else 0
                        )

                proportions = normalized_proportions

                # Sort languages alphabetically
                sorted_languages = sorted(proportions.keys())

            # Plot stacked bar chart
            x = np.arange(len(layers))
            fig, ax = plt.subplots(figsize=(11, 8))

            bottom = np.zeros(len(layers))
            for lang in sorted_languages[::-1]:
                bar = ax.bar(
                    x,
                    proportions[lang],
                    label=lang,
                    bottom=bottom,
                    color=get_language_color(lang),
                )
                bottom += proportions[lang]

            # Customize the plot
            ax.set_title(
                f"{model_print_name} - {languages_code2name[current_language]} ({current_language})",
                fontsize=font_sizes["title"],
            )
            ax.set_xticks(x[::2])
            ax.set_xticklabels(
                [l.split("_")[0] for l in layers][::2], rotation=45, ha="right"
            )
            ax.tick_params(labelsize=font_sizes["ticks"])
            ax.set_xlabel("Layers", fontsize=font_sizes["label"])
            ax.set_ylabel("Proportion", fontsize=font_sizes["label"])
            # ax.set_title(f"Language Composition by Layer: {file.replace('_language_composition_stats.json', '')}")
            handles, labels = ax.get_legend_handles_labels()

            # Remove 'OTHERS' and 'UNK' from the legend
            handles_labels = [
                (h, l) for h, l in zip(handles, labels) if l not in {"OTHERS", "UNK"}
            ]

            # Unpack the filtered handles and labels
            handles, labels = zip(*handles_labels) if handles_labels else ([], [])

            # ax.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1, 1), title="Languages", ncols=2)
            fig.legend(
                handles=handles,
                labels=labels,
                loc="lower center",
                ncol=7,
                fontsize=font_sizes["legend"],
            )
            # ax.grid(axis="y", linestyle="--", linewidth=0.5)

            # Save the figure
            output_filename = f"{os.path.splitext(file)[0]}.png"
            if args.without_unknown == "True":
                output_filename = output_filename.replace(".png", "_normalized.png")
            # plt.tight_layout()
            plt.tight_layout(
                rect=[0, 0.2, 1, 1]
            )  # Reserve space at the bottom for the legend
            plt.savefig(output_filename)
            plt.savefig(output_filename.replace("png", "pdf"), dpi=200)
            plt.close(fig)
            print(f"Saved plot: {output_filename}")

        """
        ##################################################################
        Plot stacked bar charts averaged across all languages
        ##################################################################
        """
        # Aggregate data
        aggregate_data = defaultdict(lambda: defaultdict(list))

        # Process each file
        for file in json_files:
            with open(file, "r") as f:
                data = json.load(f)
                for layer, layer_data in data.items():
                    for lang, value in layer_data.items():
                        aggregate_data[layer][lang].append(value)

        # Calculate averages
        average_data = {
            layer: {
                lang: np.sum(values) / len(json_files)
                for lang, values in layer_data.items()
            }
            for layer, layer_data in aggregate_data.items()
        }

        # Sort languages for consistent order: "OTHERS" -> "UNK" -> alphabetical
        all_languages = set(
            lang for layer in average_data.values() for lang in layer.keys()
        )
        sorted_languages = sorted(all_languages - {"OTHERS", "UNK"}) + ["UNK", "OTHERS"]

        layers = list(average_data.keys())

        # Prepare data for plotting
        proportions = {}
        for lang in sorted_languages:
            proportions[lang] = [average_data[layer].get(lang, 0) for layer in layers]

        if args.without_unknown == "True":
            # Remove "UNK" and "OTHERS" and normalize
            filtered_proportions = {}
            for lang, values in proportions.items():
                if lang not in ["UNK", "OTHERS"]:
                    filtered_proportions[lang] = values

            # Normalize each layer
            normalized_proportions = {}
            for layer_idx in range(len(layers)):
                total = sum(
                    filtered_proportions[lang][layer_idx]
                    for lang in filtered_proportions
                )
                for lang in filtered_proportions:
                    if lang not in normalized_proportions:
                        normalized_proportions[lang] = []
                    normalized_proportions[lang].append(
                        filtered_proportions[lang][layer_idx] / total
                        if total > 0
                        else 0
                    )

            proportions = normalized_proportions

            # Sort languages alphabetically
            sorted_languages = sorted(proportions.keys())

        # Plot stacked bar chart
        x = np.arange(len(layers))
        fig, ax = plt.subplots(figsize=(12, 6))

        bottom = np.zeros(len(layers))
        for lang in sorted_languages[::-1]:
            ax.bar(
                x,
                proportions[lang],
                label=lang,
                bottom=bottom,
                color=get_language_color(lang),
            )
            bottom += proportions[lang]

        # Customize the plot
        ax.set_xticks(x[::2])
        ax.set_xticklabels(
            [l.split("_")[0] for l in layers][::2], rotation=45, ha="right"
        )
        ax.set_xlabel("Layers")
        ax.set_ylabel("Proportion (Average)")
        # ax.set_title("Average Language Composition Across All Files")
        handles, labels = ax.get_legend_handles_labels()
        # Remove 'OTHERS' and 'UNK' from the legend
        handles_labels = [
            (h, l) for h, l in zip(handles, labels) if l not in {"OTHERS", "UNK"}
        ]

        # Unpack the filtered handles and labels
        handles, labels = zip(*handles_labels) if handles_labels else ([], [])

        # ax.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1, 1), title="Languages", ncols=2)
        # ax.grid(axis="y", linestyle="--", linewidth=0.5)
        fig.legend(
            handles=handles,
            labels=labels,
            loc="lower center",
            ncol=7,
            fontsize=font_sizes["legend"],
        )

        # Save the figure
        output_path = f"results_dev/language_id_{args.model_name}/all_language_composition_stats.png"
        if args.without_unknown == "True":
            output_path = output_path.replace(".png", "_normalized.png")
        plt.tight_layout(rect=[0, 0.225, 1, 1])
        plt.savefig(output_path)
        plt.savefig(output_filename.replace("png", "pdf"), dpi=200)
        plt.close(fig)
