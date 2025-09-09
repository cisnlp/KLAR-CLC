import os
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tqdm import tqdm


def plot_rank_curves(
    data_idx,
    sentence,
    plot_types,
    tokenized_sentence,
    token_idxs,
    correct_answer,
    prediction,
    resid_data,
    output_dir,
    plot_token_by_token,
    plot_all_tokens,
    plot_last_n_layers,
    language=None,
):
    y_type = plot_types[0]["plot_type"].split("_")[0]

    if plot_token_by_token:
        # Plot and save individual figures for each token_idx
        for token_idx in token_idxs:
            plt.figure(figsize=(10, 6.5))

            for plot_attr in plot_types:
                x_values = []
                y_values = []

                plot_type = plot_attr["plot_type"]
                color = plot_attr["color"]
                linestyle = plot_attr["linestyle"]
                for key in resid_data:
                    # Get the specified rank type value for the current token_idx in the current key
                    for item in resid_data[key]:
                        if item["token_idx"] == token_idx and plot_type in item:
                            x_values.append(key)
                            if plot_type == f"{y_type}_topk_pred":
                                y_values.append(item[plot_type][0])
                            else:
                                y_values.append(item[plot_type])
                            break

                # Plot the data for the current rank type
                if plot_type.endswith("_topk_pred"):
                    plot_type.replace("_topk_pred", "_prediction")

                x_values = (
                    x_values[-plot_last_n_layers:]
                    if plot_last_n_layers != -1
                    else x_values
                )
                y_values = (
                    y_values[-plot_last_n_layers:]
                    if plot_last_n_layers != -1
                    else y_values
                )

                plot_label = (
                    plot_type if "label" not in plot_attr else plot_attr["label"]
                )
                plt.plot(
                    x_values,
                    y_values,
                    label=plot_label,
                    color=color,
                    linestyle=linestyle,
                )
                if y_type == "rank":
                    plt.yscale("symlog")

                if plot_type == "rank_answer":
                    plt.ylim(-0.5, None)
                    # if plot_last_n_layers == -1:
                    #     plt.gca().set_ylim(bottom=-1000)
                if plot_type == "prob_answer":
                    plt.gca().set_ylim(0, 1)
                    plt.gca().set_xlim(0, len(x_values))

                # Add star marker (*) for points where y_value == 0
                for j, y_value in enumerate(y_values):
                    if y_value == 0:
                        plt.scatter(
                            x_values[j],
                            y_value,
                            color=color,
                            linestyle=linestyle,
                            marker="*",
                        )

            plt.rcParams["font.size"] = "20"
            plt.xlabel("Layer", fontsize=20)
            plt.ylabel("Rank", fontsize=20)
            if token_idx == max(token_idxs) and plot_type == f"{y_type}_topk_pred":
                plt.title(
                    f'"{sentence.replace(correct_answer, "").strip().strip('.').strip('。')}"',
                    fontproperties=font_prop,
                    wrap=True,
                )

                # Add the correct and wrong answer on the figure
                plt.text(
                    0.15,
                    0.54,
                    f"$✓$Correct answer: {correct_answer.strip().strip('.').strip('。')}\n$✕$Wrong answer: {prediction.strip().strip('.').strip('。')}",
                    fontsize=22,
                    transform=plt.gcf().transFigure,  # Transform to figure-relative coordinates
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox=dict(facecolor="white", alpha=0.5),
                    fontproperties=font_prop,
                )
            else:
                plt.title(
                    f'"{sentence.replace(correct_answer, "").strip().strip('.').strip('。')}"',
                    fontproperties=font_prop,
                    wrap=True,
                )
            plt.legend(loc="lower left", fontsize=18)
            plt.xticks(
                ticks=list(range(len(x_values)))[::2],
                labels=[v.split("_")[0] for v in x_values[::2]],
                rotation=45,
                fontsize=16,
            )
            plt.yticks(fontsize=20)
            plt.tight_layout()

            y_type = plot_types[0]["plot_type"].split("_")[0]  # "rank" / "prob"
            save_filename = (
                os.path.join(
                    output_dir, f"{y_type}_plot_{data_idx}_token_{token_idx}.png"
                )
                if plot_last_n_layers == -1
                else os.path.join(
                    output_dir,
                    f"{y_type}_plot_{data_idx}_token_{token_idx}_last_{plot_last_n_layers//2}_layers.png",
                )
            )
            plt.savefig(save_filename)
            plt.close()

    print(f"Rank plots for data-{data_idx} have been saved successfully.")


def load_sample_data(file_path):
    """Load sample data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_relations_languages(results_dir):
    # Extract relations and languages
    relation_names = set()
    language_codes = set()

    for subfolder in os.listdir(results_dir):
        # Check if the entry is a directory
        subfolder_path = os.path.join(results_dir, subfolder)
        if os.path.isdir(subfolder_path):
            # Split the subfolder name into relation and language
            parts = subfolder.rsplit("_", 1)
            if len(parts) == 2:  # Ensure valid relation_language format
                relation, language = parts
                # Check if the 'correct' sub-subfolder exists
                if os.path.isdir(os.path.join(subfolder_path, "correct")):
                    relation_names.add(relation)
                    language_codes.add(language)

    relations = sorted(list(relation_names))
    languages = sorted(list(language_codes))

    print(f"Relation list: {relations}")
    print(f"Language list: {languages}")
    return relations, languages


def plot_average_rank_curves_all_relations(
    results_dir, plot_types, output_dir, plot_last_n_layers
):
    # Extract relations and languages
    relations, languages = extract_relations_languages(results_dir)
    y_type = plot_types[0]["plot_type"].split("_")[0]

    for language in languages:
        # Organize data by relation and language, then separate correct and incorrect samples
        data_by_language = {"correct": [], "incorrect": []}

        if language != "en" and f"{y_type}_en_answer" not in [
            plot["plot_type"] for plot in plot_types
        ]:
            plot_types.append(
                {
                    "plot_type": f"{y_type}_en_answer",
                    "label": f"{y_type}_en_correct",
                    "color": "#2ba279",
                    "linestyle": "--",
                }
            )

        for relation in relations:
            # Group samples by correctness
            for correctness in ["correct", "incorrect"]:
                result_folder = os.path.join(
                    results_dir, f"{relation}_{language}", correctness
                )
                if not os.path.exists(result_folder):
                    continue

                for filename in os.listdir(result_folder):
                    if filename.endswith("_high_rank.json"):
                        continue
                    file_path = os.path.join(result_folder, filename)
                    if os.path.isdir(file_path):
                        continue

                    data_by_language[correctness].append(file_path)

        # Process each group and calculate average rank values
        for correctness, filenames in data_by_language.items():
            if filenames == []:
                continue
            # Get the unique list of layers from the first sample to iterate over them
            sample_first = load_sample_data(filenames[0])
            layers = list(sample_first["logit_lens_result"]["resid"].keys())

            plot_types = check_plot_types(correctness, plot_types)

            # Initialize containers for average calculations
            avg_ranks = {
                plot_attr["plot_type"]: {layer_key: None for layer_key in layers}
                for plot_attr in plot_types
            }
            var_ranks = {
                plot_attr["plot_type"]: {layer_key: None for layer_key in layers}
                for plot_attr in plot_types
            }

            layer_avg_ranks = {
                layer_key: {plot_attr["plot_type"]: [] for plot_attr in plot_types}
                for layer_key in layers
            }

            for filename in tqdm(filenames):
                # Load residual data for the current sample
                data_per_sentence = load_sample_data(filename)
                resid_data = data_per_sentence["logit_lens_result"]["resid"]

                for layer_key in layers:

                    last_sentence_token = (
                        data_per_sentence["answer_token_span_test"][0] - 1
                    )
                    if data_per_sentence["subj_token_span_test"]:
                        subject_end_token = data_per_sentence["subj_token_span_test"][
                            -1
                        ]

                    # Find ranks for last_sentence_token and subject_end_token
                    for item in resid_data[layer_key]:
                        if item["token_idx"] == last_sentence_token:
                            for plot_type in layer_avg_ranks[layer_key].keys():
                                if plot_type != f"{y_type}_topk_pred":
                                    layer_avg_ranks[layer_key][plot_type].append(
                                        item[plot_type]
                                    )
                                else:
                                    layer_avg_ranks[layer_key][plot_type].append(
                                        item[plot_type][0]
                                    )

            # Compute averages for this layer across all samples
            for plot_type in layer_avg_ranks[layer_key].keys():
                for layer_key in layers:
                    avg_ranks[plot_type][layer_key] = sum(
                        layer_avg_ranks[layer_key][plot_type]
                    ) / len(layer_avg_ranks[layer_key][plot_type])

                    # Calculate variance
                    var_ranks[plot_type][layer_key] = sum(
                        (x - avg_ranks[plot_type][layer_key]) ** 2
                        for x in layer_avg_ranks[layer_key][plot_type]
                    ) / len(layer_avg_ranks[layer_key][plot_type])
            # Plot the averaged ranks
            plt.figure(figsize=(10, 6.5))
            x_values = (
                layers[-plot_last_n_layers:] if plot_last_n_layers != -1 else layers
            )
            for plot_attr in plot_types:
                plot_type = plot_attr["plot_type"]

                # Calculate average and variance values across layers
                y_values = [avg_ranks[plot_type][layer_key] for layer_key in layers]
                y_variance = [var_ranks[plot_type][layer_key] for layer_key in layers]

                y_values = (
                    y_values[-plot_last_n_layers:]
                    if plot_last_n_layers != -1
                    else y_values
                )
                y_variance = (
                    y_variance[-plot_last_n_layers:]
                    if plot_last_n_layers != -1
                    else y_variance
                )

                # Plot the average curve
                plot_label = (
                    plot_type if "label" not in plot_attr else plot_attr["label"]
                )
                plt.plot(
                    x_values,
                    y_values,
                    label=plot_label,
                    color=plot_attr["color"],
                    linestyle=plot_attr["linestyle"],
                )
                # Add star marker (*) for points where y_value == 0
                for j, y_value in enumerate(y_values):
                    if y_value == 0:
                        plt.scatter(
                            x_values[j],
                            y_value,
                            color=plot_attr["color"],
                            linestyle=plot_attr["linestyle"],
                            marker="*",
                        )

                # Plot variance as a shaded region (mean ± sqrt(variance))
                plt.fill_between(
                    x_values,
                    [y - (v**0.5) for y, v in zip(y_values, y_variance)],
                    [y + (v**0.5) for y, v in zip(y_values, y_variance)],
                    color=plot_attr["color"],
                    alpha=0.1,
                )

                if y_type == "rank":
                    plt.yscale("symlog")

                if plot_type == "rank_answer":
                    plt.ylim(-0.5, None)
                    # if plot_last_n_layers == -1:
                    #     plt.gca().set_ylim(bottom=-1000)
                    # elif y_variance[0] > 1000:
                    #     plt.gca().set_ylim(-1000, 1000)
                if plot_type == "prob_answer":
                    plt.gca().set_ylim(0, 1)
                    plt.gca().set_xlim(0, len(x_values))

            plt.rcParams["font.size"] = "20"
            plt.xlabel("Layer", fontsize=20)
            plt.ylabel("Rank", fontsize=20)
            plt.title(
                f"Average Ranks for all relations: Language: {language} ({correctness})",
                fontproperties=font_prop,
                wrap=True,
            )
            plt.legend(loc="lower left", fontsize=18)
            plt.xticks(
                ticks=list(range(len(x_values)))[::2],
                labels=[v.split("_")[0] for v in x_values[::2]],
                rotation=45,
                fontsize=16,
            )
            plt.yticks(fontsize=20)
            plt.tight_layout()

            # Save the plot
            os.makedirs(os.path.join(output_dir, language), exist_ok=True)
            save_filename = (
                os.path.join(
                    output_dir, language, f"avg_{y_type}_{language}_{correctness}.png"
                )
                if plot_last_n_layers == -1
                else os.path.join(
                    output_dir,
                    language,
                    f"avg_rank_{language}_{correctness}_{plot_last_n_layers//2}_layers.png",
                )
            )
            plt.savefig(save_filename)
            plt.close()

        print("Average rank plots have been saved successfully.")


def plot_average_rank_curves(results_dir, plot_types, output_dir, plot_last_n_layers):
    # Organize data by relation and language, then separate correct and incorrect samples
    data_by_relation_language = {"correct": [], "incorrect": []}
    avg_ranks_all = {}

    # Group samples by relation, language, and correctness
    for correctness in ["correct", "incorrect"]:
        result_folder = os.path.join(results_dir, correctness)
        if not os.path.exists(result_folder):
            continue
        for filename in os.listdir(result_folder):
            if filename.endswith("_high_rank.json"):
                continue
            file_path = os.path.join(result_folder, filename)
            if os.path.isdir(file_path):
                continue

            data_by_relation_language[correctness].append(file_path)
    # Process each group and calculate average rank values
    for correctness, filenames in data_by_relation_language.items():
        if filenames == []:
            continue
        # Get the unique list of layers from the first sample to iterate over them
        sample_first = load_sample_data(filenames[0])
        layers = list(sample_first["logit_lens_result"]["resid"].keys())

        plot_types = check_plot_types(correctness, plot_types)

        # Initialize containers for average calculations
        avg_ranks = {
            plot_attr["plot_type"]: {layer_key: None for layer_key in layers}
            for plot_attr in plot_types
        }
        var_ranks = {
            plot_attr["plot_type"]: {layer_key: None for layer_key in layers}
            for plot_attr in plot_types
        }

        layer_avg_ranks = {
            layer_key: {plot_attr["plot_type"]: [] for plot_attr in plot_types}
            for layer_key in layers
        }

        for filename in tqdm(filenames):
            # Load residual data for the current sample
            data_per_sentence = load_sample_data(filename)
            resid_data = data_per_sentence["logit_lens_result"]["resid"]

            for layer_key in layers:

                last_sentence_token = data_per_sentence["answer_token_span_test"][0] - 1
                if data_per_sentence["subj_token_span_test"]:
                    subject_end_token = data_per_sentence["subj_token_span_test"][-1]

                # Find ranks for last_sentence_token and subject_end_token
                for item in resid_data[layer_key]:
                    if item["token_idx"] == last_sentence_token:
                        for plot_type in layer_avg_ranks[layer_key].keys():
                            if plot_type != f"{y_type}_topk_pred":
                                layer_avg_ranks[layer_key][plot_type].append(
                                    item[plot_type]
                                )
                            else:
                                layer_avg_ranks[layer_key][plot_type].append(
                                    item[plot_type][0]
                                )

        # Compute averages for this layer across all samples
        for plot_type in layer_avg_ranks[layer_key].keys():
            for layer_key in layers:
                avg_ranks[plot_type][layer_key] = sum(
                    layer_avg_ranks[layer_key][plot_type]
                ) / len(layer_avg_ranks[layer_key][plot_type])

                # Calculate variance
                var_ranks[plot_type][layer_key] = sum(
                    (x - avg_ranks[plot_type][layer_key]) ** 2
                    for x in layer_avg_ranks[layer_key][plot_type]
                ) / len(layer_avg_ranks[layer_key][plot_type])
        # Plot the averaged ranks
        plt.figure(figsize=(10, 6.5))
        x_values = layers[-plot_last_n_layers:] if plot_last_n_layers != -1 else layers
        for plot_attr in plot_types:
            plot_type = plot_attr["plot_type"]

            # Calculate average and variance values across layers
            y_values = [avg_ranks[plot_type][layer_key] for layer_key in layers]
            y_variance = [var_ranks[plot_type][layer_key] for layer_key in layers]

            y_values = (
                y_values[-plot_last_n_layers:] if plot_last_n_layers != -1 else y_values
            )
            y_variance = (
                y_variance[-plot_last_n_layers:]
                if plot_last_n_layers != -1
                else y_variance
            )

            # Plot the average curve
            plot_label = plot_type if "label" not in plot_attr else plot_attr["label"]
            plt.plot(
                x_values,
                y_values,
                label=plot_label,
                color=plot_attr["color"],
                linestyle=plot_attr["linestyle"],
            )

            # Add star marker (*) for points where y_value == 0
            for j, y_value in enumerate(y_values):
                if y_value == 0:
                    plt.scatter(
                        x_values[j],
                        y_value,
                        color=plot_attr["color"],
                        linestyle=plot_attr["linestyle"],
                        marker="*",
                    )

            # Plot variance as a shaded region (mean ± sqrt(variance))
            plt.fill_between(
                x_values,
                [y - (v**0.5) for y, v in zip(y_values, y_variance)],
                [y + (v**0.5) for y, v in zip(y_values, y_variance)],
                color=plot_attr["color"],
                alpha=0.1,
            )

            if y_type == "rank":
                plt.yscale("symlog")

            if plot_type == "rank_answer":
                plt.ylim(-0.5, None)
                # if plot_last_n_layers == -1:
                #     plt.gca().set_ylim(bottom=-1000)
                # elif y_variance[0] > 1000:
                #     plt.gca().set_ylim(-1000, 1000)
            if plot_type == "prob_answer":
                plt.gca().set_ylim(0, 1)
                plt.gca().set_xlim(0, len(x_values))

        plt.rcParams["font.size"] = "20"
        plt.xlabel("Layer", fontsize=20)
        plt.ylabel("Rank", fontsize=20)
        plt.title(
            f"Average Ranks for Relation: {relation}, Language: {language} ({correctness})",
            fontproperties=font_prop,
            wrap=True,
        )
        plt.legend(loc="lower left", fontsize=18)
        plt.xticks(
            ticks=list(range(len(x_values)))[::2],
            labels=[v.split("_")[0] for v in x_values[::2]],
            rotation=45,
            fontsize=16,
        )
        plt.yticks(fontsize=20)
        plt.tight_layout()

        # Save the plot
        save_dir = os.path.join(output_dir, f"{correctness}/plots")
        save_filename = (
            os.path.join(
                save_dir, f"avg_{y_type}_{relation}_{language}_{correctness}.png"
            )
            if plot_last_n_layers == -1
            else os.path.join(
                save_dir,
                f"avg_rank_{relation}_{language}_{correctness}_last_{plot_last_n_layers//2}_layers.png",
            )
        )
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_filename)
        plt.close()

    print("Average rank plots have been saved successfully.")


def check_plot_types(correctness, plot_types):
    y_type = plot_types[0]["plot_type"].split("_")[0]
    if correctness == "incorrect" and f"{y_type}_topk_pred" not in [
        plot["plot_type"] for plot in plot_types
    ]:
        plot_types.append(
            {
                "plot_type": f"{y_type}_topk_pred",
                "label": f"{y_type}_wrong",
                "color": "#d5697c",
                "linestyle": "-",
            }
        )
    if correctness == "correct" and f"{y_type}_topk_pred" in [
        plot["plot_type"] for plot in plot_types
    ]:
        plot_types.remove(
            {
                "plot_type": f"{y_type}_topk_pred",
                "label": f"{y_type}_wrong",
                "color": "#d5697c",
                "linestyle": "-",
            }
        )

    # Remove duplicates
    seen = set()
    unique_list = []

    for d in plot_types:
        # Convert dictionary to a tuple of items (hashable type)
        tuple_representation = tuple(sorted(d.items()))
        if tuple_representation not in seen:
            seen.add(tuple_representation)
            unique_list.append(d)

    return unique_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default="llm-transparency-tool-bak/backend_results_llama2_fewshot_demo",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        type=str,
        default=[
            "zh",
            "ca",
            "en",
            "es",
            "fr",
            "hu",
            "ja",
            "ko",
            "nl",
            "ru",
            "uk",
            "vi",
            "zh",
        ],
    )
    parser.add_argument(
        "--relations", nargs="+", type=str, default=["language_of_work_or_name_1"]
    )
    parser.add_argument("--y_type", type=str, default="rank")
    parser.add_argument("--plot_token_by_token", type=bool, default=True)
    parser.add_argument("--plot_key_tokens_only", type=bool, default=True)
    parser.add_argument("--plot_all_tokens", type=bool, default=False)
    parser.add_argument("--plot_last_n_layers", type=int, default=-1)
    parser.add_argument("--plot_individual_relations", type=str, default="True")
    args = parser.parse_args()

    y_type = args.y_type
    plot_last_n_layers = args.plot_last_n_layers * 2 + 1

    # Define the list of rank types to plot
    plot_types = [
        {
            "plot_type": f"{y_type}_answer",
            "label": f"{y_type}_correct",
            "color": "#2ba279",
            "linestyle": "-",
        },
        # {
        #     'plot_type': f'{y_type}_subject',
        #     'color': 'tab:blue',
        #     'linestyle': '-'
        # },
    ]

    if args.plot_individual_relations == "False":
        plot_average_rank_curves_all_relations(
            args.results_dir,
            plot_types,
            os.path.join(args.results_dir, "plots_all_relations"),
            plot_last_n_layers,
        )
    else:
        for language in args.languages:
            font_paths = {
                "zh": "~/.local/share/fonts/SimSun.ttf",
                "ja": "~/.local/share/fonts/NotoSansJP-Regular.otf",
                "ko": "~/.local/share/fonts/NotoSansKR-Regular.otf",
            }
            if language in ["zh", "ja", "ko"]:
                font_path = font_paths[language]  # Use the path to the downloaded font
                font_prop = FontProperties(
                    fname=font_path, size=20, math_fontfamily="dejavusans"
                )
            else:
                font_prop = None

            if language != "en":
                plot_types.append(
                    {
                        "plot_type": f"{y_type}_en_answer",
                        "label": f"{y_type}_en_correct",
                        "color": "#2ba279",
                        "linestyle": "--",
                    }
                )
                # plot_types.append(
                #     {
                #         'plot_type': f'{y_type}_en_subject',
                #         'color': 'tab:blue',
                #         'linestyle': '--'
                #     }
                # )

            for relation in args.relations:
                # Plot the rank curve across all samples
                results_dir = f"{args.results_dir}/{relation}_{language}/"
                output_dir = f"{args.results_dir}/{relation}_{language}/"
                plot_average_rank_curves(
                    results_dir, plot_types, output_dir, plot_last_n_layers
                )

                correctness_options = ["correct", "incorrect"]
                correctness_options = ["incorrect"]
                for correctness in correctness_options:
                    result_folder = os.path.join(
                        args.results_dir, f"{relation}_{language}", correctness
                    )
                    if not os.path.exists(result_folder):
                        continue
                    for filename in tqdm(sorted(os.listdir(result_folder))):
                        if filename.endswith("_high_rank.json"):
                            continue
                        file_path = os.path.join(result_folder, filename)
                        if os.path.isdir(file_path):
                            continue

                        with open(file_path, "r", encoding="utf-8") as f:
                            data_per_sentence = json.load(f)

                        # TODO: restructure the following as a function and also plot the relation-wise curve
                        data_idx = data_per_sentence["data_idx"]
                        # Extract the sentence
                        sentence = data_per_sentence["test_sentence"]
                        tokenized_sentence = data_per_sentence[
                            "test_tokenized_sentence"
                        ]
                        correct_answer = data_per_sentence["answer_strings"]
                        prediction = data_per_sentence["pred_answer_strings"]
                        # Extract the 'resid' data
                        resid_data = data_per_sentence["logit_lens_result"]["resid"]

                        # List tokens to be plotted
                        last_sentence_token = (
                            data_per_sentence["answer_token_span_test"][0] - 1
                        )
                        if not args.plot_key_tokens_only:
                            token_idxs = list(range(last_sentence_token))
                        else:
                            token_idxs = [last_sentence_token]

                            if (
                                data_per_sentence["subj_token_span_test"]
                                and data_per_sentence["subj_token_span_test"][-1]
                                < last_sentence_token
                            ):
                                token_idxs.append(
                                    data_per_sentence["subj_token_span_test"][-1]
                                )

                        plot_types = check_plot_types(correctness, plot_types)

                        output_dir = f"{args.results_dir}/{relation}_{language}/{correctness}/plots/{data_idx}/"

                        os.makedirs(output_dir, exist_ok=True)

                        plot_rank_curves(
                            data_idx,
                            sentence,
                            plot_types,
                            tokenized_sentence,
                            token_idxs,
                            correct_answer,
                            prediction,
                            resid_data,
                            output_dir,
                            args.plot_token_by_token,
                            args.plot_all_tokens,
                            plot_last_n_layers,
                            language=language,
                        )
