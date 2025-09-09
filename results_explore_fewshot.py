import os
import json
import pickle
import torch
import argparse
from tqdm import tqdm


def load_file(filename):
    with open(filename, "rb") as file:
        if filename.endswith(".pkl"):
            data = pickle.load(file)
        else:
            data = json.load(file)
    return data


def remove_answer_token_items(data, answer_start_token):
    filtered = {
        "resid": {},
        "output": {},
        "heads": {},
        "neurons": data["neurons"] if "neurons" in data else None,
    }

    for key in ["resid", "output", "heads"]:
        # for key in ['resid', 'output']:
        if data[key] is None:
            continue
        for k, v in data[key].items():
            token_specifics = v
            filtered[key][k] = []
            for token_dict in token_specifics:
                if token_dict["token_idx"] < answer_start_token:
                    filtered[key][k].append(token_dict)
    return filtered


def process_pkl_results(
    results_dir,
    relation,
    language,
    convert_and_save_json=True,
    no_answer_token_items=True,
    filter_high_rank_states=False,
):
    if convert_and_save_json:
        source_filename = (
            f"{results_dir}/{relation}_{language}/{relation}_{language}.pkl"
        )
        data = load_file(source_filename)

        # Convert any tensor in the dictionary to a list
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_tensors(v) for v in obj]
            return obj

        data_filtered = []
        data_filtered = {"correct": [], "incorrect": [], "other": []}
        for data_per_sentence in tqdm(data):
            data_per_sentence_filtered = {
                key: None
                for key in data_per_sentence.keys()
                if key
                not in [
                    "contributions",
                    "neuron_contributions_all",
                    "neuron_contributions",
                ]
            }
            answer_start_token = data_per_sentence["answer_token_span_test"][0]
            for k, v in data_per_sentence.items():
                if k in data_per_sentence_filtered.keys():
                    if k == "logit_lens_result" and no_answer_token_items:
                        data_per_sentence_filtered[k] = remove_answer_token_items(
                            v, answer_start_token
                        )
                    else:
                        data_per_sentence_filtered[k] = v
            correctness = data_per_sentence["correctness"]
            data_filtered[correctness].append(data_per_sentence_filtered)

            output_dir = os.path.join(os.path.dirname(source_filename), correctness)
            os.makedirs(output_dir, exist_ok=True)
            save_filename = os.path.basename(source_filename).replace(
                ".pkl", f'_{data_per_sentence_filtered["data_idx"]}.json'
            )
            with open(
                os.path.join(output_dir, save_filename), "w", encoding="utf-8"
            ) as json_file:
                json.dump(
                    convert_tensors(data_per_sentence_filtered),
                    json_file,
                    ensure_ascii=False,
                    indent=4,
                )

            print(
                f'Data for sentence-{data_per_sentence_filtered["data_idx"]} saved in {os.path.join(output_dir, save_filename)}'
            )

    if filter_high_rank_states:
        # Recursive function to filter items based on 'rank_answer'
        def filter_items(data):
            if isinstance(data, dict):
                filtered_data = {}
                # token_idx_to_keep = set()
                token_idx_to_keep = []

                # First pass: Collect token_idx values where rank_answer < 1
                for key, value in data.items():
                    if isinstance(value, dict) and "rank_answer" in value:
                        if value["rank_answer"] < 1:
                            if "token_idx" in value:
                                token_idx_to_keep.append(value["token_idx"])

                # Second pass: Filter items based on the collected token_idx values
                for key, value in data.items():
                    if isinstance(value, dict) and "rank_answer" in value:
                        if value["rank_answer"] < 1 or (
                            "token_idx" in value
                            and value["token_idx"] in token_idx_to_keep
                        ):
                            filtered_data[key] = value
                    elif isinstance(value, dict) or isinstance(value, list):
                        # Recursively filter nested structures
                        filtered_data[key] = filter_items(value)
                    else:
                        # Preserve non-dict items as they are
                        filtered_data[key] = value

                return filtered_data

            elif isinstance(data, list):
                # Apply filtering to each item in the list and remove None values
                return [
                    filter_items(item)
                    for item in data
                    if filter_items(item) is not None
                ]

            else:
                return data

        # Function to filter out dictionaries with 'rank_answer' > 10
        def filter_rank_answer(data):
            filtered = {}
            for data_k, data_v in data.items():
                if data_k != "logit_lens_result":
                    filtered[data_k] = data_v

            filtered["logit_lens_result"] = {
                "resid": {},
                "output": {},
                "heads": {},
                "neurons": data["logit_lens_result"]["neurons"],
            }

            answer_start_token = data["answer_token_span_test"][0]
            for key in ["resid", "output", "heads"]:
                # for key in ['resid', 'output']:
                for k, v in data["logit_lens_result"][key].items():
                    token_specifics = v
                    for token_dict in token_specifics:
                        # if token_dict['rank_answer'] < 1 and token_dict['token_idx'] < answer_start_token:
                        if (
                            token_dict["rank_answer"] < 1
                            and token_dict["token_idx"] == answer_start_token - 1
                        ):
                            filtered["logit_lens_result"][key][k] = []
                            filtered["logit_lens_result"][key][k].append(token_dict)
            return filtered

        input_dirs = [
            f"{results_dir}/{relation}_{language}/correct",
            f"{results_dir}/{relation}_{language}/incorrect",
            f"{results_dir}/{relation}_{language}/other",
        ]

        for input_dir in input_dirs:
            if not os.path.exists(input_dir):
                continue

            source_filename_list = [
                file
                for file in os.listdir(input_dir)
                if file.startswith(f"{relation}_")
                and file.endswith(".json")
                and not file.endswith("_high_rank.json")
            ]

            for filename in sorted(source_filename_list):
                source_filename = os.path.join(input_dir, filename)
                data = load_file(source_filename)
                # Apply the filtering function
                filtered_data = filter_items(data)  # Filter 'neurons'
                filtered_data = filter_rank_answer(
                    filtered_data
                )  # Filter 'resid', 'output', 'heads', 'neurons'

                if "incorrect" in input_dir:
                    # Function to filter items based on presence of wrong_pred in 'top_token_strings'
                    def filter_items_with_wrong_pred(
                        data_section, wrong_pred, answer_start_token
                    ):
                        filtered_data_section = {}
                        for k, v in data_section.items():
                            if (
                                wrong_pred
                                in v[answer_start_token - 1]["top_token_strings"]
                                and v[answer_start_token - 1][
                                    "top_token_strings"
                                ].index(wrong_pred)
                                == 0
                            ):
                                filtered_data_section[k] = [v[answer_start_token - 1]]
                        return filtered_data_section

                    components = ["resid", "output"]
                    filtered_data["logit_lens_wrong"] = {
                        component: {} for component in components
                    }
                    answer_start_token = data["answer_token_span_test"][0]
                    top_token_strings = data["logit_lens_result"]["resid"][
                        "final_post"
                    ][answer_start_token - 1]["top_token_strings"]
                    for t in top_token_strings:
                        if t != "":
                            wrong_pred = t
                            break

                    # Filter items where the wrong_pred exists in the 'top_token_strings'
                    if wrong_pred:
                        for component in components:
                            filtered_data["logit_lens_wrong"][component] = (
                                filter_items_with_wrong_pred(
                                    data["logit_lens_result"][component],
                                    wrong_pred,
                                    answer_start_token,
                                )
                            )

                # Save the filtered data back to a JSON file
                with open(
                    source_filename.replace(".json", "_high_rank.json"),
                    "w",
                    encoding="utf-8",
                ) as json_file:
                    json.dump(filtered_data, json_file, ensure_ascii=False, indent=4)

                print(
                    f"Filtered JSON data saved to '{source_filename.replace('.json', '_high_rank.json')}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default="llm-transparency-tool-bak/backend_results_bloom_fewshot_demo_baseline_ar",
    )
    parser.add_argument("--convert_and_save_json", type=bool, default=True)
    parser.add_argument("--no_answer_token_items", type=bool, default=True)
    parser.add_argument("--filter_high_rank_states", type=bool, default=True)
    parser.add_argument('--languages', nargs="+", type=str, default=['ar', 'ca', 'en', 'es', 'fr', 'vi', 'zh'])
    parser.add_argument("--relations", nargs="+", type=str, default=["capital"])
    args = parser.parse_args()

    for relation in args.relations:
        for language in args.languages:
            process_pkl_results(
                args.results_dir,
                relation,
                language,
                args.convert_and_save_json,
                args.no_answer_token_items,
                args.filter_high_rank_states,
            )
