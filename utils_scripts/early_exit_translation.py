import os
import json
import time
import argparse
from pygoogletranslation import Translator
import httpx
from tqdm import tqdm

import sys

sys.path.append(".")
sys.path.append("..")
from plot_scripts.rank_plot_en_pivot import extract_relations_languages


def translate_with_retry(text, src, dest, retries=5, delay=2):
    translator = Translator(timeout=httpx.Timeout(10.0))
    dest = "zh-cn" if dest == "zh" else dest
    for attempt in range(retries):
        try:
            return translator.translate(text, src=src, dest=dest).text
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                try:
                    return translator.translate(text, src="en", dest=dest).text
                except:
                    raise e  # Re-raise the exception if all retries fail


def compare_prediction_translation(file_path, target_language, translate=True):
    """
    Compare the translated prediction of a specified layer with the test tokenized sentence.

    Args:
        file_path (str): Path to the input JSON file.

    Returns:
        tuple: (translated_pred, expected_token, comparison_result)
    """
    # Load the JSON file
    with open(file_path, "r") as f:
        data = json.load(f)

    # Specify the layer and calculate last token
    extraction_layer = 20
    last_token = data["answer_token_span_test"][0] - 1

    # Normalize both strings: remove spaces/symbols and ignore case
    def normalize_string(s):
        return "".join(filter(str.isalnum, s)).lower()

    # Extract the prediction of the corresponding layer
    early_exit_pred = data["logit_lens_result"]["resid"][f"{extraction_layer}_pre"][
        last_token
    ]["top_token_strings"][0]
    early_exit_pred_normalized = normalize_string(early_exit_pred)

    # Get the expected token from the test sentence
    expected_token = data["test_tokenized_sentence"][last_token + 1]
    expected_token_normalized = normalize_string(expected_token)

    # Initialize Google Translate API client
    translator = Translator()

    # Translate the prediction to English
    early_exit_pred_normalized = normalize_string(early_exit_pred)
    if early_exit_pred_normalized == "" or early_exit_pred_normalized is None:
        return "", expected_token, False

    if translate:
        translated_pred = translate_with_retry(
            early_exit_pred_normalized, src="auto", dest=target_language
        )
        translated_pred_normalized = normalize_string(translated_pred)
    else:
        translated_pred = None
        translated_pred_normalized = early_exit_pred_normalized

    # Compare the normalized strings
    comparison_result = translated_pred_normalized == expected_token_normalized

    return translated_pred, expected_token, comparison_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama2")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="llm-transparency-tool-bak/backend_results_<MODEL>_fewshot_demo_new",
    )
    parser.add_argument("--translate", type=bool, default=False)
    args = parser.parse_args()

    results_dir = args.results_dir.replace("<MODEL>", args.model_name)
    relations, languages = extract_relations_languages(results_dir)
    correctness_list = ["correct", "incorrect", "others"]
    languages = languages[::-1]
    """
    ##################################################################
    Batch process all JSON files and calculate the translation accuracy
    ##################################################################
    """
    translation_acc_summary = {language: {} for language in languages}
    for language in languages:
        if args.translate:
            output_filepath = f"results_dev/early_exit_translation_acc/{args.model_name}_{language}_accuracy.json"
        else:
            output_filepath = f"results_dev/early_exit_no_translation_acc/{args.model_name}_{language}_accuracy.json"
        # if os.path.exists(output_filepath) and os.path.exists(f"results_dev/early_exit_translation_acc/translation_correct_indices_{args.model_name}/{language}_{relations[-1]}.json"):
        #     with open(output_filepath, "r") as f:
        #         acc_summary = json.load(f)
        #     translation_acc_summary[language] = acc_summary
        #     continue

        for relation in tqdm(relations):
            results_path = os.path.join(results_dir, f"{relation}_{language}")
            translation_acc_summary[language][relation] = []

            correct_indices = []
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
                    result = compare_prediction_translation(
                        filepath, language, args.translate
                    )

                    translation_acc_summary[language][relation].append(result[-1])

                    if result[-1]:
                        sample_index = int(file.split(".")[0].split("_")[-1])
                        correct_indices.append(sample_index)
            if args.translate:
                correct_indices_save_dir = f"results_dev/early_exit_translation_acc/translation_correct_indices_{args.model_name}"
            else:
                correct_indices_save_dir = f"results_dev/early_exit_no_translation_acc/correct_indices_{args.model_name}"
            os.makedirs(correct_indices_save_dir, exist_ok=True)

            with open(
                os.path.join(correct_indices_save_dir, f"{language}_{relation}.json"),
                "w",
            ) as f:
                json.dump(correct_indices, f, indent=4)

            num_correct = sum(translation_acc_summary[language][relation])
            translation_acc_summary[language][relation] = num_correct / len(
                translation_acc_summary[language][relation]
            )

        with open(output_filepath, "w") as f:
            json.dump(translation_acc_summary[language], f, indent=4)

    # Process each language and calculate the average accuracy
    for language, results in translation_acc_summary.items():
        # Exclude the "AVG" key if it exists, and calculate the mean of the remaining values
        accuracies = [value for key, value in results.items() if key != "AVG"]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        # Add the calculated average as a new key "AVG"
        translation_acc_summary[language]["AVG"] = avg_accuracy

    if args.translate:
        output_filepath = f"results_dev/early_exit_translation_acc/{args.model_name}_all_languages_accuracy.json"
    else:
        output_filepath = f"results_dev/early_exit_no_translation_acc/{args.model_name}_all_languages_accuracy.json"
    with open(output_filepath, "w") as f:
        json.dump(translation_acc_summary, f, indent=4)
