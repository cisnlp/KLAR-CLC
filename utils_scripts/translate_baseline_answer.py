import json
import os
import time
import argparse
from pygoogletranslation import Translator
from tqdm import tqdm


def translate_text(text, translator, target_lang="ar", max_retries=5, base_delay=2):
    """
    Translates a given text to the specified target language using Google Translate.
    Implements a retry mechanism with exponential backoff in case of failure.

    Args:
        text (str): The text to be translated.
        translator (google_translator): Google Translator instance.
        target_lang (str): Target language code (default: "ar" for Arabic).
        max_retries (int): Maximum number of retries.
        base_delay (int): Base delay in seconds for exponential backoff.

    Returns:
        str: Translated text or original text if translation fails.
    """
    retries = 0
    while retries < max_retries:
        try:
            translation = translator.translate(text, src="en", dest=target_lang)
            return translation if translation else text
        except Exception as e:
            print(
                f"Error translating '{text}': {e}. Retrying {retries + 1}/{max_retries}..."
            )
            retries += 1
            time.sleep(base_delay * (2 ** (retries - 1)))  # Exponential backoff

    print(f"Failed to translate '{text}' after {max_retries} retries. Skipping.")
    return text  # Return the original text if all retries fail


def translate_json(input_file, target_lang="ar", max_retries=5):
    """
    Extracts the predicted answer from a JSON file and translates it to the target language.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to save the translated JSON file.
        target_lang (str): Target language code (default: "ar" for Arabic).
        max_retries (int): Maximum number of retries for failed translations.

    Returns:
        None
    """
    translator = Translator()

    # Load JSON data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    data_idx = data["data_idx"]
    # Translate the predicted answer
    if "pred_answer_strings" in data and isinstance(data["pred_answer_strings"], str):
        pred_answer_strings_translated = translate_text(
            data["pred_answer_strings"].strip().strip("▁").strip(".").strip("。"),
            translator,
            target_lang,
            max_retries,
        )

    return data_idx, pred_answer_strings_translated


def batch_translate(folder_path, output_folder, target_lang="ar", max_retries=5):
    """
    Translates predicted answers in all JSON files within a folder.

    Args:
        folder_path (str): Path to the folder containing JSON files.
        output_folder (str): Path to save the translated JSON files.
        target_lang (str): Target language code (default: "ar" for Arabic).
        max_retries (int): Maximum number of retries for failed translations.

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            input_file = os.path.join(folder_path, filename)
            output_file = os.path.join(output_folder, filename)
            translate_json(input_file, output_file, target_lang, max_retries)

    print(f"Batch translation completed. Translated files are in {output_folder}")


def find_subject_object(json_file, index):
    """
    Find the subject and object for a given index in the JSON file.

    Args:
        json_file (str): Path to the JSON file.
        index (int): The index to search for.

    Returns:
        tuple: (subject, object) if found, otherwise (None, None).
    """
    # Load JSON file
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Search for the given index
    for sample in data["samples"]:
        if sample["index"] == index:
            return sample["subject"], sample["object"]

    # Return None if index is not found
    return None, None


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bloom")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="llm-transparency-tool-bak/backend_results_MODEL_fewshot_demo_baseline_LANG",
    )
    parser.add_argument(
        "--languages", nargs="+", type=str, default=["es", "fr", "vi", "zh"]
    )
    parser.add_argument(
        "--relations", nargs="+", type=str, default=["official_language"]
    )
    args = parser.parse_args()

    results_dir = args.results_dir.replace("MODEL", args.model_name)
    correctness_list = ["correct", "incorrect", "others"]
    accuracy_summary = {}
    for language in args.languages:
        accuracy_summary[language] = {}
        for relation in args.relations:
            correct_indices = []
            correct_en_indices = []

            accuracy_summary[language][relation] = {}
            count, count_en_correct, count_correct = 0, 0, 0

            for correctness in correctness_list:
                results_dir = results_dir.replace("LANG", language)
                result_path = os.path.join(results_dir, f"{relation}_en", correctness)
                reference_path = os.path.join(
                    "klar", language, f"{relation}.json"
                )

                if not os.path.exists(result_path):
                    continue
                for filepath in tqdm(os.listdir(result_path)):
                    if not filepath.endswith("_high_rank.json"):
                        continue

                    target_lang = language if language != "zh" else "zh-CN"
                    data_idx, pred_answer_strings_translated = translate_json(
                        os.path.join(result_path, filepath), target_lang=target_lang
                    )
                    subject, object = find_subject_object(reference_path, data_idx)

                    count += 1
                    if correctness == "correct":
                        count_en_correct += 1
                        correct_en_indices.append(data_idx)
                    try:
                        if pred_answer_strings_translated.text == object:
                            count_correct += 1
                            correct_indices.append(data_idx)
                    except:
                        if pred_answer_strings_translated == object:
                            count_correct += 1
                            correct_indices.append(data_idx)

                os.makedirs("results_dev/translation_baseline_indices", exist_ok=True)
                with open(
                    f"results_dev/translation_baseline_indices/{args.model_name}_{relation}_{language}_en.json",
                    "w",
                ) as f:
                    json.dump(correct_en_indices, f, indent=4)

                with open(
                    f"results_dev/translation_baseline_indices/{args.model_name}_{relation}_{language}.json",
                    "w",
                ) as f:
                    json.dump(correct_indices, f, indent=4)

            accuracy_summary[language][relation] = {
                "acc": count_correct / count if count != 0 else None,
                "acc_en": count_en_correct / count if count != 0 else None,
                "count_correct": count_correct,
                "count_en_correct": count_en_correct,
                "count_all": count,
            }

            with open(
                f"results_dev/translation_baseline_acc_{args.model_name}.json", "w"
            ) as f:
                json.dump(accuracy_summary, f, indent=4)
