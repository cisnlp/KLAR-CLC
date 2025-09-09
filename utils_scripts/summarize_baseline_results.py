import json
import numpy as np


def calculate_language_avg_accuracy(json_filepath):
    # Load the JSON data
    with open(json_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Dictionary to store averaged accuracies per language
    avg_accuracy_per_language = {}

    for language, relations in data.items():
        acc_values = []
        acc_en_values = []

        for relation in relations:
            if "acc" in relations[relation]:
                acc_values.append(relations[relation]["acc"])
            if "acc_en" in relations[relation]:
                acc_en_values.append(relations[relation]["acc_en"])

        # Compute average accuracies (avoid division by zero)
        avg_acc = np.mean(acc_values) if acc_values else 0
        avg_acc_en = np.mean(acc_en_values) if acc_en_values else 0

        # Store results
        avg_accuracy_per_language[language] = {
            "avg_acc": avg_acc,
            "avg_acc_en": avg_acc_en,
        }

    output_filepath = json_filepath.replace(".json", "_summary.json")

    # Save the results to a JSON file
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(avg_accuracy_per_language, f, indent=4, ensure_ascii=False)

    print(f"Averaged accuracy per language saved to {output_filepath}")


# Example usage
calculate_language_avg_accuracy("results_dev/translation_baseline2_acc_llama2.json")
