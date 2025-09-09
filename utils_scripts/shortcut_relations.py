"""
Please run this script with the repo [relations](https://github.com/evandez/relations)
This script should be replaced in 'demo/'
"""
import os
import sys
sys.path.append('.')
sys.path.append('..')
import json
import argparse
import torch
import numpy as np
from src import models, data, lens, functional
from src.utils import experiment_utils
from baukit import Menu, show


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="meta-llama/Llama-2-7b-hf", help='Model path to load')
    parser.add_argument('--dataset_path', type=str, default="klar", help='Dataset to use')
    parser.add_argument('--output_dir', type=str, default="results_dir_shortcut_debug", help='Directory to save the results')
    parser.add_argument('--language', type=str, default="ko", help='Factual probing in which language')
    parser.add_argument('--relation', type=str, default="languages_spoken", help='Specific relation from Category')
    parser.add_argument('--mode', type=str, default="process", choices=["process", "summarize"], help='process or summarize results.')
    parser.add_argument('--rewrite', type=str, default="rewrite", choices=["rewrite", "keep"], help='rewrite or keep existing results')
    parser.add_argument('--ratio_train_samples', type=float, default=0.75, help='Split ratio for training samples')
    parser.add_argument('--max_train_samples', type=int, default=25, help='Maximum training samples')
    parser.add_argument('--n_icl_examples', type=int, default=3, help='Maximum training samples')
    parser.add_argument('--layer_h', type=int, default=30, help='The layer of hidden representation as the computation input')
    parser.add_argument('--beta', type=float, default=2.25, help='Scope coefficient for linear approximation')
    parser.add_argument('--shortcut_last_token', type=bool, default=True, help='Scope coefficient for linear approximation')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    args = parser.parse_args()
    
    model_path = args.model_path
    language = args.language
    dataset_path = os.path.join(args.dataset_path, language)
    relation = args.relation
    mode = args.mode
    ratio_train_samples = args.ratio_train_samples
    max_train_samples = args.max_train_samples
    n_icl_examples = args.n_icl_examples
    shortcut = args.shortcut_last_token
    layer = args.layer_h
    beta = args.beta
    seed = args.seed

    experiment_utils.set_seed(seed) # set seed to a constant value for sampling consistency
    
    """
    --------------------------------------------
    Section 1 - Approximate the weights and bias
    -------------------------------------------- 
    """

    ###############################################
    ### Load the dataset ###
    ###############################################

    dataset = data.load_dataset(dataset_path)
    relation_names = [r.name for r in dataset.relations]
    
    if mode == "process":
        ###############################################
        ### Load the model ###
        ###############################################
        device = "cuda:0"
        mt = models.load_model(model_path, device=device, fp16=True)
        print(f"dtype: {mt.model.dtype}, device: {mt.model.device}, memory: {mt.model.get_memory_footprint()}")
        
        ###############################################
        ### Load the relation ###
        ###############################################
        relation_name = f'{language}_{relation}'
        
        output_dir = f"{args.output_dir}/{relation_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, f"train_size_{args.max_train_samples}_ratio_{args.ratio_train_samples}_layer_{layer}_beta_{str(beta).replace('.', '-')}.json")

        assert relation_name in relation_names
        relation = dataset.filter(relation_names=[relation_name])[0]
        print("------------------------------------------------------")
        print(f"Processing {relation_name}...")
        print("------------------------------------------------------")
        
        ################### make prompts ###################
        samples = relation.samples
        prompt_template = relation.prompt_templates[0]
        prompt_template = prompt_template.replace("<subject>", "{}")
        prompt_template = prompt_template.split("<mask>")[0].strip()
        language = relation.name.split("_")[0]
        
        prompt_template_icl = functional.make_prompt(
            mt=mt,
            prompt_template=prompt_template,
            examples=samples,
            subject="{}",
            n_icl_examples=n_icl_examples,
            no_entity_space=language in ["ja", "zh"],
            language=language
        )
        
        print(f"Filter prompt_template:\n{prompt_template_icl}")
        
        ################### filter samples (1) - all ###################
        all_samples = relation.samples
        n_all_samples = len(all_samples)
        relation = functional.filter_relation_samples_based_on_provided_fewshots(
            mt=mt, test_relation=relation, prompt_template=prompt_template_icl, batch_size=4
        )
        n_filtered_samples = len(relation.samples)
        wrong_samples = [sample for sample in all_samples if sample not in relation.samples]
        
        if n_filtered_samples <= n_icl_examples+1:
            print(f"[{relation.name}] has fewer than {n_icl_examples+1} correct samples, skipping...")
            raise ValueError("Number of filtered correct samples fewer than required!")
        
        print(f"\n{relation.name} -- {n_filtered_samples}/{n_all_samples} correct samples")
        pre_accuracy = n_filtered_samples/n_all_samples
        print(f"Original accuracy: {pre_accuracy}")
        print("------------------------------------------------------")
        
        train_size = min(int(ratio_train_samples * n_filtered_samples), max_train_samples) if max_train_samples > 0 else int(ratio_train_samples * n_filtered_samples)
        train_size = max(train_size, n_icl_examples+1)
        train, test = relation.split(train_size)
        print(f"training sample examples:")
        print("\n".join([sample.__str__() for sample in train.samples[:5]]))
        print(f"training sample size: {train_size}")
        test_size = len(test.samples)
        print(f"testing sample size: {test_size}")
        
        if args.rewrite == "rewrite" and os.path.exists(output_filepath):
            with open(output_filepath, "r") as f:
                results_summary = json.load(f)
            if "results_samples_en_correct" in results_summary:
                print(f"Results exist in: {output_filepath}")
                print(f"Results exist for {relation_name}, skipping...")
        else:
            ################## filter samples (2) - english pivot ###################
            if language == "english":
                pass
            else:
                indices_filepath = f"klar_filtered_indices/en/{'_'.join(relation_name.split('_')[1:])}.json"
                with open(indices_filepath, 'r') as f:
                    sample_indices_en_correct = json.load(f)
                all_samples_en_correct = [sample for sample in all_samples if sample.index in sample_indices_en_correct]
                filtered_samples_en_correct = [sample for sample in relation.samples if sample.index in sample_indices_en_correct]
                wrong_samples_en_correct = [sample for sample in wrong_samples if sample.index in sample_indices_en_correct]
                
                samples_en_correct_stats = {
                    "n_samples_en_correct": len(all_samples_en_correct),
                    "n_correct_samples_en_correct": len(filtered_samples_en_correct),
                    "n_wrong_samples_en_correct": len(wrong_samples_en_correct),
                }
                pre_accuracy_en_correct = len(filtered_samples_en_correct)/len(all_samples_en_correct)
                print(f"Original accuracy on English correct samples: {pre_accuracy_en_correct}")
                
                # Metrics and indices list for cross-lingual consistency calculation
                samples_union_correct = [sample for sample in all_samples if sample.index in sample_indices_en_correct or sample in relation.samples]
                pre_clc_en = len(filtered_samples_en_correct) / len(samples_union_correct)

                print(f"Original cross-lingual consistency to English: {pre_clc_en}")
                print("------------------------------------------------------")

            ###############################################
            ### Approximate the weights and bias ###
            ###############################################
            if shortcut:
                from src.operators_shortcut import JacobianIclMeanEstimator
            else:
                from src.operators import JacobianIclMeanEstimator
                
                
            estimator = JacobianIclMeanEstimator(
                prompt_template = prompt_template,
                mt = mt, 
                h_layer = layer,
                beta = beta,
                shortcut=shortcut
            )
            operator = estimator(
                relation.set(
                    samples=train.samples, 
                )
            )
            print("Test prompt_template:\n" + operator.prompt_template)


            """
            --------------------------------------------
            Section 2 - Checking Faithfulness
            --------------------------------------------
            """

            # Prepare the test samples
            try:
                sample = test.samples[0]
                print(sample)
                print(operator(subject = sample.subject).predictions)
            except:
                pass


            # Prepare the hidden representations
            hs_and_zs = functional.compute_hs_and_zs(
                mt = mt,
                prompt_template = operator.prompt_template,
                subjects = [sample.subject],
                h_layer= operator.h_layer,
                shortcut=shortcut
            )
            h = hs_and_zs.h_by_subj[sample.subject] if not shortcut else hs_and_zs.h_by_subj[operator.prompt_template[-1]]

            # Approximating LM computation as an affine transformation
            z = operator.beta * (operator.weight @ h) + operator.bias

            # Show the logit lens results of the approximation
            logit_lens_res = lens.logit_lens(
                mt = mt,
                h = z,
                get_proba = True
            )
            print(logit_lens_res[0])
            
            ### Evaluate faithfulness on all samples (correct+wrong originally) ###
            correct = 0
            wrong = 0
            correct_train, wrong_train = 0, 0
            correct_test, wrong_test = 0, 0
            correct_pre_correct, wrong_pre_correct = 0, 0
            correct_pre_wrong, wrong_pre_wrong = 0, 0
            samples_correct_post = []
            
            ### Stats for samples correct in english
            if language != 'en':
                correct_en_correct, wrong_en_correct = 0, 0
                correct_pre_correct_en_correct, wrong_pre_correct_en_correct = 0, 0
                correct_pre_wrong_en_correct, wrong_pre_wrong_en_correct = 0, 0
            
            for sample in all_samples:
                predictions = operator(subject = sample.subject).predictions
                if "bloom" in mt.model.name_or_path:
                    target_tokens = sample.object
                    known_flag = functional.is_nontrivial_prefix(
                        prediction=predictions[0].token, target=sample.object
                    )
                else:
                    target_tokens = ''.join(mt.tokenizer.tokenize(sample.object)).strip('_').strip('▁')
                    known_flag = functional.is_nontrivial_prefix(
                        prediction=predictions[0].token.strip('▁'), target=target_tokens
                    )
                print(f"{sample.subject=}, {sample.object=}, {target_tokens=},", end="")
                print(f'predicted="{functional.format_whitespace(predictions[0].token)}", (p={predictions[0].prob}), known=({functional.get_tick_marker(known_flag)})')
                
                if known_flag:
                    samples_correct_post.append(sample)
                    
                correct += known_flag
                wrong += not known_flag
                
                if sample in train.samples:
                    correct_train += known_flag
                    wrong_train += not known_flag   
                    
                if sample in test.samples:
                    correct_test += known_flag
                    wrong_test += not known_flag            
                
                if sample not in wrong_samples:
                    correct_pre_correct += known_flag
                    wrong_pre_correct += not known_flag
                else:
                    correct_pre_wrong += known_flag
                    wrong_pre_wrong += not known_flag
                    
                # For samples correct in english
                if language != 'en':
                    if sample in all_samples_en_correct:
                        correct_en_correct += known_flag
                        wrong_en_correct += not known_flag
                    
                    if sample in filtered_samples_en_correct:
                        correct_pre_correct_en_correct += known_flag
                        wrong_pre_correct_en_correct += not known_flag
                    
                    if sample in wrong_samples_en_correct:
                        correct_pre_wrong_en_correct += known_flag
                        wrong_pre_wrong_en_correct += not known_flag
            
            faithfulness_all = correct/(correct + wrong)
            faithfulness_train = correct_train/(train_size)
            faithfulness_test = correct_test/(correct_test + wrong_test)
            faithfulness_pre_correct = correct_pre_correct/(correct_pre_correct + wrong_pre_correct)
            faithfulness_pre_wrong = correct_pre_wrong/(correct_pre_wrong + wrong_pre_wrong) if wrong_samples != [] else None
            post_accuracy_best = pre_accuracy + (1 - pre_accuracy) * faithfulness_pre_wrong if faithfulness_pre_wrong else faithfulness_all
            
            print("------------------------------------------------------------")
            print(f"Faithfulness (@1) on test samples = {faithfulness_test}")
            print(f"Faithfulness (@1) on all samples = {faithfulness_all}")
            print(f"Faithfulness (@1) on all originally correct samples = {faithfulness_pre_correct}")
            print(f"Faithfulness (@1) on all originally wrong samples = {faithfulness_pre_wrong}")
            print("------------------------------------------------------------")
            print(f"Recall - Original accuracy: {pre_accuracy}")
            print(f"Shortcut accuracy: {faithfulness_all}")
            print("------------------------------------------------------------")


            if language != 'en':
                faithfulness_all_en_correct = correct_en_correct/(correct_en_correct + wrong_en_correct)
                faithfulness_pre_correct_en_correct = correct_pre_correct_en_correct/(correct_pre_correct_en_correct + wrong_pre_correct_en_correct)
                faithfulness_pre_wrong_en_correct = correct_pre_wrong_en_correct/(correct_pre_wrong_en_correct + wrong_pre_wrong_en_correct) if wrong_samples_en_correct != [] else None
                post_accuracy_best_en_correct = pre_accuracy_en_correct + (1 - pre_accuracy_en_correct) * faithfulness_pre_wrong_en_correct if faithfulness_pre_wrong_en_correct else faithfulness_all_en_correct
                
                samples_intersection_correct_post = [sample for sample in all_samples if sample.index in sample_indices_en_correct and sample in samples_correct_post]
                samples_intersection_correct_post_best = [sample for sample in all_samples if sample.index in sample_indices_en_correct and (sample in samples_correct_post or sample in relation.samples)]
                samples_union_correct_post = [sample for sample in all_samples if sample.index in sample_indices_en_correct or sample in samples_correct_post]
                
                post_clc_en = len(samples_intersection_correct_post) / len(samples_union_correct_post)
                post_clc_en_best = len(samples_intersection_correct_post_best) / len(samples_union_correct_post)
            
                print("------------------------------------------------------------")
                print(f"Faithfulness (@1) on all samples correct in English = {faithfulness_all_en_correct}")
                print(f"Faithfulness (@1) on all originally correct samples in English = {faithfulness_pre_correct_en_correct}")
                print(f"Faithfulness (@1) on all originally wrong samples in English = {faithfulness_pre_wrong_en_correct}")
                print("------------------------------------------------------------")
                print(f"Recall - Original accuracy on English correct samples: {pre_accuracy_en_correct}")
                print(f"Shortcut accuracy on English correct samples: {faithfulness_all_en_correct}")
                print("------------------------------------------------------------")
                print(f"Recall - Original cross-lingual consistency to English: {pre_clc_en}")
                print(f"Shortcut cross-lingual consistency to English: {post_clc_en}")
                print(f"Shortcut cross-lingual consistency to English (best): {post_clc_en_best}")
                print("------------------------------------------------------------")
            
            results_summary = {
                "relation": relation_name,
                "results_all_samples": {
                    "faithfulness@1_all": faithfulness_all,
                    "faithfulness@1_train": faithfulness_train,
                    "faithfulness@1_test": faithfulness_test,
                    "faithfulness@1_pre_correct": faithfulness_pre_correct,
                    "faithfulness@1_pre_wrong": faithfulness_pre_wrong,
                    "pre_accuracy": pre_accuracy,
                    "post_accuracy": faithfulness_all,
                    "post_accuracy_best": post_accuracy_best,
                    "pre_clc_en": pre_clc_en if language != 'en' else 1.0,
                    "post_clc_en": post_clc_en if language != 'en' else 1.0,
                    "post_clc_en_best": post_clc_en_best if language != 'en' else 1.0,
                    "n_all_samples": n_all_samples,
                    "n_filtered_samples": n_filtered_samples,
                },
                "results_samples_en_correct": {
                    "faithfulness@1_all": faithfulness_all_en_correct,
                    "faithfulness@1_pre_correct": faithfulness_pre_correct_en_correct,
                    "faithfulness@1_pre_wrong": faithfulness_pre_wrong_en_correct,
                    "pre_accuracy": pre_accuracy_en_correct,
                    "post_accuracy": faithfulness_all_en_correct,
                    "post_accuracy_best": post_accuracy_best_en_correct,
                    "pre_clc_en": pre_clc_en,
                    "post_clc_en": post_clc_en,
                    "post_clc_en_best": post_clc_en_best,
                    "samples_en_correct_stats": samples_en_correct_stats
                } if language != 'en' else None,
                "hyperparameters": {
                    "ratio_train_samples": ratio_train_samples,
                    "train_size": train_size,
                    "test_size": test_size,
                    "layer_h": layer,
                    "beta": beta
                },
            }
            
            with open(output_filepath, 'w') as f:
                json.dump(results_summary, f, indent=4)
            
            os.makedirs(os.path.join(output_dir, "samples_indices", "pre"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "samples_indices", "post"), exist_ok=True)
            sample_indices_pre_filepath = os.path.join(output_dir, "samples_indices", "pre", f"train_size_{args.max_train_samples}_ratio_{args.ratio_train_samples}_layer_{layer}_beta_{str(beta).replace('.', '-')}.json")
            sample_indices_post_filepath = os.path.join(output_dir, "samples_indices", "post", f"train_size_{args.max_train_samples}_ratio_{args.ratio_train_samples}_layer_{layer}_beta_{str(beta).replace('.', '-')}.json")
            
            with open(sample_indices_pre_filepath, 'w') as f:
                correct_samples_indices_pre = [sample.index for sample in relation.samples]
                json.dump(correct_samples_indices_pre, f, indent=4)
                
            with open(sample_indices_post_filepath, 'w') as f:
                correct_samples_indices_post = [sample.index for sample in samples_correct_post]
                json.dump(correct_samples_indices_post, f, indent=4)
            
            print(f"Results ready for {relation_name}!")
            
    
    if mode == "summarize":
        results_summary_all_relations = {}
        results_summary_all_relations["relations"] = []
        
        for relation_name in relation_names:
        
            output_dir = f"{args.output_dir}/{relation_name}"
            output_filepath = os.path.join(output_dir, f"train_size_{args.max_train_samples}_ratio_{args.ratio_train_samples}_layer_{layer}_beta_{str(beta).replace('.', '-')}.json")
            if os.path.exists(output_filepath):
                with open(output_filepath, "r") as f:
                    results_summary = json.load(f)
                results_summary_all_relations[relation_name] = results_summary
                results_summary_all_relations["relations"].append(relation_name.replace(f"{language}_", ""))
                
        print(f"Summarizing results for *{language}* - beta {beta}...")
        metrics = [
            'faithfulness@1_pre_correct',
            'faithfulness@1_pre_wrong',
            'pre_accuracy', 'post_accuracy', 'post_accuracy_best',
            'pre_clc_en', 'post_clc_en', 'post_clc_en_best'
            ]
        # Initialize storage for accumulated metrics
        aggregated_metrics = {
            "results_all_samples": {metric: [] for metric in metrics},
            "results_samples_en_correct": {metric: [] for metric in metrics},
            }
        number_samples_micro = {
            "results_all_samples": {metric: [] for metric in metrics},
            "results_samples_en_correct": {metric: [] for metric in metrics},
            }


        # Iterate over all relations and accumulate the metrics
        for relation, results in results_summary_all_relations.items():
            if relation == "relations":
                continue
            
            if "results_all_samples" in results:
                for metric in metrics:
                    if metric in results["results_all_samples"]:
                        aggregated_metrics["results_all_samples"][metric].append(results["results_all_samples"][metric])
                        
                        if metric == "faithfulness@1_pre_correct":
                            n_samples = results["results_all_samples"]["n_filtered_samples"]
                        elif metric == "faithfulness@1_pre_wrong":
                            n_samples = results["results_all_samples"]["n_all_samples"] - results["results_all_samples"]["n_filtered_samples"]
                        elif metric.startswith("pre_accuracy") or metric.startswith("post_accuracy"):
                            n_samples = results["results_all_samples"]["n_all_samples"]
                        elif "clc_en" in metric:
                            n_samples = results["results_samples_en_correct"]["samples_en_correct_stats"]["n_samples_en_correct"] if language != "en" else 1
                        else:
                            n_samples = 0
                        number_samples_micro["results_all_samples"][metric].append(n_samples)
            
            if "results_samples_en_correct" in results and results["results_samples_en_correct"]:
                for metric in metrics:
                    if metric in results["results_samples_en_correct"]:
                        aggregated_metrics["results_samples_en_correct"][metric].append(results["results_samples_en_correct"][metric])
                        
                        if metric == "faithfulness@1_pre_correct":
                            n_samples = results["results_samples_en_correct"]["samples_en_correct_stats"]["n_correct_samples_en_correct"]
                        elif metric == "faithfulness@1_pre_wrong":
                            n_samples = results["results_samples_en_correct"]["samples_en_correct_stats"]["n_wrong_samples_en_correct"]
                        elif metric.startswith("pre_accuracy") or metric.startswith("post_accuracy") or "clc_en" in metric:
                            n_samples = results["results_samples_en_correct"]["samples_en_correct_stats"]["n_samples_en_correct"]
                        else:
                            n_samples = 0
                            
                        number_samples_micro["results_samples_en_correct"][metric].append(n_samples)

        # Calculate average and variance for each metric
        average_across_relations = {
            # "average_across_relations": {},
            "results_all_samples": {},
            "results_samples_en_correct": {},
            }
        micro_average_across_relations = {
            # "average_across_relations": {},
            "results_all_samples": {},
            "results_samples_en_correct": {},
            }
        for key in ["results_all_samples", "results_samples_en_correct"]:
            if key == "results_samples_en_correct" and language == "en":
                continue
            
            for metric, values in aggregated_metrics[key].items():
                values = [v for v in values]
                numbers = [n for n in number_samples_micro[key][metric]]
                
                # Filter both lists to remove None elements in values
                filtered_values, filtered_numbers = zip(*[(v, n) for v, n in zip(values, numbers) if v is not None])
                values, numbers = list(filtered_values), list(filtered_numbers)
                
                if values:  # Avoid empty lists
                    average_across_relations[key][metric] = {
                        "mean": np.mean(values).item(),
                        "variance": np.var(values).item()
                    }
                    
                    # Weighted average
                    weighted_avg = np.sum(np.array(values) * np.array(numbers)) / np.sum(numbers)
                    # Weighted variance
                    variance = np.sum(np.array(numbers) * (np.array(values) - weighted_avg) ** 2) / np.sum(numbers)
                    
                    micro_average_across_relations[key][metric] = {
                        "mean": weighted_avg.item(),
                        "variance": variance.item()
                    }

        # Save the averaged results back into the data structure
        results_summary_all_relations["average_across_relations"] = average_across_relations
        results_summary_all_relations["micro_average_across_relations"] = micro_average_across_relations
        
        output_dir = f"{args.output_dir}/averaged/{relation_name.split('_', 1)[0]}_average"
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, f"train_size_{args.max_train_samples}_ratio_{args.ratio_train_samples}_layer_{layer}_beta_{str(beta).replace('.', '-')}.json")
        
        with open(output_filepath, 'w') as f:
            json.dump(results_summary_all_relations, f, indent=4)
        
        print(f"Averaged results across relations ready!")