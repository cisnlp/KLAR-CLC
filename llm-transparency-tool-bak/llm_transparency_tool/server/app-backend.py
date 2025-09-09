# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os
from tqdm import tqdm
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
import numpy as np
import random
import networkx as nx

import torch
from jaxtyping import Float
from torch.amp import autocast

import llm_transparency_tool.routes.contributions as contributions
import llm_transparency_tool.routes.graph_off
from llm_transparency_tool.models.transparent_llm import TransparentLlm
from llm_transparency_tool.routes.graph_node import NodeType
from llm_transparency_tool.server.graph_selection import (
    GraphSelection,
    UiGraphNode,
)
from llm_transparency_tool.server.utils_off import (
    B0,
    get_contribution_graph,
    load_model,
    get_val,
)


from dataclasses import dataclass
from typing import List
import json
import pickle
from collections import defaultdict


@dataclass
class LogLensResult:
    token_idx: int = 0
    top_tokens: List[str] = field(default_factory=list)
    rank_subject: int = 0
    rank_answer: int = 0
    logit_subject: float = 0.0
    logit_answer: float = 0.0
    prob_subject: float = 0.0
    prob_answer: float = 0.0
    max_logit: float = 0.0
    max_prob: float = 0.0
    entropy: float = 0.0


# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def cached_build_paths_to_predictions(
    graph: nx.Graph,
    n_layers: int,
    n_tokens: int,
    starting_tokens: List[int],
    threshold: float,
):
    return llm_transparency_tool.routes.graph_off.build_paths_to_predictions(
        graph, n_layers, n_tokens, starting_tokens, threshold
    )


def cached_run_inference_and_populate_state(
    stateless_model, sentences, subject, object, en_subject, en_object, demo
):
    # stateful_model = stateless_model.copy()
    stateless_model.run(sentences, subject, object, en_subject, en_object, demo)
    return stateless_model


def load_json_files(
    directory: str, selected_category: str = None, selected_relation: str = None
) -> Dict[str, List[Dict]]:
    data = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                with open(path, "r") as f:
                    category = os.path.basename(root)
                    if category == selected_category or not selected_category:
                        relation = os.path.basename(file).replace(".json", "")
                        if selected_relation and relation != selected_relation:
                            continue
                        if relation not in data:
                            data[relation] = []
                        data[relation].append(json.load(f))

    return data


def find_subject_token_position(template: str, subject: str) -> int:
    # Format the template with the subject
    fact = template.format(subject)

    # Tokenize the sentence by splitting on whitespace
    tokens = fact.split()

    # Find the position of the subject in the tokenized list (if contained in a token)
    for i, token in enumerate(tokens):
        if subject in token:
            return i

    # If subject is not found, raise an error
    raise ValueError(f"Subject '{subject}' not found in the tokenized string.")


def parse_samples(
    data: Dict[str, List[Dict]], no_entity_space=False, few_shot_demo=0
) -> Tuple[List[str], List[str], List[str], List[str], List[int]]:
    facts = []
    subjects = []
    targets = []
    subjects_en = []
    targets_en = []
    demos = []
    sentences = []
    all_subject_positions = []
    indices = []
    subject_object_to_index = {}
    current_index = 0

    for entries in data:
        prompt_templates = entries["prompt_templates"]
        samples = entries["samples"]

        for sample in samples:
            subject = sample["subject"]
            obj = sample["object"]
            subject_en = (
                sample["subject_en"] if "subject_en" in sample else sample["subject"]
            )
            obj_en = sample["object_en"] if "object_en" in sample else sample["object"]

            # Use the subject-object pair as the unique identifier for indexing
            subject_object_pair = (subject, obj)

            if subject_object_pair not in subject_object_to_index:
                subject_object_to_index[subject_object_pair] = current_index
                current_index += 1

            # Get the index for this subject-object pair
            # fact_index = subject_object_to_index[subject_object_pair]
            fact_index = sample["index"]
            prompt_templates = [
                prompt_templates[0]
            ]  # Only use the first template for now, delete this line if probing with all templates
            for template in prompt_templates:  # Iterate through multiple templates
                if no_entity_space:
                    template = template.replace("<subject> ", "<subject>")

                # Sample demonstration examples from 'samples' (excluding the target sample)
                demo_samples = random.sample(
                    [s for s in samples if s != sample], few_shot_demo
                )

                # Create demonstration text by filling the template with sampled subjects and objects
                demo_texts = []
                for demo in demo_samples:
                    demo_subject = demo["subject"]
                    demo_obj = (
                        demo["object"] + "."
                        if not no_entity_space
                        else demo["object"] + "。"
                    )

                    demo_fact = (
                        template.replace("<subject>", demo_subject)
                        .replace("<MASK>", "<mask>")
                        .replace("<mask>", demo_obj)
                    )
                    demo_texts.append(demo_fact)

                # Join demonstration texts with the target prompt
                demonstrations = (
                    " ".join(demo_texts) + " "
                    if not no_entity_space
                    else "".join(demo_texts)
                )  # Combine demos into one prompt
                demos.append(demonstrations)

                template = (
                    template.replace("<MASK>", "<mask>")
                    .split("<mask>")[0]
                    .replace("<subject>", "{}")
                )
                template = demonstrations + template
                subject_pos = find_subject_token_position(
                    template, "{}"
                )  # Wrong implementation, not used as well

                fact = template.format(subject)
                facts.append(fact)
                if subject_pos != 0:
                    subjects_en.append(" " + subject_en)
                    if not no_entity_space:
                        subjects.append(" " + subject)
                    else:
                        subjects.append(subject)

                else:
                    subjects.append(subject)
                    subjects_en.append(" " + subject_en)
                sentence = (
                    fact + obj + "." if not no_entity_space else fact + obj + "。"
                )
                obj = " " + obj + "." if not no_entity_space else obj + "。"
                obj_en = " " + obj_en + "."
                targets.append(obj)
                targets_en.append(obj_en)
                sentences.append(sentence)
                all_subject_positions.append(subject_pos)
                indices.append(
                    fact_index
                )  # Append the index corresponding to this subject-object pair

    return (
        sentences,
        facts,
        subjects,
        targets,
        subjects_en,
        targets_en,
        demos,
        all_subject_positions,
        indices,
    )


def save_hidden_states_per_instance(data, relation, language, sample_idx, output_dir):
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(output_dir, f"{relation}_{language}", "hidden_states")
    os.makedirs(output_dir, exist_ok=True)

    # Create a filename for each relation
    filename = os.path.join(output_dir, f"{relation}_{language}_{sample_idx}.pkl")

    # Save the relation analysis to a pickle file
    with open(filename, "wb") as f:
        pickle.dump(data, f)

    print(
        f"Hidden states for relation '{relation}_{language}_{sample_idx}' saved to {output_dir}"
    )


def save_analysis_per_relation(data, relation, output_dir, language=""):
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.join(output_dir, f"{relation}_{language}"), exist_ok=True)

    # Create a filename for each relation
    filename = os.path.join(
        output_dir, f"{relation}_{language}", f"{relation}_{language}.pkl"
    )

    # Save the relation analysis to a pickle file
    with open(filename, "wb") as f:
        pickle.dump(data, f)

    print(f"Analysis for relation '{relation}' saved to {filename}")


# Function to find the span indices in a tensor


def find_span_indices(tokens, subj_tokens):
    # Get the length of the span to match
    span_len = len(subj_tokens)
    result_list = []
    # Sliding window over the tokens to find where the subj_tokens match
    for i in range(len(tokens) - span_len + 1):
        if torch.equal(tokens[i : i + span_len], subj_tokens):
            result_list.append(list(range(i, i + span_len)))
    if result_list == []:
        return None
    if type(result_list[0]) == list:
        return result_list[-1]


class App:
    _stateful_model: TransparentLlm = None
    _graph: Optional[nx.Graph] = None
    _contribution_threshold: float = 0.0
    _renormalize_after_threshold: bool = False
    _normalize_before_unembedding: bool = True

    @property
    def stateful_model(self) -> TransparentLlm:
        return self._stateful_model

    def __init__(self):
        pass

    def _get_representation(
        self, node: Optional[UiGraphNode]
    ) -> Optional[Float[torch.Tensor, "d_model"]]:
        if node is None:
            return None
        fn = {
            NodeType.AFTER_ATTN: self.stateful_model.residual_after_attn,
            NodeType.AFTER_FFN: self.stateful_model.residual_out,
            NodeType.FFN: None,
            NodeType.ORIGINAL: self.stateful_model.residual_in,
        }
        return fn[node.type](node.layer)[B0][node.token]

    def load_config(self, config_file):
        with open(config_file, "r") as f:
            config = json.load(f)

        self.model_name = config.get("model_name", "allenai/OLMo-7B-0424-hf")
        self.device = config.get("device", "gpu")
        self._model_path = config.get("_model_path", None)
        dtype_str = config.get("dtype", "torch.bfloat16")
        self.dtype = getattr(torch, dtype_str, torch.bfloat16)
        self.amp_enabled = config.get("amp_enabled", True)
        self._renormalize_after_threshold = config.get(
            "renormalize_after_threshold", True
        )
        self._normalize_before_unembedding = config.get(
            "normalize_before_unembedding", True
        )
        self._prepend_bos = config.get("prepend_bos", False)
        self._do_neuron_level = config.get("do_neuron_level", False)
        self._neuron_extraction_mode = config.get("neuron_extraction_mode", "threshold")
        self._contribution_threshold = config.get("contribution_threshold", 0.01)
        self._logit_lens_topK = config.get("logit_lens_topK", 10)
        self._logit_lens_topK_neurons = config.get("logit_lens_topK_neurons", 10)
        self._run_logit_lens_on_heads = config.get("run_logit_lens_on_heads", True)

    def _unembed(
        self,
        representation: torch.Tensor,
    ) -> torch.Tensor:
        return self.stateful_model.unembed(
            representation, normalize=self._normalize_before_unembedding
        )

    def draw_graph(self, contribution_threshold: float) -> Optional[GraphSelection]:
        tokens = self.stateful_model.tokens()[B0]
        n_tokens = tokens.shape[0]
        model_info = self.stateful_model.model_info()

        graphs = cached_build_paths_to_predictions(
            self._graph,
            model_info.n_layers,
            n_tokens,
            range(n_tokens),
            contribution_threshold,
        )

        return llm_transparency_tool.components.contribution_graph(
            model_info,
            self.stateful_model._model.tokenizer.convert_ids_to_tokens(tokens),
            graphs,
            key=f"graph_{hash(self.sentence)}",
        )

    def run_inference(self):

        with autocast(enabled=self.amp_enabled, device_type="cuda", dtype=self.dtype):
            self._stateful_model = cached_run_inference_and_populate_state(
                self.stateful_model,
                [self.sentence],
                self.subj_token,
                self.obj_token,
                self.en_subj_token,
                self.en_obj_token,
                self.demo,
            )

        demo_tokens = self.stateful_model._last_run.demo_tokens[B0]
        n_demo_tokens = (
            demo_tokens.shape[0]
            if args.language in ["ja", "zh"] or "bloom" in self.model_name.lower()
            else demo_tokens.shape[0] - 1
        )  # one additional <eos> token
        n_demo_tokens = max(0, n_demo_tokens)

        with autocast(enabled=self.amp_enabled, device_type="cuda", dtype=self.dtype):
            self._graph, self._contributions_dict = get_contribution_graph(
                self.stateful_model,
                "",
                self.stateful_model.tokens()[B0].tolist(),
                (
                    self._contribution_threshold
                    if self._renormalize_after_threshold
                    else 0.0
                ),
                n_demo_tokens,
            )

    # @profile
    def process_logits(
        self,
        logit_scores,
        attn_contribution=None,
        subj_voc_id=None,
        answer_voc_id=None,
        en_subj_voc_id=None,
        en_answer_voc_id=None,
        topk_pred_tokens=None,
    ):
        # s2 = time.time()
        # tokens = self.stateful_model.tokens()[B0]
        probs = torch.nn.functional.softmax(logit_scores, dim=-1).cpu()
        entropy = (-probs * torch.log(probs + 1e-10)).sum(dim=-1).cpu()
        sorted_indices = torch.argsort(logit_scores, dim=-1, descending=True)
        rank_subj = (sorted_indices == subj_voc_id).nonzero(as_tuple=True)[1].cpu()
        rank_ans = (sorted_indices == answer_voc_id).nonzero(as_tuple=True)[1].cpu()
        rank_en_subj = (
            (sorted_indices == en_subj_voc_id).nonzero(as_tuple=True)[1].cpu()
        )
        rank_en_ans = (
            (sorted_indices == en_answer_voc_id).nonzero(as_tuple=True)[1].cpu()
        )

        if topk_pred_tokens is not None:
            rank_topk_pred_list = []
            for pred in topk_pred_tokens:
                rank_pred = (sorted_indices == pred).nonzero(as_tuple=True)[1].cpu()
                rank_topk_pred_list.append(rank_pred)

        results = []

        for token_idx in range(len(entropy)):
            result = {}
            result["token_idx"] = token_idx
            if attn_contribution is not None:
                result["attn_contribution"] = [
                    round(c, 4)
                    for c in attn_contribution[token_idx].tolist()[: token_idx + 1]
                ]
                if all(c == 0.0 for c in result["attn_contribution"]):
                    continue
            # tok = self.stateful_model.tokens_to_strings(sorted_indices[token_idx][:self._logit_lens_topK].cpu())
            result["top_tokens"] = [
                int(i) for i in sorted_indices[token_idx][: self._logit_lens_topK].cpu()
            ]
            result["rank_subject"] = get_val(rank_subj[token_idx])
            result["rank_en_subject"] = get_val(rank_en_subj[token_idx])
            result["rank_answer"] = get_val(rank_ans[token_idx])
            result["rank_en_answer"] = get_val(rank_en_ans[token_idx])
            result["rank_topk_pred"] = (
                [
                    get_val(rank_topk_pred_list[i][token_idx])
                    for i in range(len(rank_topk_pred_list))
                ]
                if topk_pred_tokens is not None
                else None
            )

            result["logit_subject"] = get_val(
                logit_scores[token_idx][subj_voc_id].cpu()
            )
            result["logit_answer"] = get_val(
                logit_scores[token_idx][answer_voc_id].cpu()
            )
            result["logit_en_answer"] = get_val(
                logit_scores[token_idx][en_answer_voc_id].cpu()
            )
            result["logit_topk_pred"] = (
                [
                    get_val(logit_scores[token_idx][pred_voc_id].cpu())
                    for pred_voc_id in topk_pred_tokens
                ]
                if topk_pred_tokens is not None
                else None
            )

            result["prob_subject"] = get_val(probs[token_idx][subj_voc_id])
            result["prob_answer"] = get_val(probs[token_idx][answer_voc_id])
            result["prob_en_answer"] = get_val(probs[token_idx][en_answer_voc_id])
            result["prob_topk_pred"] = (
                [
                    get_val(probs[token_idx][pred_voc_id])
                    for pred_voc_id in topk_pred_tokens
                ]
                if topk_pred_tokens is not None
                else None
            )

            result["max_logit"] = get_val(logit_scores[token_idx].max().cpu())
            result["max_prob"] = get_val(probs[token_idx].max())
            result["entropy"] = get_val(entropy[token_idx])

            token_ids = result["top_tokens"]

            if "llama" in self.model_name.lower():
                token_strings = (
                    self._stateful_model._model.tokenizer.convert_ids_to_tokens(
                        token_ids
                    )
                )
            elif "bloom" in self.model_name.lower():
                token_strings = [
                    self.stateful_model._model.tokenizer.decode(token)
                    for token in token_ids
                ]

            result["top_token_strings"] = token_strings
            results.append(result)
        return results

    @torch.no_grad()
    def run_logit_lens_on_resid(
        self,
        n_layers,
        n_tokens,
        n_demo_tokens,
        subj_voc_id=None,
        answer_voc_id=None,
        en_subj_voc_id=None,
        en_answer_voc_id=None,
        topk_pred_tokens=None,
    ):
        full_results = {}
        representation_results = {}
        for layer in range(n_layers):
            for resid_pos in ["pre", "mid"]:
                if resid_pos == "pre":
                    representations = self.stateful_model.residual_in(layer)[B0][
                        n_demo_tokens:n_tokens
                    ]
                else:
                    representations = self.stateful_model.residual_after_attn(layer)[
                        B0
                    ][n_demo_tokens:n_tokens]
                logit_scores = self._unembed(representations)
                hook_name = f"{layer}_{resid_pos}"
                full_results[hook_name] = self.process_logits(
                    logit_scores,
                    subj_voc_id=subj_voc_id,
                    answer_voc_id=answer_voc_id,
                    en_subj_voc_id=en_subj_voc_id,
                    en_answer_voc_id=en_answer_voc_id,
                    topk_pred_tokens=topk_pred_tokens,
                )
                if self._log_hidden_states:
                    representation_results[hook_name] = representations[
                        self.input_last_token
                    ].cpu()
        representations = self.stateful_model.residual_out(n_layers - 1)[B0][
            n_demo_tokens:n_tokens
        ]
        logit_scores = self._unembed(representations)
        full_results["final_post"] = self.process_logits(
            logit_scores,
            subj_voc_id=subj_voc_id,
            answer_voc_id=answer_voc_id,
            en_subj_voc_id=en_subj_voc_id,
            en_answer_voc_id=en_answer_voc_id,
            topk_pred_tokens=topk_pred_tokens,
        )
        if self._log_hidden_states:
            representation_results["final_post"] = representations[
                self.input_last_token
            ].cpu()
        return full_results, representation_results

    @torch.no_grad()
    def run_logit_lens_on_outputs(
        self,
        n_layers,
        n_tokens,
        n_demo_tokens,
        subj_voc_id=None,
        answer_voc_id=None,
        en_subj_voc_id=None,
        en_answer_voc_id=None,
        topk_pred_tokens=None,
    ):
        full_results = {}
        representations = self.stateful_model.residual_in(0)[B0][n_demo_tokens:n_tokens]
        logit_scores = self._unembed(representations)
        full_results["embed"] = self.process_logits(
            logit_scores,
            subj_voc_id=subj_voc_id,
            answer_voc_id=answer_voc_id,
            en_subj_voc_id=en_subj_voc_id,
            en_answer_voc_id=en_answer_voc_id,
            topk_pred_tokens=topk_pred_tokens,
        )
        for layer in range(n_layers):
            for resid_pos in ["attn", "mlp"]:
                if resid_pos == "attn":
                    representations = self.stateful_model._get_block(
                        layer, "hook_attn_out"
                    )[B0][n_demo_tokens:n_tokens]
                else:
                    representations = self.stateful_model.ffn_out(layer)[B0][
                        n_demo_tokens:n_tokens
                    ]
                logit_scores = self._unembed(representations)
                hook_name = f"{layer}_{resid_pos}_out"
                full_results[hook_name] = self.process_logits(
                    logit_scores,
                    subj_voc_id=subj_voc_id,
                    answer_voc_id=answer_voc_id,
                    en_subj_voc_id=en_subj_voc_id,
                    en_answer_voc_id=en_answer_voc_id,
                    topk_pred_tokens=topk_pred_tokens,
                )
        return full_results

    @torch.no_grad()
    def run_logit_lens_on_heads(
        self,
        n_layers,
        n_heads,
        n_tokens,
        n_demo_tokens,
        subj_voc_id=None,
        answer_voc_id=None,
        en_subj_voc_id=None,
        en_answer_voc_id=None,
        topk_pred_tokens=None,
    ):
        full_results = {}
        for layer in range(n_layers):
            resid_representations = self.stateful_model.residual_in(layer)[B0][
                self.n_demo_tokens :
            ]
            representations = self.stateful_model._get_block(layer, "attn.hook_result")[
                B0
            ][n_demo_tokens:n_tokens]
            for head in range(n_heads):
                hook_name = f"L{layer}H{head}"
                attn_contribution = self._contributions_dict["c_attns"][layer][
                    0, :, :, head
                ]
                if torch.all(attn_contribution == 0.0):
                    continue
                logit_scores = self._unembed(representations[:, head])
                full_results[hook_name] = self.process_logits(
                    logit_scores,
                    attn_contribution=attn_contribution,
                    subj_voc_id=subj_voc_id,
                    answer_voc_id=answer_voc_id,
                    en_subj_voc_id=en_subj_voc_id,
                    en_answer_voc_id=en_answer_voc_id,
                    topk_pred_tokens=topk_pred_tokens,
                )

                representations_plus = resid_representations + representations[:, head]
                logit_scores_plus = self._unembed(representations_plus)
                resid_plus_results = self.process_logits(
                    logit_scores_plus,
                    attn_contribution=attn_contribution,
                    subj_voc_id=subj_voc_id,
                    answer_voc_id=answer_voc_id,
                    en_subj_voc_id=en_subj_voc_id,
                    en_answer_voc_id=en_answer_voc_id,
                    topk_pred_tokens=topk_pred_tokens,
                )

                if [attr["token_idx"] for attr in full_results[hook_name]] == [
                    attr["token_idx"] for attr in resid_plus_results
                ]:
                    for idx, attr in enumerate(full_results[hook_name]):
                        token_idx = attr["token_idx"]
                        resid_results_layer = self.resid_logit_lens_results[
                            f"{layer}_pre"
                        ]
                        resid_results_layer = [
                            res
                            for res in resid_results_layer
                            if res["token_idx"]
                            in [attr["token_idx"] for attr in full_results[hook_name]]
                        ]

                        diff_keys = [
                            key
                            for key in attr.keys()
                            if key
                            not in [
                                "token_idx",
                                "attn_contribution",
                                "top_tokens",
                                "max_logit",
                                "max_prob",
                                "top_token_strings",
                                "resid_plus",
                            ]
                        ]
                        for key in diff_keys:
                            if key.endswith("_topk_pred"):
                                resid_plus_results[idx][f"{key}_diff"] = [
                                    v1 - v2
                                    for v1, v2 in zip(
                                        resid_plus_results[idx][key],
                                        resid_results_layer[idx][key],
                                    )
                                ]
                            else:
                                resid_plus_results[idx][f"{key}_diff"] = (
                                    resid_plus_results[idx][key]
                                    - resid_results_layer[idx][key]
                                )

                        full_results[hook_name][idx]["resid_plus"] = resid_plus_results[
                            idx
                        ]

        return full_results

    @torch.no_grad()
    def compute_neuron_contributions(self, n_layers, n_tokens, n_demo_tokens):
        tokens = self.stateful_model.tokens()[B0][n_demo_tokens:n_tokens]

        ffn_contributions = []
        for layer in range(n_layers):
            hook_name = f"L{layer}"
            resid_mid = self.stateful_model.residual_after_attn(layer)[B0][
                n_demo_tokens:n_tokens
            ]
            resid_post = self.stateful_model.residual_out(layer)[B0][
                n_demo_tokens:n_tokens
            ]
            decomposed_ffn = self.stateful_model.decomposed_ffn_out(
                B0, layer, -1, n_demo_tokens
            )
            results = []
            for token in range(len(tokens)):
                c_ffn, _ = contributions.get_decomposed_mlp_contributions(
                    resid_mid[token], resid_post[token], decomposed_ffn[token]
                )
                results.append(c_ffn)
            ffn_contributions.append(torch.stack(results))
        return torch.stack(ffn_contributions).transpose(1, 0)  # pos layer #neurons

    @torch.no_grad()
    def run_logit_lens_on_neurons(
        self,
        n_layers,
        sel_neurons_layerwise,
        subj_voc_id=None,
        answer_voc_id=None,
        en_answer_voc_id=None,
    ):
        full_results = {}
        for layer in range(n_layers):
            sel_neurons = sel_neurons_layerwise[layer]
            for neuron in sel_neurons:
                hook_name = f"L{layer}N{neuron}"
                representations = (
                    self.stateful_model.neuron_output(layer, neuron)
                    .unsqueeze(0)
                    .unsqueeze(1)
                )
                logit_scores = self._unembed(representations)[B0][0]
                probs = torch.nn.functional.softmax(logit_scores, dim=-1).cpu()
                entropy = (-probs * torch.log(probs + 1e-10)).sum(dim=-1).cpu()
                sorted_indices = torch.argsort(
                    logit_scores, dim=-1, descending=True
                ).cpu()

                result = {}
                result["top_tokens"] = [
                    int(i) for i in sorted_indices[: self._logit_lens_topK].cpu()
                ]
                result["rank_subject"] = get_val(
                    (sorted_indices == subj_voc_id).nonzero(as_tuple=True)[0]
                )
                result["rank_answer"] = get_val(
                    (sorted_indices == answer_voc_id).nonzero(as_tuple=True)[0]
                )
                result["rank_en_answer"] = get_val(
                    (sorted_indices == en_answer_voc_id).nonzero(as_tuple=True)[0]
                )
                result["logit_subject"] = get_val(logit_scores[subj_voc_id].cpu())
                result["logit_answer"] = get_val(logit_scores[answer_voc_id].cpu())
                result["logit_en_answer"] = get_val(
                    logit_scores[en_answer_voc_id].cpu()
                )
                result["prob_subject"] = get_val(probs[subj_voc_id])
                result["prob_answer"] = get_val(probs[answer_voc_id])
                result["prob_en_answer"] = get_val(probs[en_answer_voc_id])
                result["max_logit"] = get_val(logit_scores.max().cpu())
                result["max_prob"] = get_val(probs.max().cpu())
                result["entropy"] = get_val(entropy)

                full_results[hook_name] = result
        return full_results

    @torch.no_grad()
    def run_logit_lens_on_neurons_per_token(
        self,
        n_layers,
        sel_neurons_layerwise,
        top_neuron_contvals=None,
        subj_voc_id=None,
        answer_voc_id=None,
        en_subj_voc_id=None,
        en_answer_voc_id=None,
        topk_pred_tokens=None,
    ):
        full_results = {}
        n_test_tokens = len(sel_neurons_layerwise[0])
        for layer in range(n_layers):
            for token_idx in range(n_test_tokens):
                sel_neurons = sel_neurons_layerwise[layer][token_idx]
                residual_after_attn_representation = (
                    self.stateful_model.residual_after_attn(layer)[B0][
                        self.n_demo_tokens :
                    ]
                )
                resid_results_layer_token = self.resid_logit_lens_results[
                    f"{layer}_mid"
                ][token_idx]
                for neuron_idx, neuron in enumerate(sel_neurons):
                    hook_name = f"L{layer}N{neuron}"
                    if hook_name not in full_results:
                        full_results[hook_name] = {}
                        full_results[hook_name]["resid_plus"] = defaultdict(list)

                    neuron_contribution = round(
                        top_neuron_contvals[token_idx][layer][neuron_idx].item(), 4
                    )

                    neuron_ffn_output = self.stateful_model.decomposed_ffn_out(
                        B0, layer, -1, self.n_demo_tokens
                    )
                    token_output_representation = neuron_ffn_output[token_idx, neuron]
                    token_output_representation_plus = (
                        (
                            token_output_representation
                            + residual_after_attn_representation[token_idx]
                        )
                        .unsqueeze(0)
                        .unsqueeze(1)
                    )

                    token_output_logit_scores = self._unembed(
                        token_output_representation_plus
                    )[B0][0]
                    token_output_probs = torch.nn.functional.softmax(
                        token_output_logit_scores, dim=-1
                    ).cpu()
                    token_output_entropy = (
                        (-token_output_probs * torch.log(token_output_probs + 1e-10))
                        .sum(dim=-1)
                        .cpu()
                    )
                    token_output_sorted_indices = torch.argsort(
                        token_output_logit_scores, dim=-1, descending=True
                    ).cpu()

                    top_tokens = [
                        int(i)
                        for i in token_output_sorted_indices[
                            : self._logit_lens_topK
                        ].cpu()
                    ]
                    rank_subject = get_val(
                        (token_output_sorted_indices == subj_voc_id).nonzero(
                            as_tuple=True
                        )[0]
                    )
                    rank_en_subject = get_val(
                        (token_output_sorted_indices == en_subj_voc_id).nonzero(
                            as_tuple=True
                        )[0]
                    )
                    rank_answer = get_val(
                        (token_output_sorted_indices == answer_voc_id).nonzero(
                            as_tuple=True
                        )[0]
                    )
                    rank_en_answer = get_val(
                        (token_output_sorted_indices == en_answer_voc_id).nonzero(
                            as_tuple=True
                        )[0]
                    )
                    rank_topk_pred = (
                        [
                            get_val(
                                (token_output_sorted_indices == pred_voc_id).nonzero(
                                    as_tuple=True
                                )[0]
                            )
                            for pred_voc_id in topk_pred_tokens
                        ]
                        if topk_pred_tokens is not None
                        else None
                    )

                    logit_subject = get_val(
                        token_output_logit_scores[subj_voc_id].cpu()
                    )
                    logit_en_subject = get_val(
                        token_output_logit_scores[en_subj_voc_id].cpu()
                    )
                    logit_answer = get_val(
                        token_output_logit_scores[answer_voc_id].cpu()
                    )
                    logit_en_answer = get_val(
                        token_output_logit_scores[en_answer_voc_id].cpu()
                    )
                    logit_topk_pred = (
                        [
                            get_val(token_output_logit_scores[pred_voc_id].cpu())
                            for pred_voc_id in topk_pred_tokens
                        ]
                        if topk_pred_tokens is not None
                        else None
                    )

                    prob_subject = get_val(token_output_probs[subj_voc_id])
                    prob_en_subject = get_val(token_output_probs[en_subj_voc_id])
                    prob_answer = get_val(token_output_probs[answer_voc_id])
                    prob_en_answer = get_val(token_output_probs[en_answer_voc_id])
                    prob_topk_pred = (
                        [
                            get_val(token_output_probs[pred_voc_id])
                            for pred_voc_id in topk_pred_tokens
                        ]
                        if topk_pred_tokens is not None
                        else None
                    )

                    max_logit = get_val(token_output_logit_scores.max().cpu())
                    max_prob = get_val(token_output_probs.max().cpu())
                    entropy = get_val(token_output_entropy)

                    if "token_idx" in full_results[hook_name]:
                        full_results[hook_name]["token_idx"].append(token_idx)
                        full_results[hook_name]["neuron_contribution"].append(
                            neuron_contribution
                        )
                    else:
                        full_results[hook_name]["token_idx"] = [token_idx]
                        full_results[hook_name]["neuron_contribution"] = [
                            neuron_contribution
                        ]

                        # Input-agnostic neurons (token-agnostic)
                        representations = (
                            self.stateful_model.neuron_output(layer, neuron)
                            .unsqueeze(0)
                            .unsqueeze(1)
                        )
                        logit_scores = self._unembed(representations)[B0][0]
                        probs = torch.nn.functional.softmax(logit_scores, dim=-1).cpu()
                        entropy = (-probs * torch.log(probs + 1e-10)).sum(dim=-1).cpu()
                        sorted_indices = torch.argsort(
                            logit_scores, dim=-1, descending=True
                        ).cpu()

                        full_results[hook_name]["top_tokens"] = [
                            int(i)
                            for i in sorted_indices[: self._logit_lens_topK].cpu()
                        ]

                        full_results[hook_name]["rank_subject"] = get_val(
                            (sorted_indices == subj_voc_id).nonzero(as_tuple=True)[0]
                        )
                        full_results[hook_name]["rank_en_subject"] = get_val(
                            (sorted_indices == en_subj_voc_id).nonzero(as_tuple=True)[0]
                        )
                        full_results[hook_name]["rank_answer"] = get_val(
                            (sorted_indices == answer_voc_id).nonzero(as_tuple=True)[0]
                        )
                        full_results[hook_name]["rank_en_answer"] = get_val(
                            (sorted_indices == en_answer_voc_id).nonzero(as_tuple=True)[
                                0
                            ]
                        )
                        full_results[hook_name]["rank_topk_pred"] = (
                            [
                                get_val(
                                    (sorted_indices == pred_voc_id).nonzero(
                                        as_tuple=True
                                    )[0]
                                )
                                for pred_voc_id in topk_pred_tokens
                            ]
                            if topk_pred_tokens is not None
                            else None
                        )

                        full_results[hook_name]["logit_subject"] = get_val(
                            logit_scores[subj_voc_id].cpu()
                        )
                        full_results[hook_name]["logit_en_subject"] = get_val(
                            logit_scores[en_subj_voc_id].cpu()
                        )
                        full_results[hook_name]["logit_answer"] = get_val(
                            logit_scores[answer_voc_id].cpu()
                        )
                        full_results[hook_name]["logit_en_answer"] = get_val(
                            logit_scores[en_answer_voc_id].cpu()
                        )
                        full_results[hook_name]["logit_topk_pred"] = (
                            [
                                get_val(logit_scores[pred_voc_id].cpu())
                                for pred_voc_id in topk_pred_tokens
                            ]
                            if topk_pred_tokens is not None
                            else None
                        )

                        full_results[hook_name]["prob_subject"] = get_val(
                            probs[subj_voc_id]
                        )
                        full_results[hook_name]["prob_en_subject"] = get_val(
                            probs[en_subj_voc_id]
                        )
                        full_results[hook_name]["prob_answer"] = get_val(
                            probs[answer_voc_id]
                        )
                        full_results[hook_name]["prob_en_answer"] = get_val(
                            probs[en_answer_voc_id]
                        )
                        full_results[hook_name]["prob_topk_pred"] = (
                            [
                                get_val(probs[pred_voc_id])
                                for pred_voc_id in topk_pred_tokens
                            ]
                            if topk_pred_tokens is not None
                            else None
                        )

                        full_results[hook_name]["max_logit"] = get_val(
                            logit_scores.max().cpu()
                        )
                        full_results[hook_name]["max_prob"] = get_val(probs.max().cpu())
                        full_results[hook_name]["entropy"] = get_val(entropy)

                    full_results[hook_name]["resid_plus"]["top_tokens"].append(
                        top_tokens
                    )

                    full_results[hook_name]["resid_plus"]["rank_subject"].append(
                        rank_subject
                    )
                    full_results[hook_name]["resid_plus"]["rank_en_subject"].append(
                        rank_en_subject
                    )
                    full_results[hook_name]["resid_plus"]["rank_answer"].append(
                        rank_answer
                    )
                    full_results[hook_name]["resid_plus"]["rank_en_answer"].append(
                        rank_en_answer
                    )
                    full_results[hook_name]["resid_plus"]["rank_topk_pred"].append(
                        rank_topk_pred
                    )

                    full_results[hook_name]["resid_plus"]["logit_subject"].append(
                        logit_subject
                    )
                    full_results[hook_name]["resid_plus"]["logit_en_subject"].append(
                        logit_en_subject
                    )
                    full_results[hook_name]["resid_plus"]["logit_answer"].append(
                        logit_answer
                    )
                    full_results[hook_name]["resid_plus"]["logit_en_answer"].append(
                        logit_en_answer
                    )
                    full_results[hook_name]["resid_plus"]["logit_topk_pred"].append(
                        logit_topk_pred
                    )

                    full_results[hook_name]["resid_plus"]["prob_subject"].append(
                        prob_subject
                    )
                    full_results[hook_name]["resid_plus"]["prob_en_subject"].append(
                        prob_en_subject
                    )
                    full_results[hook_name]["resid_plus"]["prob_answer"].append(
                        prob_answer
                    )
                    full_results[hook_name]["resid_plus"]["prob_en_answer"].append(
                        prob_en_answer
                    )
                    full_results[hook_name]["resid_plus"]["prob_topk_pred"].append(
                        prob_topk_pred
                    )

                    full_results[hook_name]["resid_plus"]["max_logit"].append(max_logit)
                    full_results[hook_name]["resid_plus"]["max_prob"].append(max_prob)
                    full_results[hook_name]["resid_plus"]["entropy"].append(entropy)

                    diff_keys = [
                        key
                        for key in resid_results_layer_token.keys()
                        if key
                        not in [
                            "token_idx",
                            "attn_contribution",
                            "top_tokens",
                            "max_logit",
                            "max_prob",
                            "top_token_strings",
                            "resid_plus",
                        ]
                    ]
                    for key in diff_keys:
                        full_results[hook_name]["resid_plus"][f"{key}_diff"] = []
                        if key.endswith("_topk_pred"):
                            for v in full_results[hook_name]["resid_plus"][key]:
                                full_results[hook_name]["resid_plus"][
                                    f"{key}_diff"
                                ].append(
                                    [
                                        v1 - v2
                                        for v1, v2 in zip(
                                            v, resid_results_layer_token[key]
                                        )
                                    ]
                                )
                        else:
                            for v in full_results[hook_name]["resid_plus"][key]:
                                full_results[hook_name]["resid_plus"][
                                    f"{key}_diff"
                                ].append(v - resid_results_layer_token[key])

        return full_results

    def process_sentences(
        self,
        sentences,
        all_subject_positions,
        facts,
        subjects,
        targets,
        subjects_en,
        targets_en,
        demos,
        relation,
        indices,
        rank_topk_pred=5,
    ):
        relation_sentence_analyses = (
            []
        )  # List to hold analyses for the current relation
        print(f"Relation {relation} - *{args.language}*")
        for idx, sent in enumerate(tqdm(sentences)):
            start_time = time.time()
            self.sentence = sent
            self.subj_token = subjects[idx]
            self.obj_token = targets[idx]
            self.en_subj_token = subjects_en[idx]
            self.en_obj_token = targets_en[idx]
            self.demo = demos[idx] if len(demos) != 0 else ""
            self.test_sentence = sent.replace(self.demo, "")

            torch.cuda.empty_cache()

            # Run inference
            self.run_inference()
            """Let the model generate a few tokens (e.g., 10)
            case 1: if the rank_answer = 0 and the following toknes are also correct -> correct
            case 2: if the rank_answer = 0 but the following tokens are wrong -> filter out
            case 3: if the rank_answer != 0 -> incorrect
            
            For case 3, also save the incorrect prediction tokens and strings, and track their ranks
            in the following 'run_logit_lens_on...' functions
            """

            tokens = self.stateful_model.tokens()[B0]
            subj_tokens = self.stateful_model.subj_tokens()[B0]
            obj_tokens = self.stateful_model.obj_token()[B0]
            demo_tokens = self.stateful_model._last_run.demo_tokens[B0]
            en_subj_tokens = self.stateful_model._last_run.en_subj_tokens[B0]
            en_obj_tokens = self.stateful_model._last_run.en_obj_tokens[B0]

            if args.language in ["ja", "zh"] and "llama" in self.model_name.lower():
                try:
                    if obj_tokens == en_obj_tokens:
                        pass
                except:
                    obj_tokens = obj_tokens[1:]
                if args.few_shot_demo != 0:
                    try:
                        if subj_tokens == en_subj_tokens:
                            pass
                    except:
                        subj_tokens = subj_tokens[1:]
            n_tokens = tokens.shape[0]
            n_demo_tokens = (
                demo_tokens.shape[0]
                if args.language in ["ja", "zh"] or "bloom" in self.model_name.lower()
                else demo_tokens.shape[0] - 1
            )  # one additional <eos> token
            n_demo_tokens = max(0, n_demo_tokens)

            self.n_tokens = n_tokens
            self.n_demo_tokens = n_demo_tokens

            # print(f"Number of Tokens: {n_tokens}")
            if "llama" in self.model_name.lower():
                tokenized_sentence = (
                    self.stateful_model._model.tokenizer.convert_ids_to_tokens(tokens)
                )
            elif "bloom" in self.model_name.lower():
                tokenized_sentence = [
                    self.stateful_model._model.tokenizer.decode(token)
                    for token in tokens
                ]
            assert len(tokenized_sentence) == n_tokens

            if "llama" in self.model_name.lower():
                test_tokenized_sentence = (
                    self.stateful_model._model.tokenizer.convert_ids_to_tokens(
                        tokens[n_demo_tokens:n_tokens]
                    )
                )
            elif "bloom" in self.model_name.lower():
                test_tokenized_sentence = [
                    self.stateful_model._model.tokenizer.decode(token)
                    for token in tokens[n_demo_tokens:n_tokens]
                ]
            assert len(test_tokenized_sentence) == n_tokens - n_demo_tokens

            subj_token_span = find_span_indices(tokens, subj_tokens)
            obj_token_span = find_span_indices(tokens, obj_tokens)

            subj_token_span_test = find_span_indices(
                tokens[n_demo_tokens:n_tokens], subj_tokens
            )
            obj_token_span_test = find_span_indices(
                tokens[n_demo_tokens:n_tokens], obj_tokens
            )
            self.input_last_token = obj_token_span_test[0] - 1

            subj_voc_id = subj_tokens[
                0
            ].cpu()  # tokens[all_subject_positions[idx]].cpu()
            answer_voc_id = obj_tokens[0].cpu()  # tokens[-1].cpu()

            en_subj_voc_id = en_subj_tokens[
                0
            ].cpu()  # tokens[all_subject_positions[idx]].cpu()
            en_answer_voc_id = en_obj_tokens[0].cpu()  # tokens[-1].cpu()

            model_info = self.stateful_model.model_info()

            # # Build contribution graphs
            # graphs = cached_build_paths_to_predictions(
            #     self._graph,
            #     model_info.n_layers,
            #     n_tokens,
            #     range(n_tokens),
            #     self._contribution_threshold,
            # )
            # print(f"Done with Graph {time.time() - start_time}")

            # Get the top-k final prediction tokens
            if rank_topk_pred is not None:
                final_representation = self.stateful_model.residual_out(
                    model_info.n_layers - 1
                )[B0][obj_token_span[0] - 1, :]
                final_logit_scores = self._unembed(final_representation)
                sorted_indices = torch.argsort(
                    final_logit_scores, dim=-1, descending=True
                )
                topk_pred_tokens = sorted_indices[:rank_topk_pred].cpu()
            else:
                topk_pred_tokens = None

            # Run logit lens on various outputs
            resid_logit_lens_results, representation_results = (
                self.run_logit_lens_on_resid(
                    model_info.n_layers,
                    n_tokens,
                    n_demo_tokens,
                    subj_voc_id=subj_voc_id,
                    answer_voc_id=answer_voc_id,
                    en_subj_voc_id=en_subj_voc_id,
                    en_answer_voc_id=en_answer_voc_id,
                    topk_pred_tokens=topk_pred_tokens,
                )
            )
            self.resid_logit_lens_results = resid_logit_lens_results
            output_logit_lens_results = self.run_logit_lens_on_outputs(
                model_info.n_layers,
                n_tokens,
                n_demo_tokens,
                subj_voc_id=subj_voc_id,
                answer_voc_id=answer_voc_id,
                en_subj_voc_id=en_subj_voc_id,
                en_answer_voc_id=en_answer_voc_id,
                topk_pred_tokens=topk_pred_tokens,
            )

            if self._log_hidden_states:
                save_hidden_states_per_instance(
                    representation_results,
                    relation,
                    args.language,
                    indices[idx],
                    self.revision_output_dir,
                )
                continue
            final_rank_answer = resid_logit_lens_results["final_post"][
                obj_token_span_test[0] - 1
            ]["rank_answer"]

            first_pred_token = resid_logit_lens_results["final_post"][
                obj_token_span_test[0] - 1
            ]["top_tokens"][0]
            first_pred_token = torch.tensor([first_pred_token], device=tokens.device)
            updated_tokens = torch.cat((tokens[: obj_token_span[0]], first_pred_token))
            pred_tokens = self._stateful_model._model.generate(
                updated_tokens[None, :], do_sample=False
            )[0, obj_token_span[0] :]
            pred_obj = self.stateful_model._model.tokenizer.decode(pred_tokens)
            if final_rank_answer == 0 and pred_obj.startswith(self.obj_token[:-1]):
                # correct
                pred_answer = self.obj_token
                correctness_group = "correct"
            else:
                # incorrect
                match = re.match(r"^[^,.;?!，。；？！]*", pred_obj)
                try:
                    pred_answer = match.group(0)
                except:
                    pred_answer = pred_obj
                if final_rank_answer == 0:
                    correctness_group = "other"
                else:
                    correctness_group = "incorrect"

            # self._stateful_model._model.generate(tokens[None, :], do_sample=False)[0, answer_start_token]
            self._contributions_dict_cpu = {}

            if self._run_logit_lens_on_heads:
                heads_logit_lens_results = self.run_logit_lens_on_heads(
                    model_info.n_layers,
                    model_info.n_heads,
                    n_tokens,
                    n_demo_tokens,
                    subj_voc_id=subj_voc_id,
                    answer_voc_id=answer_voc_id,
                    en_subj_voc_id=en_subj_voc_id,
                    en_answer_voc_id=en_answer_voc_id,
                    topk_pred_tokens=topk_pred_tokens,
                )
                # for key, tensor_list in self._contributions_dict.items():
                #     self._contributions_dict_cpu[key] = [tensor.cpu() for tensor in tensor_list]
            else:
                heads_logit_lens_results = None
            # Create sentence analysis dictionary
            # Retrieve everything to cpu and save

            sentence_analysis = {
                "sentence": self.sentence,
                # "tokenized_sentence": tokenized_sentence,
                # "tokens": tokens.tolist(),
                "test_sentence": self.test_sentence,
                "correctness": correctness_group,
                "test_tokenized_sentence": test_tokenized_sentence,
                "test_tokens": tokens[n_demo_tokens:n_tokens].tolist(),
                "subject_tokens": subj_tokens.tolist(),
                "answer_tokens": obj_tokens.tolist(),
                "en_subject_tokens": en_subj_tokens.tolist(),
                "en_answer_tokens": en_obj_tokens.tolist(),
                "answer_strings": self.obj_token,
                "pred_answer_strings": pred_answer,
                "data_idx": indices[idx],
                # "subj_token_span": subj_token_span,
                # "answer_token_span": obj_token_span,
                "subj_token_span_test": subj_token_span_test,
                "answer_token_span_test": obj_token_span_test,
                "contributions": self._contributions_dict_cpu,
                # "full_graph": self._graph.copy(),
                # "token_subgraphs": graphs,
                "logit_lens_result": {
                    "resid": resid_logit_lens_results,
                    "output": output_logit_lens_results,
                    "heads": heads_logit_lens_results,
                },
            }

            # If neuron level analysis is enabled
            if self._do_neuron_level:
                neuron_contributions = self.compute_neuron_contributions(
                    model_info.n_layers, n_tokens, n_demo_tokens
                )
                sentence_analysis["neuron_contributions_all"] = (
                    neuron_contributions.cpu()
                )

                if self._neuron_extraction_mode == "topk":
                    top_neuron_contvals, top_neuron_indices = torch.topk(
                        neuron_contributions, k=self._logit_lens_topK_neurons
                    )
                    top_neuron_indices = top_neuron_indices.cpu().numpy()
                    top_neuron_contvals = (
                        top_neuron_contvals.cpu().to(torch.float16).numpy()
                    )

                    sel_neurons_layerwise = []
                    sel_neurons_layerwise = [[] for _ in range(model_info.n_layers)]
                    for layer in range(model_info.n_layers):
                        sel_neurons_layerwise[layer] = [
                            [] for _ in range(n_tokens - n_demo_tokens)
                        ]
                        for token in range(n_tokens - n_demo_tokens):
                            sel_neurons_layerwise[layer][token] = top_neuron_indices[
                                token, layer
                            ].tolist()
                elif self._neuron_extraction_mode == "threshold":

                    def get_values_and_indices(tensor, threshold=0.01):
                        # Create a mask of values greater than the threshold
                        mask = tensor > threshold

                        # Get the values that satisfy the condition
                        values = tensor[mask]

                        # Get the indices of these values
                        indices = torch.nonzero(mask, as_tuple=False)

                        return values, indices

                    top_neuron_contvals, top_neuron_indices = get_values_and_indices(
                        neuron_contributions, self._contribution_threshold
                    )
                    sel_neurons_layerwise = [[] for _ in range(model_info.n_layers)]
                    top_neuron_contvals_list = [
                        [] for _ in range(n_tokens - n_demo_tokens)
                    ]
                    for layer in range(model_info.n_layers):
                        sel_neurons_layerwise[layer] = [
                            [] for _ in range(n_tokens - n_demo_tokens)
                        ]
                    for token in range(n_tokens - n_demo_tokens):
                        top_neuron_contvals_list[token] = [
                            [] for _ in range(model_info.n_layers)
                        ]

                    for index in top_neuron_indices:
                        token = index[0].item()
                        layer = index[1].item()
                        neuron_id = index[2].item()
                        sel_neurons_layerwise[layer][token].append(neuron_id)

                    for index, val in enumerate(top_neuron_contvals):
                        neuron_pos = top_neuron_indices[index]
                        token = neuron_pos[0].item()
                        layer = neuron_pos[1].item()
                        top_neuron_contvals_list[token][layer].append(
                            val.cpu().to(torch.float16).numpy()
                        )

                    top_neuron_contvals = top_neuron_contvals_list

                run_logit_lens_on_neurons = self.run_logit_lens_on_neurons_per_token(
                    model_info.n_layers,
                    sel_neurons_layerwise=sel_neurons_layerwise,
                    top_neuron_contvals=top_neuron_contvals,
                    subj_voc_id=subj_voc_id,
                    answer_voc_id=answer_voc_id,
                    en_subj_voc_id=en_subj_voc_id,
                    en_answer_voc_id=en_answer_voc_id,
                    topk_pred_tokens=topk_pred_tokens,
                )
                # print(f"Done with Neurons {time.time() - start_time}")
                sentence_analysis["neuron_contributions"] = {
                    "vals": top_neuron_contvals,
                    "ind": top_neuron_indices,
                }

                for k, v in run_logit_lens_on_neurons.items():
                    token_ids = v["top_tokens"]
                    token_strings = (
                        self._stateful_model._model.tokenizer.convert_ids_to_tokens(
                            token_ids
                        )
                    )
                    v["top_token_strings"] = token_strings

                    v["resid_plus"]["top_token_strings"] = []
                    for token_ids in v["resid_plus"]["top_tokens"]:
                        token_strings = (
                            self._stateful_model._model.tokenizer.convert_ids_to_tokens(
                                token_ids
                            )
                        )
                        v["resid_plus"]["top_token_strings"].append(token_strings)

                sentence_analysis["logit_lens_result"][
                    "neurons"
                ] = run_logit_lens_on_neurons

            # Append sentence analysis for this sentence to the relation list
            relation_sentence_analyses.append(sentence_analysis)

        return relation_sentence_analyses

    def run(self, args):
        self.load_config(args.config_file)
        self._log_hidden_states = True if args.log_hidden_states == "True" else False
        self._stateful_model = load_model(
            model_name=self.model_name,
            revision=args.revision,
            prepend_bos=self._prepend_bos,
            _model_path=self._model_path,
            _device=self.device,
            _dtype=self.dtype,
        )

        args.dataset_path += f"/{args.language}"
        relations = load_json_files(args.dataset_path, selected_relation=args.relation)
        all_sentence_analyses = {}

        # Create the base output directory (based on model name)
        base_output_dir = args.output_path

        # Create the revision-specific directory
        self.revision_output_dir = os.path.join(base_output_dir, args.revision)

        with torch.inference_mode():
            no_entity_space = args.language in ["ja", "zh"]
            # If a specific relation is provided, process only that relation
            if args.relation:
                (
                    sentences,
                    facts,
                    subjects,
                    targets,
                    subjects_en,
                    targets_en,
                    demos,
                    all_subject_positions,
                    indices,
                ) = parse_samples(
                    relations[args.relation],
                    no_entity_space=no_entity_space,
                    few_shot_demo=args.few_shot_demo,
                )
                # print(len(sentences))
                if "llama" in self.model_name.lower():
                    subjects = [s.strip() for s in subjects]
                    targets = [t.strip() for t in targets]
                    subjects_en = [s.strip() for s in subjects_en]
                    targets_en = [t.strip() for t in targets_en]
                relation_sentence_analyses = self.process_sentences(
                    sentences,
                    all_subject_positions,
                    facts,
                    subjects,
                    targets,
                    subjects_en,
                    targets_en,
                    demos,
                    args.relation,
                    indices,
                )
                if not self._log_hidden_states:
                    all_sentence_analyses[args.relation] = relation_sentence_analyses
                    save_analysis_per_relation(
                        relation_sentence_analyses,
                        args.relation,
                        self.revision_output_dir,
                        args.language,
                    )
            else:
                # Process all relations
                for relation, relation_data in tqdm(
                    relations.items(), desc="Processing Relations"
                ):
                    # print(f"Relation {relation}")
                    (
                        sentences,
                        facts,
                        subjects,
                        targets,
                        subjects_en,
                        targets_en,
                        demos,
                        all_subject_positions,
                        indices,
                    ) = parse_samples(
                        relations[args.relation],
                        no_entity_space=no_entity_space,
                        few_shot_demo=args.few_shot_demo,
                    )
                    # print(len(sentences))
                    if "llama" in self.model_name.lower():
                        subjects = [s.strip() for s in subjects]
                        targets = [t.strip() for t in targets]
                        subjects_en = [s.strip() for s in subjects_en]
                        targets_en = [t.strip() for t in targets_en]
                    relation_sentence_analyses = self.process_sentences(
                        sentences,
                        all_subject_positions,
                        facts,
                        subjects,
                        targets,
                        subjects_en,
                        targets_en,
                        demos,
                        args.relation,
                        indices,
                    )
                    if not self._log_hidden_states:
                        all_sentence_analyses[relation] = relation_sentence_analyses

                        save_analysis_per_relation(
                            relation_sentence_analyses,
                            relation,
                            self.revision_output_dir,
                            args.language,
                        )

        # Return or save all_sentence_analyses as needed
        return all_sentence_analyses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="llm-transparency-tool-bak/config/backend_llama-2-7b-baseline.json",
        help="Model config file to use",
    )
    parser.add_argument(
        "--revision", type=str, default="", help="Model revision to use"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="klar",
        help="Dataset to use",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="factual",
        help="Select from Main Categories: Factual, Commonsense, Bias, Linguistic",
    )
    parser.add_argument(
        "--relation",
        type=str,
        default="native_language",
        help="Specific relation from Category",
    )
    parser.add_argument(
        "--language", type=str, default="en", help="Factual probing in which language"
    )
    parser.add_argument(
        "--few_shot_demo", type=int, default=3, help="Number of few-shot demostartions"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="llm-transparency-tool-bak/backend_results_llama2_debug",
        help="Output path for saved pickle file",
    )
    parser.add_argument("--log_hidden_states", type=str, default="False")
    parser.add_argument("--seed", type=int, default=26)
    args = parser.parse_args()

    set_seed(args.seed)
    app = App()
    app.run(args)
