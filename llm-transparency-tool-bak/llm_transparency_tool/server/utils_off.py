# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union, Dict

import networkx as nx

# import streamlit as st
import torch

import llm_transparency_tool.routes.graph_off
from llm_transparency_tool.models.tlens_model_off import TransformerLensTransparentLlm
from llm_transparency_tool.models.transparent_llm import TransparentLlm

GPU = "gpu"
CPU = "cpu"

# This variable is for expressing the idea that batch_id = 0, but make it more
# readable than just 0.
B0 = 0


def get_val(x: torch.Tensor):
    # return x.squeeze().to(torch.float16).numpy()
    return x.squeeze().item()


def possible_devices() -> List[str]:
    devices = []
    if torch.cuda.is_available():
        devices.append("gpu")
    devices.append("cpu")
    return devices


def load_dataset(filename) -> List[str]:
    with open(filename) as f:
        dataset = [s.strip("\n") for s in f.readlines()]
    print(f"Loaded {len(dataset)} sentences from {filename}")
    return dataset


# @st.cache_resource(
#     hash_funcs={
#         TransformerLensTransparentLlm: id
#     }
# )
def load_model(
    model_name: str,
    revision: str,
    _device: str,
    _model_path: Optional[str] = None,
    _dtype: torch.dtype = torch.float32,
    prepend_bos: bool = True,
) -> TransparentLlm:
    """
    Returns the loaded model along with its key. The key is just a unique string which
    can be used later to identify if the model has changed.
    """
    assert _device in possible_devices()

    causal_lm = None
    tokenizer = None

    tl_lm = TransformerLensTransparentLlm(
        model_name=model_name,
        revision=revision,
        model_path=_model_path,
        hf_model=causal_lm,
        tokenizer=tokenizer,
        prepend_bos=prepend_bos,
        device=_device,
        dtype=_dtype,
    )

    return tl_lm


def run_model(model: TransparentLlm, sentence: str) -> None:
    print(f"Running inference for '{sentence}'")
    model.run([sentence])


def load_model_with_session_caching(
    **kwargs,
) -> Tuple[TransparentLlm, str]:
    return load_model(**kwargs)


def run_model_with_session_caching(
    _model: TransparentLlm,
    model_key: str,
    sentence: str,
):

    run_model(_model, sentence)


# @st.cache_resource(
#     hash_funcs={
#         TransformerLensTransparentLlm: id
#     }
# )
def get_contribution_graph(
    model: TransparentLlm,  # TODO bug here
    model_key: str,
    tokens: List[str],
    threshold: float,
    n_demo_tokens: int = 0,
) -> Union[nx.Graph, Dict[str, torch.Tensor]]:
    """
    The `model_key` and `tokens` are used only for caching. The model itself is not
    hashed, hence the `_` in the beginning.
    """
    return llm_transparency_tool.routes.graph_off.build_full_graph(
        model,
        B0,
        threshold,
        n_demo_tokens,
    )
