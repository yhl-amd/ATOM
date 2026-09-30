# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Capacity of the two TBO ubatch MoRI ops (ATOM_MORI_TBO_HALF_BUFFERS).

Halving is only sound while no ubatch can carry more than half of a rank's
token budget. MoRI does not bound-check its input, so a ubatch past the op's
capacity writes off the end of the symmetric buffers instead of failing. These
pin the split that makes half enough, and the cases that must keep full size.
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

import torch

import atom.model_ops.fused_moe.mori_prepare_finalize as mpf
from atom.utils.tbo.ubatch_splitting import maybe_create_ubatch_slices
from atom.utils.tbo.ubatching import _precompute_prefill_token_split

MAX_TOKENS = 16384


def _config(**kw):
    base = {"enable_tbo_decode": False, "prefill_context_parallel_size": 1}
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def half_on(monkeypatch):
    monkeypatch.setattr(mpf, "_TBO_HALF_BUFFERS", True)
    monkeypatch.setenv("ATOM_TBO_PREFILL_TOKEN_SPLIT", "1")


def test_off_keeps_the_full_budget(monkeypatch):
    monkeypatch.setattr(mpf, "_TBO_HALF_BUFFERS", False)
    assert mpf.tbo_max_tokens_per_rank(MAX_TOKENS, _config()) == MAX_TOKENS


@pytest.mark.parametrize("budget,expected", [(16384, 8192), (16383, 8192), (1, 1)])
def test_token_split_prefill_gets_half(half_on, budget, expected):
    assert mpf.tbo_max_tokens_per_rank(budget, _config()) == expected


@pytest.mark.parametrize(
    "config",
    [
        _config(enable_tbo_decode=True),
        _config(prefill_context_parallel_size=2),
    ],
)
def test_uneven_splits_keep_full_size(half_on, config):
    assert mpf.tbo_max_tokens_per_rank(MAX_TOKENS, config) == MAX_TOKENS


def test_request_boundary_split_keeps_full_size(half_on, monkeypatch):
    monkeypatch.setenv("ATOM_TBO_PREFILL_TOKEN_SPLIT", "0")
    assert mpf.tbo_max_tokens_per_rank(MAX_TOKENS, _config()) == MAX_TOKENS


@pytest.mark.parametrize(
    "lens",
    [
        [MAX_TOKENS],
        [MAX_TOKENS - 1],
        [1, MAX_TOKENS - 1],
        [MAX_TOKENS - 1, 1],
        [3, 5000, 11381],
        [2],
    ],
)
def test_no_token_split_ubatch_exceeds_half(half_on, lens):
    """Both the DP-reduced sizes and the realised slices, forced or not: a rank
    below the min-token bar is still split when a peer clears it."""
    capacity = mpf.tbo_max_tokens_per_rank(MAX_TOKENS, _config())
    toks = np.asarray(lens, dtype=np.int32)
    _, can_split, ub0, ub1 = _precompute_prefill_token_split(toks, len(lens), 8192)
    assert can_split and max(ub0, ub1) <= capacity
    slices = maybe_create_ubatch_slices(
        num_reqs=len(lens),
        num_tokens=int(toks.sum()),
        is_prefill=True,
        num_scheduled_tokens=toks,
        force=True,
    )
    widths = [s.token_slice.stop - s.token_slice.start for s in slices]
    assert widths == [ub0, ub1]


def test_an_oversized_ubatch_is_refused_before_dispatch():
    pf = mpf.MoriPrepareAndFinalize.__new__(mpf.MoriPrepareAndFinalize)
    pf._tbo_max_tokens_per_rank = 8
    a1 = torch.zeros(9, 32, dtype=torch.bfloat16)
    ids = torch.zeros(9, 6, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="holds 8 per rank"):
        pf.prepare_async(a1, torch.ones(9, 6), ids, 384, None, False)
