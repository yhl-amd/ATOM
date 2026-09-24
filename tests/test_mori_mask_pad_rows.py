# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DP pad rows routed to expert -1 before the mori dispatch.

A padded decode step runs `running_tokens` rows and only the leading
`scheduled_tokens` carry a request. mori IntraNode dispatch sends nothing for a
negative expert id, so masking the tail makes padding free -- but only the
tail: a real row masked by mistake comes back from combine as zeros, which is a
silent accuracy loss, not an error. These pin which rows are touched.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

import torch

import atom.model_ops.fused_moe.mori_prepare_finalize as mpf
import atom.utils.forward_context as fc

TOPK = 6


def _pf(monkeypatch, *, scheduled, running, capturing=False, is_async=False):
    context = SimpleNamespace(scheduled_tokens=scheduled, running_tokens=running)
    monkeypatch.setattr(
        mpf, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    pf = mpf.MoriPrepareAndFinalize.__new__(mpf.MoriPrepareAndFinalize)
    pf._is_async = is_async
    pf._pad_row_index = torch.arange(256, dtype=torch.int32).unsqueeze(1)
    pf._scheduled_tokens_device = torch.tensor([scheduled], dtype=torch.int32)
    return pf


def _ids(rows):
    return torch.arange(rows * TOPK, dtype=torch.int32).reshape(rows, TOPK) % 384


@pytest.mark.parametrize("capturing", [False, True])
def test_only_the_tail_past_scheduled_is_masked(monkeypatch, capturing):
    pf = _pf(monkeypatch, scheduled=12 * 7, running=16 * 7, capturing=capturing)
    ids = _ids(16 * 7)
    out = pf.mask_pad_topk_ids(ids)
    assert out.dtype == torch.int32
    assert torch.equal(out[: 12 * 7], ids[: 12 * 7])
    assert (out[12 * 7 :] == -1).all()
    # The router's own tensor is not written.
    assert torch.equal(ids, _ids(16 * 7))


def test_the_capture_reads_the_device_count_not_the_host_one(monkeypatch):
    """A recording must follow what is published at replay, not what the
    capture context said -- which is always "every row is real"."""
    pf = _pf(monkeypatch, scheduled=32, running=32, capturing=True)
    pf._scheduled_tokens_device.fill_(20)
    out = pf.mask_pad_topk_ids(_ids(32))
    assert (out[:20] >= 0).all() and (out[20:] == -1).all()


def test_an_unpadded_step_is_returned_as_is(monkeypatch):
    pf = _pf(monkeypatch, scheduled=40, running=40)
    ids = _ids(40)
    assert pf.mask_pad_topk_ids(ids) is ids


@pytest.mark.parametrize("rows", [7, 48])
def test_rows_that_are_not_the_step_width_are_left_alone(monkeypatch, rows):
    """Some other row set (a PCP shard, a sub-slice): it has no padded tail."""
    pf = _pf(monkeypatch, scheduled=5, running=32)
    ids = _ids(rows)
    assert pf.mask_pad_topk_ids(ids) is ids


def test_tbo_ubatches_are_left_alone(monkeypatch):
    pf = _pf(monkeypatch, scheduled=5, running=32, is_async=True)
    monkeypatch.setattr(pf, "supports_async", lambda: True)
    ids = _ids(32)
    assert pf.mask_pad_topk_ids(ids) is ids


def test_disabled_is_identity(monkeypatch):
    pf = _pf(monkeypatch, scheduled=5, running=32)
    pf._pad_row_index = None
    ids = _ids(32)
    assert pf.mask_pad_topk_ids(ids) is ids


def test_publish_is_a_no_op_until_enabled_and_skipped_while_capturing(monkeypatch):
    monkeypatch.setattr(fc, "_scheduled_tokens_device", None)
    fc.publish_scheduled_tokens(7)  # nothing allocated, nothing written
    buf = fc.enable_scheduled_tokens_device(torch.device("cpu"))
    assert buf.item() == torch.iinfo(torch.int32).max
    assert fc.enable_scheduled_tokens_device(torch.device("cpu")) is buf
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    fc.publish_scheduled_tokens(84)
    assert buf.item() == 84
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    fc.publish_scheduled_tokens(3)
    assert buf.item() == 84
