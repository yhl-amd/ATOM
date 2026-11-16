# SPDX-License-Identifier: MIT
"""Aux taps whose `extract` writes the capture buffer's rows itself
(`AuxCaptureSpec.writes_into`): the hook offers the rows, and copies nothing
when they come back."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip(
    "atom.spec_decode.drafter",
    reason="requires AITER to import the drafter chain",
    exc_type=ImportError,
)

from atom.spec_decode import drafter as drafter_module
from atom.spec_decode.drafter import Drafter


def _context(monkeypatch, *, offset=0, order=None, is_draft=False):
    ctx = SimpleNamespace(
        is_draft=is_draft, ubatch_token_offset=offset, ubatch_token_order=order
    )
    monkeypatch.setattr(
        drafter_module, "get_forward_context", lambda: SimpleNamespace(context=ctx)
    )


def _hook(buffer, extract, writes_into):
    drafter = SimpleNamespace(_aux_buffers=[buffer])
    return Drafter._make_aux_hook(drafter, 0, extract, writes_into=writes_into)


def _writing_extract(calls):
    def extract(output, module, out_for=None):
        calls.append(out_for)
        if out_for is None:
            return output * 2
        out = out_for(output.shape[0])
        if out.shape != output.shape:  # as post_collapse_streams does
            return output * 2
        torch.mul(output, 2, out=out)
        calls.append(out.data_ptr())
        return out

    return extract


@pytest.mark.parametrize("offset", [0, 5])
def test_extract_writes_the_offered_rows(monkeypatch, offset):
    _context(monkeypatch, offset=offset)
    buffer = torch.zeros(16, 4)
    output = torch.arange(12.0).view(3, 4)
    calls = []
    hook = _hook(buffer, _writing_extract(calls), writes_into=True)
    hook(None, (), output)
    assert calls[0] is not None
    assert calls[1] == buffer[offset].data_ptr()  # written where it lives
    assert torch.equal(buffer[offset : offset + 3], output * 2)
    assert not buffer[:offset].any() and not buffer[offset + 3 :].any()


def test_a_returned_new_tensor_is_still_copied(monkeypatch):
    _context(monkeypatch, offset=2)
    buffer = torch.zeros(8, 4)
    output = torch.ones(3, 4)

    def extract(output, module, out_for=None):
        return output + 1  # ignores the offered rows

    hook = _hook(buffer, extract, writes_into=True)
    hook(None, (), output)
    assert torch.equal(buffer[2:5], output + 1)


def test_micro_batch_token_order_keeps_the_copy(monkeypatch):
    order = torch.tensor([6, 1, 3])
    _context(monkeypatch, order=order)
    buffer = torch.zeros(8, 4)
    output = torch.arange(12.0).view(3, 4)
    calls = []
    hook = _hook(buffer, _writing_extract(calls), writes_into=True)
    hook(None, (), output)
    assert calls == [None]
    assert torch.equal(buffer[order], output * 2)


def test_without_writes_into_extract_is_called_as_before(monkeypatch):
    _context(monkeypatch, offset=1)
    buffer = torch.zeros(8, 4)
    output = torch.ones(2, 4)
    seen = []

    def extract(output, module):
        seen.append(True)
        return output * 3

    hook = _hook(buffer, extract, writes_into=False)
    hook(None, (), output)
    assert seen == [True]
    assert torch.equal(buffer[1:3], output * 3)


def test_draft_forwards_are_skipped(monkeypatch):
    _context(monkeypatch, is_draft=True)
    buffer = torch.zeros(4, 4)
    hook = _hook(buffer, _writing_extract([]), writes_into=True)
    hook(None, (), torch.ones(2, 4))
    assert not buffer.any()


def test_rows_past_the_buffer_still_raise(monkeypatch):
    _context(monkeypatch, offset=3)
    buffer = torch.zeros(4, 4)
    hook = _hook(buffer, _writing_extract([]), writes_into=True)
    with pytest.raises(RuntimeError):
        hook(None, (), torch.ones(3, 4))
