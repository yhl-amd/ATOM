# SPDX-License-Identifier: MIT
"""Packed block-table publishes send only the rows the destination lacks, and
only as wide as needed, yet leave the destination byte-identical to a
whole-prefix copy of every staged table."""

import os
import random

import numpy as np
import pytest
import torch

from atom.model_engine.sequence import new_block_table
from atom.utils import CpuGpuBuffer
from atom.utils.block_tables import block_table_state
from atom.utils.h2d import PublicationOwner

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_H2D_GPU_TESTS") != "1", reason="set RUN_H2D_GPU_TESTS=1"
)


def _buffers(rows, columns, extra=2):
    table = CpuGpuBuffer(rows, columns, dtype=torch.int32, device="cuda")
    others = [
        CpuGpuBuffer(97 + i, dtype=torch.int32, device="cuda") for i in range(extra)
    ]
    return table, others


def _owner(table, others):
    owner = PublicationOwner("cuda", torch.cuda.Event())
    members = [owner.bind(table, "block_tables")]
    members += [owner.bind(b, f"o{i}") for i, b in enumerate(others)]
    group = owner.group("step", members)
    assert group.use_packed() == "packed"
    return owner, group


def test_scatter_rows_kernel_matches_row_writes():
    """The word kernel's row path: arbitrary increasing rows, any 16-byte width."""
    rng = np.random.default_rng(0)
    table, others = _buffers(64, 1024)
    owner, group = _owner(table, others)
    binding = table._publication
    reference = torch.zeros_like(table.gpu)
    for step in range(60):
        owner.begin()
        table.np[:] = rng.integers(0, 1 << 30, size=table.np.shape, dtype=np.int32)
        for b in others:
            b.np[:] = step
        count = int(rng.integers(1, 65))
        k = int(rng.integers(1, count + 1))
        rows = np.sort(rng.choice(count, size=k, replace=False)).astype(np.int32)
        width = int(rng.integers(1, 257)) * 4
        if step % 7 == 0:
            rows, width = np.arange(count, dtype=np.int32), 1024
        binding.row_plan = (count, rows, width * 4)
        group.set_count(table, count)
        for b in others:
            group.set_count(b, int(rng.integers(1, 98)))
        group.publish()
        owner.finish()
        reference[torch.from_numpy(rows).long().cuda(), :width] = torch.from_numpy(
            table.np[rows, :width]
        ).cuda()
        owner.completion.synchronize()
        assert torch.equal(table.gpu, reference), step


def _ids(rng, n, page_limit):
    return rng.integers(0, page_limit, size=n).tolist()


def _simulate(seed, steps, columns, capacity, packed=True):
    """A scheduler's churn against the full-copy reference: appends, finishes
    (later rows shift up), admissions (with prefix hits), preemption,
    speculative rollback, bucket changes, prefill interleaving, unversioned
    and dummy rows, rows of exactly ``columns`` pages."""
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    page_limit = 1 << 20
    table, others = _buffers(capacity, columns)
    owner, group = _owner(table, others)
    if not packed:
        group._backend, group.transport = None, "direct"
    state = block_table_state(table)
    reference = table.gpu.clone()
    running = [
        new_block_table(_ids(rng, int(rng.integers(1, 64)), page_limit))
        for _ in range(40)
    ]
    sent = full = 0
    buckets = [b for b in (8, 16, 24, 32, 40, 48, 56, 64, 96, 128) if b <= capacity]
    for step in range(steps):
        owner.begin()
        kind = pyrng.random()
        if kind < 0.08:  # a prefill step: another batch in the same buffer
            rows = [
                new_block_table(
                    _ids(rng, int(rng.integers(1, columns + 1)), page_limit)
                )
                for _ in range(pyrng.randint(1, 6))
            ]
            if pyrng.random() < 0.5:  # a prefix hit shares a running request's pages
                src = pyrng.choice(running)
                rows[0] = new_block_table(list(src[: pyrng.randint(1, len(src))]) + [7])
            count = len(rows)
        elif kind < 0.1:  # a dummy run: private arange rows
            n = pyrng.randint(1, 8)
            rows = [np.arange(i * 3, i * 3 + 3, dtype=np.int32) for i in range(n)]
            count = pyrng.choice([b for b in buckets if b >= n])
        else:
            for t in running:  # decode growth
                if pyrng.random() < 0.15 and len(t) < columns:
                    t.append(int(rng.integers(0, page_limit)))
            if pyrng.random() < 0.05:  # speculative rollback / truncation
                t = pyrng.choice(running)
                if len(t) > 1:
                    del t[len(t) - pyrng.randint(1, min(3, len(t) - 1)) :]
            if pyrng.random() < 0.03:  # preemption: a table remapped in place
                t = pyrng.choice(running)
                t[0] = int(rng.integers(0, page_limit))
            for _ in range(np.random.default_rng(step).poisson(0.6)):  # finishes
                if len(running) > 1:
                    running.pop(pyrng.randrange(len(running)))
            while len(running) < capacity and pyrng.random() < 0.5:  # admissions
                n = (
                    columns
                    if pyrng.random() < 0.02
                    else int(rng.integers(1, columns // 4))
                )
                new = new_block_table(_ids(rng, n, page_limit))
                running.insert(pyrng.randrange(len(running) + 1), new)
            if pyrng.random() < 0.05:  # a reorder
                pyrng.shuffle(running)
            rows = (
                running
                if pyrng.random() > 0.05
                else [np.array(t, np.int32) for t in running]
            )
            count = next((b for b in buckets if b >= len(rows)), len(rows))
        state.prepare(rows, pad_to=count, page_limit=page_limit)
        published = state.published
        dirty = state._dirty(count)[0]
        state.stage(count, group)
        plan = table._publication.row_plan
        if plan is not None:
            sent += len(plan[1]) * plan[2]
        elif group.counts[0]:
            sent += count * columns * 4
        for b in others:
            group.set_count(b, 5)
        group.publish()
        owner.finish()
        if dirty:  # what the whole-prefix copy would have done
            reference[:count] = torch.from_numpy(table.np[:count]).cuda()
            full += count * columns * 4
        owner.completion.synchronize()
        assert torch.equal(table.gpu, reference), (seed, step)
        assert published is None or state.published is not None
    return sent, full


@pytest.mark.parametrize("seed", range(6))
def test_churn_is_byte_identical_to_full_copies(seed):
    sent, full = _simulate(seed, 400, columns=512, capacity=128)
    assert sent < full


def test_direct_transport_ignores_plans():
    _simulate(11, 150, columns=256, capacity=64, packed=False)


def test_destination_replacement_and_failed_publish_resend_everything():
    table, others = _buffers(8, 64)
    owner, group = _owner(table, others)
    state = block_table_state(table)
    rows = [new_block_table([1, 2, 3]), new_block_table([4])]
    owner.begin()
    state.prepare(rows, pad_to=4)
    state.stage(4, group)
    group.publish()
    owner.finish()
    rows[0].append(9)
    owner.begin()
    state.prepare(rows, pad_to=4)
    state.stage(4, group)
    plan = table._publication.row_plan
    assert plan is not None and plan[1].tolist() == [0] and plan[2] == 16
    group.publish()
    owner.finish()
    owner.completion.synchronize()
    assert table.gpu[:2, :5].tolist() == [[1, 2, 3, 9, 0], [4, 0, 0, 0, 0]]
    # An unknown destination: every row again, at full width.
    state._track(torch.empty(1))
    rows[1].append(5)
    owner.begin()
    state.prepare(rows, pad_to=4)
    state.stage(4, group)
    assert table._publication.row_plan is None
    group.publish()
    owner.finish()
    owner.completion.synchronize()
    assert torch.equal(table.gpu[:4].cpu(), torch.from_numpy(table.np[:4]))
