# SPDX-License-Identifier: MIT
"""Row tracking of packed block-table publishes, without a GPU.

A fake group stands in for the packed transport: at publish it applies the
counted table to a numpy "device" the way `PackedCopy` would (the planned
rows' leading bytes, or the whole prefix), reading the host table at publish
time, and runs the `after_publish` callbacks only when it succeeds. The device
prefix must equal the host's after every successful publish."""

import random

import numpy as np
import pytest
import torch

from atom.model_engine.sequence import new_block_table
from atom.utils import CpuGpuBuffer
from atom.utils.block_tables import block_table_state
from atom.utils.h2d import RowPlan


class _Group:
    """Records what `stage` counts; `publish` emulates the packed transport."""

    def __init__(self, table, device):
        self.table, self.device = table, device
        self.count = self.plan = None
        self.callbacks = []
        self.sent = 0

    def set_count(self, buffer, count, row_plan=None):
        assert buffer is self.table
        self.count = count
        self.plan = None if count is None else row_plan

    def after_publish(self, callback):
        self.callbacks.append(callback)

    def publish(self, mode="ok", rng=None):
        count, plan = self.count, self.plan
        host = self.table.np
        if count and mode != "fail":
            if plan is None:
                rows = np.arange(count)
                width = host.shape[1]
            else:
                rows, width = plan.rows, plan.row_bytes // host.itemsize
            if mode == "half":  # an interrupted upload: some rows landed
                rows = rows[rng.random(len(rows)) < 0.5]
            self.device[rows, :width] = host[rows, :width]
            self.sent += len(rows) * width
        self.count = self.plan = None
        callbacks, self.callbacks = self.callbacks, []
        if mode != "ok":
            return False
        for callback in callbacks:
            callback()
        return True


def _setup(rows, columns):
    table = CpuGpuBuffer(
        rows, columns, dtype=torch.int32, device="cpu", pin_memory=False
    )
    device = np.full((rows, columns), -7, dtype=np.int32)  # unknown contents
    return table, device, _Group(table, device), block_table_state(table)


def _ids(rng, n):
    return rng.integers(0, 1 << 20, size=n).tolist()


@pytest.mark.parametrize("seed", range(4))
def test_churn_matches_the_host_after_every_successful_publish(seed):
    """Appends, finishes, admissions, preemption, rollback, reorders, prefill
    prepares that are never published, prepares between stage and publish, and
    failed or half-applied publishes."""
    columns, capacity = 96, 48
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    table, device, group, state = _setup(capacity, columns)
    running = [new_block_table(_ids(rng, pyrng.randint(1, 20))) for _ in range(20)]
    buckets = [8, 16, 24, 32, 40, 48]
    full = 0
    for step in range(600):
        kind = pyrng.random()
        if kind < 0.05:  # a prefill prepares the same buffer and never publishes
            batch = [new_block_table(_ids(rng, pyrng.randint(1, columns)))]
            state.prepare(batch, pad_to=1)
            continue
        for t in running:
            if pyrng.random() < 0.2 and len(t) < columns:
                t.append(int(rng.integers(0, 1 << 20)))
        if pyrng.random() < 0.05:  # speculative rollback
            t = pyrng.choice(running)
            if len(t) > 1:
                del t[len(t) - pyrng.randint(1, min(3, len(t) - 1)) :]
        if pyrng.random() < 0.03:  # preemption remaps a page in place
            pyrng.choice(running)[0] = int(rng.integers(0, 1 << 20))
        if pyrng.random() < 0.3 and len(running) > 1:  # a finish
            running.pop(pyrng.randrange(len(running)))
        while len(running) < 40 and pyrng.random() < 0.4:  # admissions
            new = new_block_table(_ids(rng, pyrng.randint(1, columns)))
            running.insert(pyrng.randrange(len(running) + 1), new)
        if pyrng.random() < 0.05:
            pyrng.shuffle(running)
        count = next(b for b in buckets if b >= len(running))
        state.prepare(running, pad_to=count)
        dirty = state._dirty(count)[0]
        state.stage(count, group)
        # Prepared again before the publish: that upload may leave rows it did
        # not plan stale; they must not be recorded as held, so the check
        # below holds again from the next publish on.
        raced = pyrng.random() < 0.05
        if raced:
            t = pyrng.choice(running)
            if len(t) < columns:
                t.append(int(rng.integers(0, 1 << 20)))
            state.prepare(running, pad_to=count)
        mode = pyrng.choices(["ok", "fail", "half"], [0.9, 0.05, 0.05])[0]
        if group.publish(mode, rng) and not raced:
            np.testing.assert_array_equal(
                device[:count], table.np[:count], err_msg=f"{seed}/{step}"
            )
            full += dirty * count * columns
    assert group.sent < full


def test_a_prepare_between_stage_and_publish_is_not_recorded_as_received():
    """The upload reads the host at publish time; a row prepared after the
    plan was built and not in it must not be marked as held, and the next
    publish sends it."""
    table, device, group, state = _setup(8, 64)
    rows = [new_block_table([1, 2, 3]), new_block_table([4, 5])]
    state.prepare(rows, pad_to=4).stage(4, group)
    assert group.publish()
    rows[0].append(9)
    state.prepare(rows, pad_to=4).stage(4, group)
    assert group.plan.rows.tolist() == [0]
    rows[1][0] = 77  # a second prepare before the publish
    state.prepare(rows, pad_to=4)
    assert group.publish()
    assert device[1, 0] == 4  # the plan did not cover row 1
    state.prepare(rows, pad_to=4).stage(4, group)
    assert group.count == 4 and 1 in group.plan.rows.tolist()
    assert group.publish()
    np.testing.assert_array_equal(device[:4], table.np[:4])


def test_a_table_whose_columns_are_not_whole_words_gets_no_plan():
    table, device, group, state = _setup(4, 6)
    rows = [new_block_table([1, 2]), new_block_table([3])]
    state.prepare(rows, pad_to=4).stage(4, group)
    assert group.publish()
    rows[0].extend([4, 5, 6])  # five pages: the width rounds past six columns
    state.prepare(rows, pad_to=4).stage(4, group)
    assert group.count == 4 and group.plan is None
    assert group.publish()
    np.testing.assert_array_equal(device[:4], table.np[:4])
    rows[1].append(7)  # a narrow change still fits whole words
    state.prepare(rows, pad_to=4).stage(4, group)
    assert group.plan is not None and group.plan.row_bytes == 16
    assert group.publish()
    np.testing.assert_array_equal(device[:4], table.np[:4])


@pytest.mark.parametrize(
    ("rows", "row_bytes"),
    [
        ([], 16),
        ([-1, 2], 16),
        ([3, 3], 16),
        ([4, 2], 16),
        ([0], 0),
        ([0], 24),
    ],
)
def test_row_plan_rejects_malformed_plans(rows, row_bytes):
    with pytest.raises(ValueError):
        RowPlan(np.asarray(rows, dtype=np.int32), row_bytes)


def test_row_plan_accepts_increasing_rows():
    plan = RowPlan(np.asarray([0, 2, 5], dtype=np.int32), 32)
    assert plan.rows.tolist() == [0, 2, 5] and plan.row_bytes == 32
