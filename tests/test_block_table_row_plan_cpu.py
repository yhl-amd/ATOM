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
from atom.utils.h2d import RowPlan, gather_rows
from tests.block_table_churn import decode_churn, ids


@pytest.fixture(autouse=True)
def _plan_every_table(monkeypatch):
    """These tables are small and often change most rows; plan them anyway
    (`WHOLE_PREFIX_BYTES`, `PLAN_MAX_CHANGED_FRACTION`)."""
    from atom.utils import block_tables

    monkeypatch.setattr(block_tables, "WHOLE_PREFIX_BYTES", 0)
    monkeypatch.setattr(block_tables, "PLAN_MAX_CHANGED_FRACTION", 1.0)


class _Group:
    """Records what `stage` counts; `publish` emulates the packed transport."""

    transport = "packed"

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


@pytest.mark.parametrize("real_rules", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_churn_matches_the_host_after_every_successful_publish(
    monkeypatch, seed, real_rules
):
    """Appends, finishes, admissions, preemption, rollback, reorders, prefill
    prepares that are never published, prepares between stage and publish, and
    failed or half-applied publishes."""
    columns, capacity = 96, 48
    if real_rules:  # the counts cross the whole-prefix size; changed-row rule on
        from atom.utils import block_tables

        monkeypatch.setattr(block_tables, "WHOLE_PREFIX_BYTES", 20 * columns * 4)
        monkeypatch.setattr(block_tables, "PLAN_MAX_CHANGED_FRACTION", 0.5)
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    table, device, group, state = _setup(capacity, columns)
    running = [
        new_block_table(ids(rng, pyrng.randint(1, 20), 1 << 20)) for _ in range(20)
    ]
    buckets = [8, 16, 24, 32, 40, 48]
    full = 0
    for step in range(600):
        kind = pyrng.random()
        if kind < 0.05:  # a prefill prepares the same buffer and never publishes
            batch = [new_block_table(ids(rng, pyrng.randint(1, columns), 1 << 20))]
            state.prepare(batch, pad_to=1)
            continue
        decode_churn(
            running,
            rng,
            pyrng,
            step,
            columns=columns,
            max_running=40,
            page_limit=1 << 20,
        )
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


@pytest.mark.parametrize("outcome", ["ok", "fail", "half"])
def test_a_prepare_between_stage_and_publish_is_not_recorded_as_received(outcome):
    """The upload reads the host at publish time. A row prepared after the plan
    was built and not in it must not be marked as held, and the next publish
    sends it. A planned row that grows in between lands on the device past
    its stage-time bound, whether the publish succeeds, fails or is cut short;
    when that row later holds a shorter request at a smaller count, the plan
    must still clear those pages."""
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

    # A planned row grows during the race, then the count narrows.
    rows = [new_block_table([1, 2]), new_block_table(list(range(100, 120)))]
    state.prepare(rows, pad_to=2).stage(2, group)
    assert group.publish()
    rows[0].append(3)
    rows[1].append(120)
    state.prepare(rows, pad_to=2).stage(2, group)
    plan = group.plan
    assert plan.rows.tolist() == [0, 1] and plan.row_bytes == 24 * 4
    rows[0].extend(range(4, 12))  # row 0 grows to 11 pages before the publish
    state.prepare(rows, pad_to=2)
    if outcome == "half":  # the cut-short upload landed row 0 only
        device[0, :24] = table.np[0, :24]
        group.publish("fail")
    else:
        assert group.publish(outcome) == (outcome == "ok")
    if outcome != "fail":
        assert device[0, :11].tolist() == list(range(1, 12))
    state.prepare([new_block_table([500])], pad_to=1).stage(1, group)
    assert group.publish()
    np.testing.assert_array_equal(device[:1], table.np[:1])


@pytest.mark.parametrize("second", ["narrower", "nothing", "direct"])
def test_a_superseded_stage_records_nothing(second):
    """Staging the same table again (a narrower count, or one with nothing to
    send) or publishing it directly before the first stage's group publishes:
    the first stage's callback must not record rows the device never got."""
    table, device, group, state = _setup(8, 64)
    rows = [new_block_table([1, 2]) for _ in range(4)]
    state.prepare(rows, pad_to=4).stage(4, group)
    assert group.publish()
    for r in rows:
        r.append(9)
    state.prepare(rows, pad_to=4).stage(4, group)
    if second == "direct":
        state.publish(2)  # whole prefix [:2] straight to `table.gpu`
        device[:2] = table.gpu[:2].numpy()
    elif second == "nothing":
        state.stage(4, group)  # nothing new to send on top of the first
    else:
        state.stage(2, group)
    assert group.publish()
    state.prepare(rows, pad_to=4).stage(4, group)
    assert group.publish()
    np.testing.assert_array_equal(device[:4], table.np[:4])


def test_gather_rows_reads_only_the_planned_rows():
    """The planned rows of a strided view are gathered without a temporary
    copy of the whole table (what `np.take(..., out=)` makes)."""
    import tracemalloc

    rng = np.random.default_rng(0)
    table = rng.integers(0, 255, size=(4096, 1024), dtype=np.uint8)
    source = table[:, :512]  # a plan narrower than the row: strided
    rows = np.sort(rng.choice(4096, size=12, replace=False)).astype(np.int32)
    out = np.empty((len(rows), 512), dtype=np.uint8)
    tracemalloc.start()
    gather_rows(out, source, rows)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    np.testing.assert_array_equal(out, table[rows, :512])
    assert peak < source.nbytes // 16
    run = np.arange(100, 140, dtype=np.int32)
    out = np.empty((len(run), 512), dtype=np.uint8)
    gather_rows(out, source, run)
    np.testing.assert_array_equal(out, table[100:140, :512])


def test_small_tables_go_up_whole_and_restart_tracking(monkeypatch):
    """At most `WHOLE_PREFIX_BYTES` the prefix goes up whole and tracking
    stops; the first planned stage after it starts from unknown (every row,
    full width), later ones plan only changed rows, and the device always
    matches the host."""
    from atom.utils import block_tables

    columns = 64
    monkeypatch.setattr(block_tables, "WHOLE_PREFIX_BYTES", 8 * columns * 4)
    table, device, group, state = _setup(32, columns)
    rows = [new_block_table([i + 1, i + 2]) for i in range(8)]
    for _ in range(2):  # small: the whole prefix, no plan, untracked
        state.prepare(rows, pad_to=8).stage(8, group)
        assert group.count == 8 and group.plan is None
        assert state.gpu_tracked is None
        assert group.publish()
        np.testing.assert_array_equal(device[:8], table.np[:8])
        rows[0].append(50)
    rows += [new_block_table([100 + i]) for i in range(16)]
    state.prepare(rows, pad_to=24).stage(24, group)  # large: tracking restarts
    assert group.count == 24 and group.plan is None  # every row, full width
    assert state.gpu_tracked is table.gpu
    assert group.publish()
    np.testing.assert_array_equal(device[:24], table.np[:24])
    rows[3].append(77)
    state.prepare(rows, pad_to=24).stage(24, group)
    assert group.plan is not None and group.plan.rows.tolist() == [3]
    assert group.publish()
    np.testing.assert_array_equal(device[:24], table.np[:24])
    rows = rows[:8]  # back to small
    rows[5].append(88)
    state.prepare(rows, pad_to=8).stage(8, group)
    assert group.plan is None and state.gpu_tracked is None
    assert group.publish()
    np.testing.assert_array_equal(device[:8], table.np[:8])


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
    rows[1].append(7)  # not whole 16-byte rows: never a plan
    state.prepare(rows, pad_to=4).stage(4, group)
    assert group.count == 4 and group.plan is None
    assert state.gpu_tracked is None
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


def test_a_step_that_changes_most_rows_goes_up_whole(monkeypatch):
    """With the real changed-row rule: a step that rewrote most rows goes up
    whole and untracked; the next step with few changes restarts tracking
    (whole once, from unknown), then plans only the changed rows."""
    from atom.utils import block_tables

    monkeypatch.setattr(block_tables, "PLAN_MAX_CHANGED_FRACTION", 0.5)
    table, device, group, state = _setup(8, 64)
    rows = [new_block_table([i + 1]) for i in range(8)]
    state.prepare(rows, pad_to=8).stage(8, group)
    assert group.publish()
    for r in rows[:6]:
        r.append(70)
    state.prepare(rows, pad_to=8).stage(8, group)
    assert group.count == 8 and group.plan is None and state.gpu_tracked is None
    assert group.publish()
    np.testing.assert_array_equal(device[:8], table.np[:8])
    rows[2].append(71)
    state.prepare(rows, pad_to=8).stage(8, group)
    assert group.plan is None and state.gpu_tracked is table.gpu  # from unknown
    assert group.publish()
    rows[4].append(72)
    state.prepare(rows, pad_to=8).stage(8, group)
    assert group.plan is not None and group.plan.rows.tolist() == [4]
    assert group.publish()
    np.testing.assert_array_equal(device[:8], table.np[:8])
