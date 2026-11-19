# SPDX-License-Identifier: MIT
"""A forward slot's CPU page-table snapshot and its published GPU revision."""

import numpy as np

from atom.model_engine.sequence import BlockTable
from atom.utils.h2d import RowPlan

# A changed table whose published prefix (count x columns x itemsize) is at most
# this many bytes goes up whole: below it the per-row bookkeeping of a row plan
# costs more host time than the copy it saves.
WHOLE_PREFIX_BYTES = 512 << 10


def _int32_row(row):
    values = np.asarray(row)
    if values.ndim != 1:
        raise ValueError("Block-table rows must be one-dimensional")
    if values.dtype != np.int32:
        if values.size and (
            values.dtype.kind not in "iu"
            or values.min() < 0
            or values.max() > np.iinfo(np.int32).max
        ):
            raise ValueError("PAGE ids must be nonnegative int32 integers")
        values = values.astype(np.int32)
    return np.ascontiguousarray(values)


class BlockTableState:
    """Keep row versions and lengths beside the pinned snapshot, not its payload.

    All writers, including capture and padding, must use this object. Versions
    and lengths identify rows without payload snapshots or per-row key tuples.
    """

    def __init__(self, buffer):
        self.buffer = buffer
        self.cpu = buffer.np
        if (
            self.cpu.dtype != np.int32
            or self.cpu.ndim != 2
            or not self.cpu.flags.c_contiguous
        ):
            raise TypeError(
                "Block tables require a contiguous two-dimensional int32 buffer"
            )
        self.bytes = self.cpu.view(np.uint8)
        self.flat = memoryview(self.cpu).cast("B").cast("i") if self.cpu.size else ()
        self.versions = []
        self.lengths = []
        self.zero_to = 0
        self.revision = 0
        self.published = None
        self.page_limit = None
        self.checked = None
        # Per row: the write that last changed it on the host, and what the
        # destination is known to hold -- the write it last received (-1:
        # unknown) and a bound past which it holds zeros. Rows a publish
        # covers and the destination already holds go up no more, and the
        # rest only as wide as their pages and old destination bound need
        # (`_row_plan`).
        rows = self.cpu.shape[0]
        self.row_writes = np.zeros(rows, dtype=np.int64)
        self.writes = 0
        self.gpu_writes = np.full(rows, -1, dtype=np.int64)
        self.gpu_bound = np.full(rows, self.cpu.shape[1], dtype=np.int64)
        self.gpu_tracked = None
        # Counts stages and direct publishes: only the latest one's publish
        # callback records rows as received (`stage`).
        self.stages = 0

    def _acquire(self):
        self.buffer.acquire_write()

    def _begin_write(self):
        self._acquire()
        self.revision += 1
        self.published = None
        # A failed host write must not leave reusable CPU keys or padding.
        self.versions = []
        self.lengths = []
        self.zero_to = 0

    def _normalize(self, rows, versions, lengths):
        rows = list(rows)
        for i, (row, version, length) in enumerate(zip(rows, versions, lengths)):
            if version is not None:
                continue
            rows[i] = values = _int32_row(row)
            unchanged = (
                i < len(self.lengths)
                and length == self.lengths[i]
                and np.array_equal(values, self.cpu[i, :length])
            )
            versions[i] = self.versions[i] if unchanged else object()
        return rows

    def _validate_range(self, rows, versions, lengths, page_limit):
        # Validation belongs to an append lineage, regardless of its row index.
        # Reordering requests must not re-read their already checked pages.
        checked = self.checked if page_limit == self.page_limit else {}
        limit = min(page_limit, 1 << 31)
        for row, version, length in zip(rows, versions, lengths):
            start = checked.get(version, 0)
            if length <= start:
                continue
            if length - start == 1:
                invalid = not 0 <= row[start] < page_limit
            else:
                tail = _int32_row(row)[start:]
                # Signed negatives sort after every valid PAGE id as uint32,
                # so one reduction checks both bounds without a payload copy.
                invalid = tail.view(np.uint32).max() >= limit
            if invalid:
                raise ValueError("Request PAGE table is incomplete or out of range")
        return dict(zip(versions, lengths))

    def prepare(self, rows, *, pad_to=None, page_limit=None, _keys=None):
        n = len(rows)
        capacity, columns = self.cpu.shape
        end = n if pad_to is None else pad_to
        if not 0 <= n <= end <= capacity:
            raise ValueError("Block-table rows exceed the declared capacity")
        if _keys is None:
            versions = [
                row.version if isinstance(row, BlockTable) else None for row in rows
            ]
            lengths = list(map(len, rows))
        else:
            versions, lengths = _keys
        if (
            versions == self.versions
            and lengths == self.lengths
            and end <= self.zero_to
            and page_limit == self.page_limit
        ):
            return self
        if max(lengths, default=0) > columns:
            raise ValueError("Block-table row exceeds the destination columns")
        if None in versions:
            rows = self._normalize(rows, versions, lengths)
        checked = (
            self._validate_range(rows, versions, lengths, page_limit)
            if page_limit is not None
            else None
        )

        old_versions, old_lengths = self.versions, self.lengths
        old_n = len(old_lengths)
        if n > old_n:
            old_versions = old_versions + [None] * (n - old_n)
            old_lengths = old_lengths + [columns] * (n - old_n)
        changes = [
            (i, length, old if version == previous and length >= old else 0, old)
            for i, (version, previous, length, old) in enumerate(
                zip(versions, old_versions, lengths, old_lengths)
            )
            if version != previous or length != old
        ]
        clear_tail = end > n and (n != old_n or end > self.zero_to)
        zero_to = max(end, self.zero_to) if n == old_n else end
        if changes or clear_tail:
            self._begin_write()  # all validation precedes the first write
            # Mark the rows before writing them, so a failed write stays dirty.
            self.writes += 1
            self.row_writes[[i for i, _, _, _ in changes]] = self.writes
            if clear_tail:
                self.row_writes[n:end] = self.writes
            # A fully replaced batch can clear exposed tails in one pass.
            # Partial updates retain every unaffected row and copied prefix.
            bulk_clear = (
                len(changes) == n
                and lengths != old_lengths
                and min(lengths, default=columns) < columns
                and not any(start for _, _, start, _ in changes)
            )
            if bulk_clear:
                self.bytes[:n] = 0
            self._copy_changes(rows, changes, columns, bulk_clear)
            if clear_tail:
                self.bytes[n:end] = 0
        self.versions, self.lengths = versions, lengths
        self.zero_to = zero_to
        self.page_limit = page_limit
        self.checked = checked
        if (changes or clear_tail) and self.gpu_tracked is not None:
            # A row whose upload is not confirmed (`_sending`) may still take
            # these host bytes: a staged publish reads the host when it runs,
            # and a failed one may have written part of them. Its destination
            # bound covers them.
            unknown = self.gpu_writes[:end] < 0
            if unknown.any():
                bounds = self._row_bounds(end)
                np.maximum(
                    self.gpu_bound[:end],
                    np.where(unknown, bounds, 0),
                    out=self.gpu_bound[:end],
                )
        return self

    def _copy_changes(self, rows, changes, columns, bulk_clear):
        flat = self.flat
        for i, length, start, old in changes:
            if not bulk_clear and length < old:
                self.bytes[i, length * 4 : old * 4] = 0
            base = i * columns
            if length - start == 1:
                flat[base + start] = rows[i][start]
            elif length > start:
                flat[base + start : base + length] = (
                    rows[i][start:] if start else rows[i]
                )

    def slice_to(self, buffer, start, count, *, pad_to):
        """Prepare a TBO destination using the source snapshot's row revisions."""
        stop = start + count
        if stop > len(self.lengths):
            # Independent callers can supply an already packed CPU table.
            return block_table_state(buffer).prepare(
                self.cpu[start:stop], pad_to=pad_to
            )
        rows = [self.cpu[i, : self.lengths[i]] for i in range(start, stop)]
        keys = self.versions[start:stop], self.lengths[start:stop]
        return block_table_state(buffer).prepare(rows, pad_to=pad_to, _keys=keys)

    def _dirty(self, count):
        key = (self.revision, count)
        previous = self.published
        dirty = (
            previous is None or previous[0] is not self.buffer.gpu or previous[1] != key
        )
        return dirty, key

    def _row_bounds(self, count):
        """Per row below ``count``: a bound past which the host row is zero.
        Prepared rows hold their pages, the padding rows zeros; any other
        row (never prepared) could hold anything."""
        bounds = np.full(count, self.cpu.shape[1], dtype=np.int64)
        n = min(len(self.lengths), count)
        bounds[:n] = self.lengths[:n]
        bounds[n : min(self.zero_to, count)] = 0
        return bounds

    def _track(self, gpu):
        if self.gpu_tracked is not gpu:
            # A new destination holds nothing known.
            self.gpu_writes.fill(-1)
            self.gpu_bound.fill(self.cpu.shape[1])
            self.gpu_tracked = gpu

    def _row_plan(self, count, bounds):
        """The rows below ``count`` the destination does not hold yet and the
        leading bytes of each that differ from it, or None for all of them.

        A row the destination holds is identical in every byte. Sending a
        changed row's first ``width`` columns makes it so too: past ``width``
        both sides are zero -- the host row past its pages, the destination
        past its bound. Rows above ``count`` stay as they are, as with the
        whole-prefix copy. ``bounds`` is `_row_bounds(count)`."""
        self._track(self.buffer.gpu)
        rows = np.flatnonzero(self.gpu_writes[:count] != self.row_writes[:count])
        if not len(rows):
            return rows, 0
        columns = self.cpu.shape[1]
        width = max(
            int(bounds[rows].max()),
            int(self.gpu_bound[rows].max()),
        )
        if width == 0:
            return rows[:0], 0  # zeros on both sides: nothing differs
        # Whole 16-byte words, as the packed word copy moves them.
        width = min(-(-width // 4) * 4, columns)
        if len(rows) == count and width == columns:
            return None, columns
        return rows.astype(np.int32), width

    def _sending(self, gpu, count, bounds):
        """Before an upload of rows below ``count``: until it has succeeded
        (`_published_rows`), its rows are unknown and may hold the host's
        bytes, so their destination bound widens to the host's. After a failed
        or partial upload every row below ``count`` is resent, as wide as the
        larger of the host's bound and that widened bound; only a destination
        that is not tracked at all gets every row at full width (`_track`).
        ``bounds`` is `_row_bounds(count)`."""
        self._track(gpu)
        self.gpu_writes[:count] = -1
        np.maximum(self.gpu_bound[:count], bounds, out=self.gpu_bound[:count])

    def _published_rows(self, gpu, count, bounds):
        """The destination took every row below ``count`` (planned or whole);
        ``bounds`` is `_row_bounds(count)` of the table it took."""
        self._track(gpu)
        self.gpu_writes[:count] = self.row_writes[:count]
        self.gpu_bound[:count] = bounds

    def stage(self, count, group):
        """Count a changed table into ``group``, whose owner publishes it.

        Omitting a table from a group does not omit its GPU view from metadata.
        The revision is recorded once that publish has succeeded. A packed
        transport sends only the rows the destination does not hold, each
        only as wide as needed (`_row_plan`); the result is the same bytes.

        A small table (a prefix of at most `WHOLE_PREFIX_BYTES`) goes up whole
        and stops the row tracking (`gpu_tracked` None), so the next planned
        stage starts from "unknown".

        A stage's group must publish before any other stage or publish of this
        table.
        """
        dirty, key = self._dirty(count)
        plan = None
        self.stages += 1
        stage = self.stages
        if (
            dirty
            and count * self.cpu.shape[1] * self.cpu.itemsize <= WHOLE_PREFIX_BYTES
        ):
            gpu = self.buffer.gpu
            self.gpu_tracked = None
            group.after_publish(lambda: setattr(self, "published", (gpu, key)))
            group.set_count(self.buffer, count)
            return self.buffer.gpu[:count]
        if dirty:
            gpu = self.buffer.gpu
            bounds = self._row_bounds(count)
            rows, width = self._row_plan(count, bounds)
            row_bytes = width * self.cpu.itemsize
            if rows is not None and not len(rows):
                dirty = False  # the destination already holds every row
            elif rows is not None and row_bytes % 16 == 0:
                # A plan only at whole 16-byte words: a full-width row of a
                # table whose column count is not a multiple of 4 is not one,
                # and such a table goes up whole.
                plan = RowPlan(rows, row_bytes)
            if dirty:
                self._sending(gpu, count, bounds)

            def published():
                if self.revision != key[0] or self.stages != stage:
                    # The host table was prepared again, or staged or published
                    # again, after this stage: the rows stay unknown
                    # (`_sending`) and are sent again. `prepare` already
                    # widened their destination bound to the host bytes.
                    return
                self._published_rows(gpu, count, bounds)
                self.published = (gpu, key)

            group.after_publish(published)
        group.set_count(self.buffer, count if dirty else None, row_plan=plan)
        return self.buffer.gpu[:count]

    def publish(self, count):
        """Upload a changed table on its own."""
        dirty, key = self._dirty(count)
        self.stages += 1  # supersedes a pending stage's record
        if dirty:
            gpu = self.buffer.gpu
            bounds = self._row_bounds(count)
            self._sending(gpu, count, bounds)
            self.buffer.copy_to_gpu(count)
            self._published_rows(gpu, count, bounds)
            self.published = (gpu, key)
        return self.buffer.gpu[:count]


def block_table_state(buffer):
    """Get the state belonging to this physical forward buffer/PP slot."""
    state = getattr(buffer, "_block_table", None)
    if state is None or state.cpu is not buffer.np:
        state = buffer._block_table = BlockTableState(buffer)
    return state
