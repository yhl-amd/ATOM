# SPDX-License-Identifier: MIT
"""Pack metadata into one pinned arena, upload once, scatter to stable addresses."""

import numpy as np
import torch
import triton
import triton.language as tl

from atom.utils.h2d import gather_rows


@triton.jit
def scatter_packed_words(arena, destinations, wide, pitches, WORDS: tl.constexpr):
    """Scatter each packed member to its destination, WORDS * 8 bytes a
    program: as int64 words where the member starts 16-byte aligned in the
    arena (packing pads every member to that) and its destination is too
    (`wide`), byte by byte otherwise.

    A member's header is (start, byte count, row bytes, row map offset). Row
    bytes 0: a contiguous prefix. Otherwise the payload holds rows of that many
    bytes (a multiple of 16), each stored at the start of the destination row
    the int32 row map names, ``pitches`` bytes apart (`PackedCopy.submit`)."""
    member = tl.program_id(0)
    header = arena.to(tl.pointer_type(tl.int64))
    start = tl.load(header + 4 * member)
    length = tl.load(header + 4 * member + 1)
    address = tl.load(destinations + member)
    span = WORDS * 8
    base = tl.program_id(1).to(tl.int64) * span
    if base < length:
        if tl.load(wide + member) != 0:
            words = length // 8
            index = base // 8 + tl.arange(0, WORDS)
            mask = index < words
            source = (arena + tl.multiple_of(start, 16)).to(tl.pointer_type(tl.int64))
            values = tl.load(source + index, mask)
            row_bytes = tl.load(header + 4 * member + 2)
            if row_bytes == 0:
                target = tl.multiple_of(address, 16).to(tl.pointer_type(tl.int64))
                tl.store(target + index, values, mask)
                # The last program also moves the 0-7 tail bytes.
                if base + span >= length:
                    tail = words * 8 + tl.arange(0, 8)
                    tl.store(
                        address.to(tl.pointer_type(tl.uint8)) + tail,
                        tl.load(arena + start + tail, tail < length),
                        tail < length,
                    )
            else:
                row_words = row_bytes // 8
                row_map = (arena + tl.load(header + 4 * member + 3)).to(
                    tl.pointer_type(tl.int32)
                )
                row = index // row_words
                column = index - row * row_words
                target_row = tl.load(row_map + row, mask, other=0).to(tl.int64)
                pitch_words = tl.load(pitches + member) // 8
                target = tl.multiple_of(address, 16).to(tl.pointer_type(tl.int64))
                tl.store(target + target_row * pitch_words + column, values, mask)
        else:
            offset = base + tl.arange(0, WORDS * 8)
            tl.store(
                address.to(tl.pointer_type(tl.uint8)) + offset,
                tl.load(arena + start + offset, offset < length),
                offset < length,
            )


def _pad16(nbytes):
    return -(-nbytes // 16) * 16


class PackedCopy:
    """Persistent byte storage; the publication owner fences reuse of the arena."""

    @classmethod
    def create(cls, members, device):
        if device.type != "cuda" or len(members) < 2:
            return None, "packing requires multiple GPU destinations"
        if any(
            not b.source.is_contiguous() or not b.destination.is_contiguous()
            for b in members
        ):
            return None, "strided regions use checked direct"
        result = cls()
        result.members = members
        # int64 (offset, byte count, row bytes, row map offset) per member.
        result.header_bytes = 32 * len(members)
        # Destinations the word copy may store to as int64 (16-byte aligned).
        wide = [int(b.destination.data_ptr() % 16 == 0) for b in members]
        # Members that may take a row plan (`RowPlan`):
        # two-dimensional rows of whole 16-byte words at a word-aligned address.
        result.row_members = tuple(
            b.unit == "rows"
            and b.source.ndim == 2
            and b.bytes_per_count % 16 == 0
            and bool(w)
            for b, w in zip(members, wide)
        )
        # Every member starts 16-byte aligned (`submit` pads), for the word
        # copy; a row member also reserves a row map of one int32 per row.
        capacity = result.header_bytes + sum(
            _pad16(b.capacity * b.bytes_per_count)
            + (_pad16(4 * b.row_capacity) if rows else 0)
            for b, rows in zip(members, result.row_members)
        )
        result.host = torch.empty(
            capacity, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        result.device = torch.empty(capacity, dtype=torch.uint8, device=device)
        result.header = result.host[: result.header_bytes].view(torch.int64).numpy()
        result.payload = memoryview(result.host.numpy())
        result.arena = result.host.numpy()
        # Byte views avoid typed copies canonicalizing bool or casting BF16.
        result.sources = tuple(
            memoryview(b.source.reshape(-1).view(torch.uint8).numpy()) for b in members
        )
        result.rows = tuple(
            (
                b.source.view(torch.uint8)
                .numpy()
                .reshape(b.row_capacity, b.bytes_per_count)
                if rows
                else None
            )
            for b, rows in zip(members, result.row_members)
        )
        result.destinations = torch.tensor(
            [b.destination.data_ptr() for b in members],
            dtype=torch.int64,
            device=device,
        )
        result.wide = torch.tensor(wide, dtype=torch.int32, device=device)
        result.pitches = torch.tensor(
            [
                b.bytes_per_count if rows else 0
                for b, rows in zip(members, result.row_members)
            ],
            dtype=torch.int64,
            device=device,
        )
        result.prefixes = {}
        result.launch = scatter_packed_words
        result.kernel = None
        return result, None

    def _plan(self, i, count, plan):
        """Member ``i``'s `RowPlan` for ``count`` rows, if it can take one."""
        if (
            plan is None
            or not self.row_members[i]
            or plan.row_bytes > self.members[i].bytes_per_count
            or plan.rows[-1] >= count
        ):
            return None
        return plan.rows, plan.row_bytes

    def submit(self, counts, plans=None):
        """Enqueue copies and return whether the packed arena is borrowed.
        ``plans`` holds each member's optional `RowPlan` for its count."""
        active = [i for i, count in enumerate(counts) if count]
        if len(active) == 1:
            # A single live member already takes one DMA; no packing/scatter.
            i = active[0]
            self.members[i]._copy(counts[i])
            return False
        offset = self.header_bytes
        largest = 0
        for i, (binding, count) in enumerate(zip(self.members, counts)):
            plan = self._plan(i, count, plans[i]) if count and plans else None
            self.header[4 * i] = offset
            self.header[4 * i + 2] = 0
            self.header[4 * i + 3] = 0
            if plan is not None:
                # Only the planned rows' leading bytes; the rest of every
                # destination row already holds the source's bytes.
                rows, row_bytes = plan
                length = len(rows) * row_bytes
                block = self.arena[offset : offset + length].reshape(
                    len(rows), row_bytes
                )
                gather_rows(block, self.rows[i][:, :row_bytes], rows)
                row_map = offset + _pad16(length)
                self.arena[row_map : row_map + 4 * len(rows)].view(np.int32)[:] = rows
                self.header[4 * i + 1] = length
                self.header[4 * i + 2] = row_bytes
                self.header[4 * i + 3] = row_map
                offset = row_map + _pad16(4 * len(rows))
                largest = max(largest, length)
                continue
            length = 0 if count is None else count * binding.bytes_per_count
            self.header[4 * i + 1] = length
            if length:
                self.payload[offset : offset + length] = self.sources[i][:length]
                offset += _pad16(length)
                largest = max(largest, length)
        prefix = self.prefixes.get(offset)
        if prefix is None:
            prefix = self.host[:offset], self.device[:offset]
            if len(self.prefixes) == 4:
                del self.prefixes[next(iter(self.prefixes))]
            self.prefixes[offset] = prefix
        # Header and payload share this single asynchronous H2D.
        prefix[1].copy_(prefix[0], non_blocking=True)
        # 1024 words (8 KiB) a program.
        grid = (len(self.members), (largest + 8191) // 8192, 1)
        if self.kernel is None:
            self.kernel = self.launch[grid](
                self.device,
                self.destinations,
                self.wide,
                self.pitches,
                WORDS=1024,
                num_warps=8,
            )
        else:
            # Both argument storages and their layouts are fixed for this
            # binding. Reuse its compiled kernel; counts and grid stay dynamic.
            self.kernel[grid](self.device, self.destinations, self.wide, self.pitches)
        return True
