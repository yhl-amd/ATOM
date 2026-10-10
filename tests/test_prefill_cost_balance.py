"""DP prefill cost balancing (ATOM_DP_PREFILL_COST_BALANCE).

Under DP attention every rank prefills in the same lockstep step, so the step
lasts as long as the costliest rank's chunk. With balancing on, the ranks agree
on that cost (`PrefillDelayer.max_across_ranks`) and a cheaper rank grows its
chunk past the base budget until it reaches it.
"""

import pytest
from conftest import MockConfig

from atom.model_engine.scheduler import Scheduler


class _Delayer:
    """Cross-DP delayer stub that releases prefill every tick and answers the
    cost exchange with a fixed target (None = echo this rank's own cost)."""

    is_local = False
    max_queue_ms = None
    any_rank_prefillable = True

    def __init__(self, target=None):
        self.target = target
        self.reported = []

    def protects_decode(self, running_decode_batch: int) -> bool:
        return False

    def should_allow_prefill(self, prefillable, pending_tokens, **kwargs):
        return True

    def max_across_ranks(self, value: float) -> float:
        self.reported.append(value)
        return value if self.target is None else self.target


def _sched(monkeypatch, *, balance=True, base=16, weight=0, max_tokens=64):
    monkeypatch.setenv("ATOM_DP_PREFILL_COST_BALANCE", "1" if balance else "0")
    monkeypatch.setenv("ATOM_DP_PREFILL_BALANCE_BASE_TOKENS", str(base))
    monkeypatch.setenv("ATOM_DP_PREFILL_BALANCE_TOKEN_WEIGHT", str(weight))
    return Scheduler(
        MockConfig(
            max_num_seqs=8,
            num_kvcache_blocks=64,
            kv_cache_block_size=4,
            max_model_len=256,
            max_num_batched_tokens=max_tokens,
        )
    )


def _prefill_tokens(sched):
    batch, _ = sched.schedule()
    return int(batch.total_tokens_num)


def test_off_keeps_the_whole_budget_and_skips_the_exchange(monkeypatch, seq_factory):
    sched = _sched(monkeypatch, balance=False)
    delayer = _Delayer(target=1e12)
    sched.set_prefill_delayer(delayer)
    sched.add(seq_factory(list(range(100, 164))))
    assert sched.prefill_base_tokens == 64
    assert _prefill_tokens(sched) == 64
    assert delayer.reported == []


@pytest.mark.parametrize("start", [0, 1000, 250_000])
def test_room_is_the_largest_chunk_within_the_cost_left(monkeypatch, start):
    sched = _sched(monkeypatch, weight=4096, max_tokens=1 << 20)
    sched._tick_token_cap = 1 << 20
    sched._tick_cost_limit = 5e8
    sched._tick_cost = 1e8
    room = sched._prefill_room(0, start)
    left = sched._tick_cost_limit - sched._tick_cost
    assert sched._chunk_cost(start, room) <= left
    assert sched._chunk_cost(start, room + 1) > left


def test_cheaper_rank_grows_its_chunk_to_the_target_cost(monkeypatch, seq_factory):
    sched = _sched(monkeypatch)
    # With weight 0, a chunk of k tokens from position 0 costs k * k / 2: the
    # base 16-token chunk costs 128, and a target of 512 admits 32 tokens.
    delayer = _Delayer(target=512.0)
    sched.set_prefill_delayer(delayer)
    sched.add(seq_factory(list(range(100, 164))))
    assert _prefill_tokens(sched) == 32
    assert delayer.reported == [128.0]


def test_costliest_rank_keeps_its_base_chunk(monkeypatch, seq_factory):
    sched = _sched(monkeypatch)
    sched.set_prefill_delayer(_Delayer())  # target = its own cost
    sched.add(seq_factory(list(range(100, 164))))
    assert _prefill_tokens(sched) == 16


def test_idle_rank_still_joins_the_exchange(monkeypatch):
    sched = _sched(monkeypatch)
    delayer = _Delayer(target=512.0)
    sched.set_prefill_delayer(delayer)
    assert sched.schedule() is None
    assert delayer.reported == [0.0]
