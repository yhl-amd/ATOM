"""DP prefill cost balancing (ATOM_DP_PREFILL_COST_BALANCE).

Under DP attention every rank prefills in the same lockstep step, so the step
lasts as long as the costliest rank's chunk. With balancing on, the ranks
exchange their estimated chunk costs (`PrefillDelayer.gather_across_ranks`)
and aim at min(costliest, mean x (1 + slack)): a costlier rank shrinks its
chunk, a cheaper one may grow it when the token cap is above the base.

With weight 0 a chunk of k tokens from position 0 costs k * k / 2.
"""

import pytest
from conftest import MockConfig

from atom.model_engine.scheduler import Scheduler


class _Delayer:
    """Cross-DP delayer stub that releases prefill every tick and answers the
    cost exchange with this rank's cost followed by `others`."""

    is_local = False
    max_queue_ms = None
    any_rank_prefillable = True

    def __init__(self, others=()):
        self.others = list(others)
        self.reported = []

    def protects_decode(self, running_decode_batch: int) -> bool:
        return False

    def should_allow_prefill(self, prefillable, pending_tokens, **kwargs):
        return True

    def gather_across_ranks(self, value: float) -> list[float]:
        self.reported.append(value)
        return [value, *self.others]


def _sched(
    monkeypatch,
    *,
    balance=True,
    base=0,
    weight=0,
    slack=0.1,
    min_tokens=8,
    max_tokens=64,
):
    monkeypatch.setenv("ATOM_DP_PREFILL_COST_BALANCE", "1" if balance else "0")
    monkeypatch.setenv("ATOM_DP_PREFILL_BALANCE_BASE_TOKENS", str(base))
    monkeypatch.setenv("ATOM_DP_PREFILL_BALANCE_TOKEN_WEIGHT", str(weight))
    monkeypatch.setenv("ATOM_DP_PREFILL_BALANCE_SLACK", str(slack))
    monkeypatch.setenv("ATOM_DP_PREFILL_BALANCE_MIN_TOKENS", str(min_tokens))
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
    delayer = _Delayer(others=[1.0] * 7)
    sched.set_prefill_delayer(delayer)
    sched.add(seq_factory(list(range(100, 164))))
    assert sched.prefill_base_tokens == 64
    assert _prefill_tokens(sched) == 64
    assert delayer.reported == []


@pytest.mark.parametrize("start", [0, 1000, 250_000])
def test_room_is_the_largest_chunk_within_the_cost_left(monkeypatch, start):
    sched = _sched(monkeypatch, weight=4096, min_tokens=0, max_tokens=1 << 20)
    sched._tick_token_cap = 1 << 20
    sched._tick_cost_limit = 5e8
    sched._tick_cost = 1e8
    room = sched._prefill_room(0, start)
    left = sched._tick_cost_limit - sched._tick_cost
    assert sched._chunk_cost(start, room) <= left
    assert sched._chunk_cost(start, room + 1) > left


def test_costliest_rank_shrinks_to_the_target(monkeypatch, seq_factory):
    sched = _sched(monkeypatch)
    # Own 64-token chunk costs 2048; seven ranks at 512: target =
    # min(2048, 1.1 * 704) = 774.4, i.e. 39 tokens.
    delayer = _Delayer(others=[512.0] * 7)
    sched.set_prefill_delayer(delayer)
    sched.add(seq_factory(list(range(100, 164))))
    assert _prefill_tokens(sched) == 39
    assert delayer.reported == [2048.0]


def test_shrinking_never_goes_below_the_floor(monkeypatch, seq_factory):
    # Seven near-idle ranks: target = 1.1 * (2048 + 3.5) / 8 = 282, i.e. 23
    # tokens, under the 32-token floor.
    sched = _sched(monkeypatch, min_tokens=32)
    sched.set_prefill_delayer(_Delayer(others=[0.5] * 7))
    sched.add(seq_factory(list(range(100, 164))))
    assert _prefill_tokens(sched) == 32


def test_cheaper_rank_keeps_its_chunk_without_room_to_grow(monkeypatch, seq_factory):
    sched = _sched(monkeypatch)
    sched.set_prefill_delayer(_Delayer(others=[1e9] * 7))
    sched.add(seq_factory(list(range(100, 164))))
    assert _prefill_tokens(sched) == 64


def test_cheaper_rank_grows_when_the_cap_is_above_the_base(monkeypatch, seq_factory):
    sched = _sched(monkeypatch, base=16)
    # Own 16-token chunk costs 128; seven ranks at 512: target =
    # min(512, 1.1 * 464) = 510.4, i.e. 31 tokens.
    delayer = _Delayer(others=[512.0] * 7)
    sched.set_prefill_delayer(delayer)
    sched.add(seq_factory(list(range(100, 164))))
    assert _prefill_tokens(sched) == 31
    assert delayer.reported == [128.0]


def test_idle_rank_still_joins_the_exchange(monkeypatch):
    sched = _sched(monkeypatch)
    delayer = _Delayer(others=[512.0] * 7)
    sched.set_prefill_delayer(delayer)
    assert sched.schedule() is None
    assert delayer.reported == [0.0]
