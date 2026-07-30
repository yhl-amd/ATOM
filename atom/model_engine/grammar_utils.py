# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""xgrammar-based grammar-constrained decoding helpers for tool calling.

Rationale: DeepSeek-V4-Pro, when free-decoding tool calls, frequently emits a
malformed shape — an extra ``arguments`` wrapper around a stringified, often
*invalid* JSON body (shell commands with unescaped quotes/newlines). Pure
post-hoc parsing cannot recover invalid JSON, so we constrain generation at the
token level with xgrammar's builtin ``deepseek_v4`` DSML structural tag, which
forces well-formed ``<｜DSML｜invoke name=...><｜DSML｜parameter ...>`` output
with correctly escaped string values (mirrors what vLLM does).

All xgrammar imports are lazy so the engine still runs if xgrammar is absent.
"""

import json
import logging
from typing import Optional

import numpy as _np

logger = logging.getLogger(__name__)

# DeepSeek-V4 DSML tool-call block close marker. Its appearance in the generated
# text means the tool call is complete; we then force EOS because xgrammar's
# structural_tag never terminates on its own after the structure.
_DSML_TOOL_END = "</｜DSML｜tool_calls>"


def _to_openai_tool(t):
    """Normalize a request tool (pydantic model or dict) to an OpenAI dict."""
    if isinstance(t, dict):
        return t
    if hasattr(t, "model_dump"):
        return t.model_dump(exclude_none=True)
    return t


def build_deepseek_v4_spec(tools, tool_choice="auto") -> Optional[str]:
    """Build a ``deepseek_v4`` structural-tag JSON spec from OpenAI tools.

    Returns the serialized spec string, or ``None`` when there are no tools or
    ``tool_choice`` disables calling — in which case generation is unconstrained.
    Never raises: on any failure it logs and returns ``None`` (fall back to
    unconstrained decoding + post-hoc parsing).
    """
    if not tools:
        return None
    tc_in = tool_choice if isinstance(tool_choice, str) else "auto"
    if tc_in == "none":
        return None
    # Force 'required' rather than 'auto': agentic tool use (mini-swe-agent)
    # mandates a tool call every step, and 'required' yields a tight grammar
    # that terminates right after the call. 'auto' appends a free-text tail
    # (AnyText) so, under greedy decoding, the model keeps generating past a
    # correct tool call all the way to max_tokens.
    tc = "required"
    try:
        import xgrammar as xgr

        dumped = [_to_openai_tool(t) for t in tools]
        # exclude_special_tokens=False is critical: the default (True) drops the
        # EOS/special tokens from the grammar's allowed set, so after the tool
        # call the model — wanting to stop — cannot emit the real EOS token and
        # instead spells out the literal string "<｜end▁of▁sentence｜>" with
        # ordinary tokens, which the engine never recognizes as a stop → decoding
        # runs to max_tokens. Allowing special tokens lets it emit the real EOS.
        tag = xgr.get_model_structural_tag(
            "deepseek_v4", tools=dumped, tool_choice=tc, reasoning=True,
            exclude_special_tokens=False,
        )
        return json.dumps(tag.model_dump())
    except Exception as e:  # noqa: BLE001 - never break a request over grammar
        logger.warning("build_deepseek_v4_spec failed, decoding unconstrained: %s", e)
        return None


class GrammarBackend:
    """Engine-level xgrammar compiler bound to the model tokenizer.

    Holds the (shareable, cache-enabled) ``GrammarCompiler``; ``compile`` mints a
    fresh per-request ``GrammarMatcher`` holding mutable FSM state.
    """

    def __init__(self, tokenizer, vocab_size: int):
        import xgrammar as xgr

        self._xgr = xgr
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.eos = getattr(tokenizer, "eos_token_id", None)
        self.tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=vocab_size
        )
        self.compiler = xgr.GrammarCompiler(self.tokenizer_info, cache_enabled=True)

    def compile(self, spec: str):
        """Compile a structural-tag spec into a fresh per-request matcher.

        Returns ``None`` on failure (request then decodes unconstrained).
        """
        try:
            ctx = self.compiler.compile_structural_tag(spec)
            return self._xgr.GrammarMatcher(ctx)
        except Exception as e:  # noqa: BLE001
            logger.warning("grammar compile failed, decoding unconstrained: %s", e)
            return None

    _INSTANCE = None

    @classmethod
    def get(cls, model_path: str):
        """Process-local singleton backend, tokenizer loaded from model_path.

        Lives in the EngineCore process. Returns ``None`` if xgrammar/tokenizer
        cannot be initialized (callers then skip grammar constraint).
        """
        if cls._INSTANCE is None:
            try:
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True
                )
                cls._INSTANCE = cls(tok, len(tok))
            except Exception as e:  # noqa: BLE001
                logger.warning("GrammarBackend init failed: %s", e)
                cls._INSTANCE = False  # sentinel: tried and failed
        return cls._INSTANCE or None


def ensure_and_fill_bitmask(seqs, model_path, matchers):
    """EngineCore-side: compile missing matchers and fill a token bitmask.

    ``seqs`` is the scheduler's ordered ``{seq_id: Sequence}`` for this forward
    (bitmask row i aligns with ``list(seqs.values())[i]``, matching how
    ScheduledBatch builds its per-seq arrays). ``matchers`` is a process-local
    ``{seq_id: GrammarMatcher}`` dict owned by the Scheduler — kept OFF the
    Sequence because the matcher is not picklable and seqs get serialized. For
    every seq carrying a ``structured_outputs`` spec: lazily compile its matcher
    into ``matchers``, then fill that row with the currently allowed tokens.
    Rows without grammar / terminated / mid-prefill stay all-ones (allow all).

    Returns a numpy bitmask ``(num_seqs, ceil(vocab/32))`` int32, or ``None`` if
    no seq is grammar-constrained (fast path: skip entirely).
    """
    seq_items = list(seqs.items())
    if not any(getattr(s, "structured_outputs", None) for _, s in seq_items):
        return None
    be = GrammarBackend.get(model_path)
    if be is None:
        return None
    try:
        import xgrammar as xgr

        bm = xgr.allocate_token_bitmask(len(seq_items), be.vocab_size)
        # allocate_token_bitmask defaults to all-ones (allow-all); only tighten
        # rows that have an active matcher.
        filled = False
        for row, (sid, s) in enumerate(seq_items):
            spec = getattr(s, "structured_outputs", None)
            if not spec:
                continue
            m = matchers.get(sid)
            if m is None:
                m = be.compile(spec)
                if m is not None:
                    matchers[sid] = m
            if m is None:
                continue
            # State-driven advance (deferred/spec-safe): before filling the mask
            # for the NEXT token, bring the matcher up to every CONFIRMED real
            # output token so far. completion_token_ids (= token_ids[num_prompt:
            # num_tokens]) carries the real tokens PLUS a trailing EOS
            # placeholder slot (see the CRITICAL note below on n_real). Idempotent
            # via _gram_pos; if the seq was preempted and num_tokens rolled back,
            # we don't re-accept (pos already past) — the regenerated tokens match.
            try:
                out = list(s.completion_token_ids)
            except Exception:
                out = getattr(s, "output_tokens", None) or []
            # CRITICAL (deferred-output timing): completion_token_ids ends with
            # the scheduler's per-step EOS *placeholder* slot(s) — appended each
            # postprocess and OVERWRITTEN with the real deferred token on the
            # NEXT step. If we advance _gram_pos to len(out) we consume the
            # placeholder position; when the real token later lands at that same
            # index, _gram_pos has already passed it, so the matcher never sees
            # it. Net effect: the matcher freezes after the first token and the
            # grammar silently stops constraining (double-wrap regresses at
            # length / concurrency). Stop advancement at the last CONFIRMED real
            # token by trimming trailing EOS placeholder slot(s); that index is
            # re-read next step once the real token overwrites it. Real decode
            # tokens are never EOS mid-generation (an EOS finishes the seq), so
            # trimming the EOS tail strips exactly the placeholder(s).
            n_real = len(out)
            if be.eos is not None:
                while n_real > 0 and out[n_real - 1] == be.eos:
                    n_real -= 1
            pos = getattr(s, "_gram_pos", 0)
            if pos < n_real:
                for t in out[pos:n_real]:
                    try:
                        m.accept_token(int(t))
                    except Exception:
                        pass
                s._gram_pos = n_real
            # Do not constrain mid-prefill chunks (their sampled token is junk).
            if getattr(s, "is_partial_prefill", False):
                continue
            # Once the grammar is satisfied the matcher terminates: leave the row
            # all-ones (allow-all, incl. EOS) so the model stops naturally after
            # the tool call. This — NOT a manual EOS-allow — is what fixes "won't
            # stop". Do NOT force-allow EOS while the matcher is still mid-tool-
            # call: with tool_choice="required" that hands the model an early exit
            # (it emits a little reasoning then EOS with no tool call → "No tool
            # calls found"). The now-correct matcher advance keeps the required
            # tool call enforced until it completes, then terminates → EOS opens.
            if m.is_terminated():
                continue
            m.fill_next_token_bitmask(bm, row)
            filled = True
        return bm.numpy() if filled else None
    except Exception as e:  # noqa: BLE001
        logger.warning("fill bitmask failed, decoding unconstrained: %s", e)
        return None
