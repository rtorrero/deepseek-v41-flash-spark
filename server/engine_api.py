"""Engine interface for the OpenAI-compatible server.

The server talks to an inference engine through the small abstract class below.
The real DeepSeek-V4.1-Flash engine lives in ``engine/v41_engine.py`` (repo
root) and is imported lazily by ``app.py --engine v41``; ``MockEngine`` is a
tokenizer-only stand-in so the HTTP layer can be exercised without a GPU.

Contract (see ``Engine``):

* ``generate`` is a generator that yields *bursts* of newly generated token ids.
  Speculative decoding (DSpark) typically produces several tokens per step, so
  a burst is ``list[int]`` of length >= 1. The engine should stop after it has
  emitted a token in ``stop_token_ids`` (the stop token itself may be included
  in the last burst; the server truncates at the first stop id it sees) and it
  must stop after ``max_tokens`` tokens at the latest. After an EOS/stop id or
  ``max_tokens`` the server drains the generator (up to 4 more bursts, ignored)
  so an engine whose loop ends there gets to run its epilogue (stats). On a
  stop *string* or a client disconnect the server closes the generator
  instead -- ``GeneratorExit`` is raised at the pending ``yield`` -- so
  cache/arena cleanup belongs in a ``try/finally`` around the decode loop and
  the engine must be ready for the next request afterwards.
* Optional ``grammar`` keyword: a decoding gate, passed to any engine whose
  ``supports_grammar`` is true. A request that carries tools gets the tool-call
  grammar; one that does not gets the plain-text gate, which only keeps the DSML
  bar out of a completion that has no legal use for it. The engine owes the gate
  two calls, and nothing else:

      ``gate.observe(ids)``      every token the loop has settled on, in order,
                                 once (a burst at a time is fine).
      ``gate.mask_rows(logits, block_ids)``
                                 mask ``logits`` in place before anything reads
                                 it. ``logits`` is [R, V] (R = 1 for a plain
                                 step, or the verify block's rows) and
                                 ``block_ids`` the R token ids those rows follow
                                 -- ``[tok, draft0, ...]`` -- so that row i can
                                 be masked for the state after block[0..i]. Pass
                                 ``block_ids=None`` when there is only one row.
                                 The call leaves the gate's own state untouched,
                                 so speculation that is rolled back needs no
                                 undo; rows past the first illegal draft are
                                 left alone, which is safe because that draft is
                                 masked out of its own row and is therefore
                                 rejected there.

  The tool-call gate masks nothing until the model opens a tool-calls block
  (beyond one token that has no legal use outside it), and the plain-text gate
  masks that one token, so an engine may call both unconditionally.
  ``server/tool_grammar.py`` implements them.
* Optional ``think`` keyword: reasoning-span controls, passed to any engine whose
  ``supports_think_controls`` is true, and only for a request with thinking on.
  Same two-call contract as the gate above -- ``observe(ids)`` for every settled
  token and ``mask_rows(logits, block_ids)`` before the accept/reject decision --
  except that it only ever masks **row 0**, because both of its decisions are
  functions of the settled history. Call it *after* any penalty pass: it can
  force ``</think>`` by masking everything else out of the row, and a penalty
  applied afterwards could ban that one surviving token. See
  ``server/think_controls.py``; with ``think=None`` nothing changes.
* Optional ``context_margin`` (int, default 8 as read by the server): tokens
  the engine needs beyond ``len(prompt_ids) + max_tokens`` (DSpark draft
  block); the server clamps ``max_tokens`` so that
  ``prompt + max_tokens + margin <= max_context``.
* The server serialises calls: at most one ``generate`` runs at any time.
* ``stats`` returns whatever the engine wants to expose about the last
  generation (tok/s, draft acceptance rate, expert cache hit rate ...). It is
  passed through verbatim as ``x_engine_stats`` in responses.
"""

from __future__ import annotations

import json
import random
import re
import time
from abc import ABC, abstractmethod
from typing import Callable, Iterator, List, Optional, Set


class Engine(ABC):
    """Single-sequence text generation engine."""

    #: id of ``<｜end▁of▁sentence｜>`` (1 for DeepSeek-V4.1).
    eos_token_id: int = 1
    #: maximum prompt + completion length the engine accepts.
    max_context: int = 32768
    #: does ``generate`` honour a ``grammar`` gate? The server only builds one if it does.
    supports_grammar: bool = False
    #: does ``generate`` honour a ``think`` control object? Same rule: no builder if not.
    supports_think_controls: bool = False

    @abstractmethod
    def generate(
        self,
        prompt_ids: List[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: Set[int],
        seed: Optional[int],
    ) -> Iterator[List[int]]:
        """Yield bursts of newly generated token ids for ``prompt_ids``."""

    def stats(self) -> dict:
        """Engine statistics for the most recent generation (free-form)."""
        return {}

    def close(self) -> None:
        """Release resources; called once at server shutdown."""


# The mock reply, split into the pieces the mock engine assembles depending on
# what the prompt asks for.
MOCK_REASONING = (
    "The user is talking to a mock engine. I should answer briefly, "
    "mention that this is canned text, and keep the tone friendly."
)
MOCK_CONTENT = (
    "Hello from the mock engine! This is a canned reply.\n\n"
    "It streams in small bursts of one to four tokens so the SSE path, "
    "the incremental detokenizer and the reasoning/content split can be "
    "tested end to end without a GPU."
)
MOCK_TOOL_CONTENT = "Let me look that up."

_TOOL_SCHEMAS_HEADING = "### Available Tool Schemas\n\n"


class MockEngine(Engine):
    """Tokenizer-only engine that replays a canned reply in random bursts.

    * If the prompt ends with ``<think>`` the reply starts with a reasoning
      block terminated by ``</think>``; if it ends with ``</think>`` (chat
      mode) the reply is content only.
    * A ``think`` control object is honoured: before each reasoning token the
      mock asks it the same question the real decode loop asks (``force_token``)
      and, when the budget is spent, emits ``</think>`` there instead and carries
      on with the answer. That is what the real engine achieves by masking every
      other token out of the row, so the HTTP layer -- the resolved budget, the
      split into ``reasoning_content``/``content``, ``reasoning_tokens`` and
      ``reasoning_budget_hit`` -- can be tested end to end without a GPU.
    * If the prompt carries a V4.1 tool schema block the reply is a valid DSML
      tool call against the first listed tool (first declared parameter,
      string-typed), so tool-call parsing can be verified.
    * Bursts have 1-4 tokens and the sequence ends with ``eos_token_id``.
    """

    def __init__(
        self,
        encode: Callable[[str], List[int]],
        decode: Callable[[List[int]], str],
        *,
        eos_token_id: int = 1,
        think_start_id: int = 128821,
        think_end_id: int = 128822,
        max_context: int = 32768,
        burst_delay_s: float = 0.0,
    ) -> None:
        self._encode = encode
        self._decode = decode
        self.eos_token_id = eos_token_id
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id
        self.max_context = max_context
        self.burst_delay_s = burst_delay_s
        self._last_stats: dict = {"engine": "mock"}

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _first_tool(prompt_text: str):
        """Return (name, first_param_name) of the first tool schema, or None."""
        pos = prompt_text.rfind(_TOOL_SCHEMAS_HEADING)
        if pos < 0:
            return None
        block = prompt_text[pos + len(_TOOL_SCHEMAS_HEADING):]
        first_line = block.split("\n", 1)[0]
        try:
            schema = json.loads(first_line)
            name = schema["name"]
            props = (schema.get("parameters") or {}).get("properties") or {}
            param = next(iter(props), "query")
            return name, param
        except Exception:
            m = re.search(r'"name":\s*"([^"]+)"', first_line)
            return (m.group(1), "query") if m else None

    def _reply_text(self, prompt_ids: List[int]) -> str:
        thinking = bool(prompt_ids) and prompt_ids[-1] == self.think_start_id
        prompt_text = self._decode(prompt_ids)
        tool = self._first_tool(prompt_text)
        parts = []
        if thinking:
            parts.append(MOCK_REASONING + "</think>")
        if tool is not None:
            name, param = tool
            parts.append(MOCK_TOOL_CONTENT)
            parts.append(
                "\n\n<｜DSML｜ calls>\n"
                f'<｜DSML｜ invoke name="{name}">\n'
                f'<｜DSML｜ parameter name="{param}" string="true">Berlin</｜DSML｜ parameter>\n'
                "</｜DSML｜ invoke>\n"
                "</｜DSML｜ calls>"
            )
        else:
            parts.append(MOCK_CONTENT)
        return "".join(parts)

    def _with_think_controls(self, ids: List[int], think) -> List[int]:
        """Replay ``ids`` past the controls, cutting the reasoning where they say.

        The canned ids are never re-tokenized -- the list is walked as it is and
        the forced ``</think>`` replaces the token that would have come next --
        so with the controls off, or with a budget the canned reasoning never
        reaches, the reply is byte-for-byte the one every other test sees.
        """
        if think is None or not getattr(think, "active", False) or self.think_end_id not in ids:
            return ids
        close_at = ids.index(self.think_end_id)
        out: List[int] = []
        for i, t in enumerate(ids):
            forced = think.force_token()
            if forced is not None:
                think.observe([forced])
                return out + [forced] + ids[close_at + 1:]
            out.append(t)
            think.observe([t])
            if not think.in_think:          # the canned close went by; the rest is the answer
                return out + ids[i + 1:]
        return out

    # -- Engine ----------------------------------------------------------
    supports_think_controls = True

    def generate(
        self,
        prompt_ids: List[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: Set[int],
        seed: Optional[int],
        think=None,
    ) -> Iterator[List[int]]:
        ids = self._with_think_controls(self._encode(self._reply_text(prompt_ids)), think)
        ids = (ids + [self.eos_token_id])[:max_tokens]
        rng = random.Random(seed if seed is not None else 1234)
        t0 = time.perf_counter()
        emitted = 0
        i = 0
        try:
            while i < len(ids):
                burst = ids[i:i + rng.randint(1, 4)]
                i += len(burst)
                emitted += len(burst)
                if self.burst_delay_s:
                    time.sleep(self.burst_delay_s)
                yield burst
                if any(t in stop_token_ids for t in burst):
                    break
        finally:
            # The server closes the generator as soon as it has seen a stop id
            # or a stop string, so stats must be finalised here, not after the
            # loop. A real engine should do the same (and abort its sequence).
            dt = max(time.perf_counter() - t0, 1e-9)
            self._last_stats = {
                "engine": "mock",
                "tok_per_s": round(emitted / dt, 1),
                "acceptance_rate": 1.0,
                "expert_cache_hit_rate": 1.0,
                "generated_tokens": emitted,
                "prompt_tokens": len(prompt_ids),
            }

    def stats(self) -> dict:
        return dict(self._last_stats)
