"""
draft_record.py -- record what the DSpark drafter would have to predict, while the target decodes.

Why this exists: the drafter's acceptance is the only decode lever left. The verify step is ~145 ms
and flat across workloads, so tok/s is `accept_len_mean / 0.145` and nothing else -- 5.07 accepted
tokens a step on single-file HTML is 36.6 tok/s, 2.46 on a long story is 17.1 (RESULTS.md 4.3). The
head shipped in the checkpoint is what it is; fine-tuning it on the target's OWN outputs is the
only way to move the prose row without touching the target.

Fine-tuning needs three things per position, and all three are already computed and thrown away on
every decode step:

  * the target's last hidden state -- `main_hidden`, the concatenation of the attention inputs of
    layers 37/38/39 averaged over the HC copies, [15360] fp32. This is the drafter's ENTIRE view of
    the target (`Model.dspark_seed`: main_proj -> main_norm -> per-block wkv -> the MTP window
    ring), so recording it before `main_proj` keeps `mtp.0.main_proj` trainable.
  * the token the target settled on at that position.
  * the target's next-token distribution at that position, top-32. Forward KL against the target
    needs a distribution, and the top-32 of a 129,280-vocabulary softmax carries essentially all of
    its mass at the temperatures this server is used at. Storing the full row would be 517 KB per
    token and is not worth it.

With those on disk the trainer never loads the 510 GB target: it runs the three MTP blocks alone
against recorded hidden states. That is what makes fine-tuning fit on one box.

Format (little-endian, fixed-size records, no per-record header):

    int32    position          absolute position in the sequence
    int32    token             the token the target settled on AT `position`
    int32    flags             bit 0: this record carries a next-token distribution
    int32    top_ids[32]       ids of the target's top-32 next-token logits, descending
    float32  top_vals[32]      the logit values, as the verify step produced them
    uint16   hidden[15360]     bfloat16 bits of `main_hidden` at `position`

    = 30,988 bytes per record; 300,000 recorded positions = 9.30 GB.

`flags` exists because of the prefill tail. A request generates ~175 tokens but the drafter's window
is 128 positions, so without context rows only the last ~47 positions of a run would have a full
window and three quarters of the data would be unusable. So the last 128 PROMPT positions are
recorded too -- their hidden states come out of the same prefill pass -- with the distribution bit
clear: prefill computes logits for the last position only. The trainer uses them as window context
and never as a loss target.

Enabled by `DSV41_RECORD_DRAFT_DATA=<dir>`, which is read once at import. With it unset this module
is never imported and the engine's decode loop runs one `is not None` test per verified block; the
arithmetic of a decode step is untouched either way.

One shard per request, `shard-<utc>-<pid>-<n>.bin`, plus a `.json` manifest written on open and
rewritten on close (so an aborted request still leaves a readable shard: the record count is
`filesize // record_bytes`, never a field that has to be trusted).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

# The drafter's input width: dim x len(dspark_target_layer_ids) = 5120 x 3. Passed in by the engine
# from its own Args rather than hard-coded, but pinned here as the default the format was sized for.
HIDDEN_DIM = 15360
TOPK = 32
FORMAT_VERSION = 1

FLAG_HAS_DIST = 1


def record_dtype(hidden_dim: int = HIDDEN_DIM, topk: int = TOPK) -> np.dtype:
    """The on-disk record, as a numpy structured dtype. Reader and writer share this one definition;
    `hidden` is the raw bfloat16 bits (numpy has no bfloat16), which torch views back with
    `torch.frombuffer(buf, dtype=torch.int16).view(torch.bfloat16)`."""
    return np.dtype([
        ("position", "<i4"),
        ("token", "<i4"),
        ("flags", "<i4"),
        ("top_ids", "<i4", (topk,)),
        ("top_vals", "<f4", (topk,)),
        ("hidden", "<u2", (hidden_dim,)),
    ])


def record_bytes(hidden_dim: int = HIDDEN_DIM, topk: int = TOPK) -> int:
    return record_dtype(hidden_dim, topk).itemsize


def estimate_gb(n_records: int, hidden_dim: int = HIDDEN_DIM, topk: int = TOPK) -> float:
    return n_records * record_bytes(hidden_dim, topk) / 1e9


def manifest(shard: str, hidden_dim: int, topk: int, meta: dict, runs: list, n_records: int) -> dict:
    """What a .json next to a .bin says. `runs` are the contiguous position ranges inside the shard:
    a training sample may never straddle two of them, because the window of the first position of a
    run belongs to a different sequence."""
    return {
        "format": "dsv41-draft-data",
        "version": FORMAT_VERSION,
        "shard": os.path.basename(shard),
        "hidden_dim": hidden_dim,
        "topk": topk,
        "record_bytes": record_bytes(hidden_dim, topk),
        "records": n_records,
        "runs": runs,
        "meta": dict(meta or {}),
    }


class DraftRecorder:
    """Writes one shard per request. Created once per engine process; `open_run`/`close_run` bracket
    a generation."""

    def __init__(self, root: str, hidden_dim: int = HIDDEN_DIM, topk: int = TOPK, meta: dict | None = None):
        self.root = root
        self.hidden_dim = int(hidden_dim)
        self.topk = int(topk)
        self.meta = dict(meta or {})
        self.dt = record_dtype(self.hidden_dim, self.topk)
        self.seq = 0
        self.f = None
        self.path = None
        self.runs = []
        self.n = 0
        self._run_start = None
        os.makedirs(root, exist_ok=True)

    # ------------------------------------------------------------------ run lifecycle
    def open_run(self, tag: str = "", prompt_tokens: int = 0):
        self.close_run("superseded")
        self.seq += 1
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        self.path = os.path.join(self.root, f"shard-{stamp}-{os.getpid()}-{self.seq:05d}.bin")
        self.f = open(self.path, "wb", buffering=1 << 20)
        self.runs = []
        self.n = 0
        self._run_start = None
        self.meta_run = {"tag": tag, "prompt_tokens": int(prompt_tokens), "started": stamp}
        self._write_manifest()
        return self

    def close_run(self, reason: str = "done"):
        if self.f is None:
            return
        self._flush_run()
        self.f.close()
        self.f = None
        self.meta_run["finished"] = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        self.meta_run["reason"] = reason
        self._write_manifest()
        if self.n == 0:  # an aborted request that never settled a token leaves nothing behind
            for p in (self.path, self.path[:-4] + ".json"):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def _flush_run(self):
        if self._run_start is not None:
            self.runs.append(self._run_start)
            self._run_start = None

    def _write_manifest(self):
        if self.path is None:
            return
        meta = dict(self.meta)
        meta.update(getattr(self, "meta_run", {}))
        runs = list(self.runs) + ([self._run_start] if self._run_start else [])
        m = manifest(self.path, self.hidden_dim, self.topk, meta, runs, self.n)
        with open(self.path[:-4] + ".json", "w") as fh:
            json.dump(m, fh, indent=1)

    # ------------------------------------------------------------------ writing
    def add(self, positions, tokens, hidden_bits, top_vals, top_ids, has_dist: bool = True):
        """Append `n` records. Arrays are numpy, already on the host:
        positions [n] int, tokens [n] int, hidden_bits [n, hidden_dim] uint16 (bfloat16 bits),
        top_vals [n, topk] float32, top_ids [n, topk] int32. `has_dist=False` writes the
        distribution fields as -1 / -inf and clears the flag (prefill context rows)."""
        n = len(positions)
        if n == 0:
            return
        rec = np.empty(n, self.dt)
        rec["position"] = positions
        rec["token"] = tokens
        rec["flags"] = FLAG_HAS_DIST if has_dist else 0
        if has_dist:
            rec["top_ids"] = top_ids
            rec["top_vals"] = top_vals
        else:
            rec["top_ids"] = -1
            rec["top_vals"] = -np.inf
        rec["hidden"] = hidden_bits
        self.f.write(rec.tobytes())
        self.n += n
        p0 = int(positions[0])
        if self._run_start is None:
            self._run_start = {"index": self.n - n, "position": p0, "n": n}
        elif p0 == self._run_start["position"] + self._run_start["n"]:
            self._run_start["n"] += n
        else:  # a gap in positions: the prefill tail, or a rollback. Start a new run.
            self.runs.append(self._run_start)
            self._run_start = {"index": self.n - n, "position": p0, "n": n}

    # ------------------------------------------------------------------ the engine's entry points
    def add_block(self, block, logits, main_hidden, pos: int, n: int):
        """One verified decode block: rows 0..n-1 of the static buffers, for positions pos..pos+n-1.

        `block` [T_VERIFY] long, `logits` [T_VERIFY, V] fp32 (row i is the distribution AFTER
        position pos+i) and `main_hidden` [T_VERIFY, 15360] fp32 are the fast path's own buffers and
        are only valid until the next step, so everything is copied to the host here. The device
        work is one top-k over n rows and three D2H copies of ~31 KB per token.
        """
        if n <= 0 or self.f is None:
            return
        import torch  # local: this module is importable without torch, for the format tests

        with torch.inference_mode():
            vals, ids = logits[:n].float().topk(self.topk, dim=-1)
            hid = main_hidden[:n].to(torch.bfloat16).view(torch.int16)
            toks = block[:n].to(torch.int32)
            hid_np = hid.cpu().numpy().view(np.uint16)
            self.add(np.arange(pos, pos + n, dtype=np.int64), toks.cpu().numpy(),
                     hid_np, vals.cpu().numpy(), ids.to(torch.int32).cpu().numpy())

    def add_context(self, token_ids, main_hidden, pos: int):
        """The prefill tail: hidden states for prompt positions pos..pos+M-1, no distribution. These
        are window context for the first decode positions of the run and nothing else."""
        if self.f is None or main_hidden is None:
            return
        import torch

        with torch.inference_mode():
            m = main_hidden.size(0)
            hid = main_hidden.to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint16)
            toks = torch.as_tensor(token_ids, dtype=torch.int32).reshape(-1)[-m:].cpu().numpy()
            self.add(np.arange(pos, pos + m, dtype=np.int64), toks, hid, None, None, has_dist=False)


# ----------------------------------------------------------------------------- reading
def read_shard(path: str, hidden_dim: int = HIDDEN_DIM, topk: int = TOPK) -> np.ndarray:
    """memmap a shard as records. The count comes from the file size, not from the manifest: a
    shard whose process was killed mid-request is still completely readable up to the last whole
    record."""
    dt = record_dtype(hidden_dim, topk)
    n = os.path.getsize(path) // dt.itemsize
    return np.memmap(path, dtype=dt, mode="r", shape=(n,))


def load_manifest(path: str) -> dict:
    with open(path[:-4] + ".json" if path.endswith(".bin") else path) as fh:
        return json.load(fh)
