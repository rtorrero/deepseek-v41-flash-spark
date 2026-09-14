"""The recorder's shard, written the way the decode loop writes it.

Run: python3 engine/test_draft_record.py       (skips itself without torch; uses CUDA when there is
                                                one, because the buffers it copies from are the
                                                engine's own device tensors)

tools/test_draft_data_gen.py covers the format arithmetic with numpy alone. What is left, and what
this file is for, is the part that only torch can get wrong:

  * `main_hidden` is fp32 on the device and is stored as bfloat16 BITS. A wrong view there gives a
    shard full of plausible numbers that are off by a factor of 2^112, and nothing downstream
    raises.
  * `add_block(block, logits, main_hidden, pos, n)` must record rows 0..n-1 and NOT the rest of the
    static buffer: rows beyond the accepted prefix belong to drafts that were rejected, and the
    positions they claim are about to be overwritten by the next step.
  * the top-32 must be the top-32 of the row that FOLLOWS the recorded position, descending.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

try:
    import torch
except ImportError:  # noqa: BLE001
    print("skip: no torch on this machine (this test belongs on the box)")
    sys.exit(0)

import draft_record as DR  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev}")

fails = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


H, K, V, T = 64, 8, 50, 6   # hidden width, top-k, vocabulary, verify block
torch.manual_seed(0)

with tempfile.TemporaryDirectory() as d:
    rec = DR.DraftRecorder(d, hidden_dim=H, topk=K, meta={"block": T - 1})
    rec.open_run("t", prompt_tokens=7)

    # the prefill tail: hidden states for prompt positions, no distribution
    ctx = torch.randn(4, H, device=dev)
    ctx_ids = torch.arange(100, 104, device=dev)
    rec.add_context(ctx_ids, ctx, 3)

    # one verified block at positions 7..12 with three accepted rows
    block = torch.arange(20, 26, device=dev)
    logits = torch.randn(T, V, device=dev)
    mh = torch.randn(T, H, device=dev)
    rec.add_block(block, logits, mh, 7, 3)
    path = rec.path
    rec.close_run("done")

    r = DR.read_shard(path, H, K)
    man = DR.load_manifest(path)

    check("the shard holds the context rows and the accepted rows only", len(r) == 4 + 3, f"{len(r)}")
    check("the context rows keep their prompt positions and tokens",
          list(r["position"][:4]) == [3, 4, 5, 6] and list(r["token"][:4]) == [100, 101, 102, 103])
    check("the context rows carry no distribution", list(r["flags"][:4]) == [0, 0, 0, 0])
    check("the block rows are the settled positions",
          list(r["position"][4:]) == [7, 8, 9] and list(r["token"][4:]) == [20, 21, 22])
    check("the block rows carry a distribution",
          all(f & DR.FLAG_HAS_DIST for f in r["flags"][4:]))

    # bfloat16 bits, back to numbers
    got = torch.from_numpy(np.array(r["hidden"][4:]).view(np.int16)).view(torch.bfloat16).float()
    want = mh[:3].to(torch.bfloat16).float().cpu()
    check("the hidden state survives as bfloat16 bits", torch.equal(got, want),
          f"max |diff| {float((got - want).abs().max()):.3g}")
    got_ctx = torch.from_numpy(np.array(r["hidden"][:4]).view(np.int16)).view(torch.bfloat16).float()
    check("so does the prefill tail", torch.equal(got_ctx, ctx.to(torch.bfloat16).float().cpu()))

    v, i = logits[:3].topk(K, dim=-1)
    check("the top-k ids are the target's own", (np.array(r["top_ids"][4:]) == i.cpu().numpy()).all())
    check("the top-k values are the logits, not probabilities",
          np.allclose(np.array(r["top_vals"][4:]), v.cpu().numpy(), atol=1e-6))
    check("the top-k is descending",
          bool((np.diff(np.array(r["top_vals"][4:]), axis=-1) <= 1e-6).all()))

    check("the context and the block are one run (positions 3..9 are contiguous)",
          len(man["runs"]) == 1 and man["runs"][0]["n"] == 7, f"{man['runs']}")
    check("the manifest records the shape of the data",
          man["hidden_dim"] == H and man["topk"] == K and man["record_bytes"] == DR.record_bytes(H, K))
    check("the record count is the file size", man["records"] * man["record_bytes"] == os.path.getsize(path))

    # a rejected block: nothing is settled but the token the step started from
    rec2 = DR.DraftRecorder(d, hidden_dim=H, topk=K)
    rec2.open_run("t2", 0)
    rec2.add_block(block, logits, mh, 30, 1)
    p2 = rec2.path
    rec2.close_run("done")
    r2 = DR.read_shard(p2, H, K)
    check("a block that accepted nothing still records its one settled position",
          len(r2) == 1 and r2["position"][0] == 30)

    # an aborted request leaves nothing behind rather than an empty shard
    rec3 = DR.DraftRecorder(d, hidden_dim=H, topk=K)
    rec3.open_run("t3", 0)
    p3 = rec3.path
    rec3.close_run("abort")
    check("a run that settled nothing leaves no shard", not os.path.exists(p3))

print()
print(f"{len(fails)} failed: {', '.join(fails)}" if fails else "all checks passed")
sys.exit(1 if fails else 0)
