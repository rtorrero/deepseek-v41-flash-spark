# DSpark head verification -- 20260915T090142Z

head: weights/mtp_finetuned.safetensors
serving: EXPERT_PROFILE=frontend PRUNE_KEEP=0.36, thinking off for the two prompts,
thinking on for the gates.

| prompt | accept_len_mean | tok/s | tokens |
|---|---|---|---|
| prose | 2.77 | 14.53 | 400 |
| markup | 5.35 | 29.57 | 700 |

Gates: see GATE-*.md beside this file.

Training manifest:
```json
{
 "format": "dsv41-mtp-finetune",
 "tensors": [
  "mtp.0.attn.attn_sink",
  "mtp.0.attn.kv_norm.weight",
  "mtp.0.attn.q_norm.weight",
  "mtp.0.attn.wkv.weight",
  "mtp.0.attn.wo_a.weight",
  "mtp.0.attn.wo_b.weight",
  "mtp.0.attn.wq_a.weight",
  "mtp.0.attn.wq_b.weight",
  "mtp.0.attn_norm.weight",
  "mtp.0.ffn.gate.bias_vl",
  "mtp.0.ffn.gate.weight",
  "mtp.0.ffn.shared_experts.w1.weight",
  "mtp.0.ffn.shared_experts.w2.weight",
  "mtp.0.ffn.shared_experts.w3.weight",
  "mtp.0.ffn_norm.weight",
  "mtp.0.hc_attn_base",
  "mtp.0.hc_attn_fn",
  "mtp.0.hc_attn_scale",
  "mtp.0.hc_ffn_base",
  "mtp.0.hc_ffn_fn",
  "mtp.0.hc_ffn_scale",
  "mtp.0.main_norm.weight",
  "mtp.0.main_proj.weight",
  "mtp.1.attn.attn_sink",
  "mtp.1.attn.kv_norm.weight",
  "mtp.1.attn.q_norm.weight",
  "mtp.1.attn.wkv.weight",
  "mtp.1.attn.wo_a.weight",
  "mtp.1.attn.wo_b.weight",
  "mtp.1.attn.wq_a.weight",
  "mtp.1.attn.wq_b.weight",
  "mtp.1.attn_norm.weight",
  "mtp.1.ffn.gate.bias_vl",
  "mtp.1.ffn.gate.weight",
  "mtp.1.ffn.shared_experts.w1.weight",
  "mtp.1.ffn.shared_experts.w2.weight",
  "mtp.1.ffn.shared_experts.w3.weight",
  "mtp.1.ffn_norm.weight",
  "mtp.1.hc_attn_base",
  "mtp.1.hc_attn_fn",
  "mtp.1.hc_attn_scale",
  "mtp.1.hc_ffn_base",
  "mtp.1.hc_ffn_fn",
  "mtp.1.hc_ffn_scale",
  "mtp.2.attn.attn_sink",
  "mtp.2.attn.kv_norm.weight",
  "mtp.2.attn.q_norm.weight",
  "mtp.2.attn.wkv.weight",
  "mtp.2.attn.wo_a.weight",
  "mtp.2.attn.wo_b.weight",
  "mtp.2.attn.wq_a.weight",
  "mtp.2.attn.wq_b.weight",
  "mtp.2.attn_norm.weight",
  "mtp.2.ffn.gate.bias_vl",
  "mtp.2.ffn.gate.weight",
  "mtp.2.ffn.shared_experts.w1.weight",
  "mtp.2.ffn.shared_experts.w2.weight",
  "mtp.2.ffn.shared_experts.w3.weight",
  "mtp.2.ffn_norm.weight",
  "mtp.2.hc_attn_base",
  "mtp.2.hc_attn_fn",
  "mtp.2.hc_attn_scale",
  "mtp.2.hc_ffn_base",
  "mtp.2.hc_ffn_fn",
  "mtp.2.hc_ffn_scale",
  "mtp.2.markov_head.embed.weight",
  "mtp.2.markov_head.head.weight",
  "mtp.2.norm.weight"
 ],
 "trained": [
  "mtp.0.attn.attn_sink",
  "mtp.0.attn.kv_norm.weight",
  "mtp.0.attn.q_norm.weight",
  "mtp.0.attn.wkv.weight",
  "mtp.0.attn.wo_a.weight",
  "mtp.0.attn.wo_b.weight",
  "mtp.0.attn.wq_a.weight",
  "mtp.0.attn.wq_b.weight",
  "mtp.0.attn_norm.weight",
  "mtp.0.ffn.gate.bias_vl",
  "mtp.0.ffn.gate.weight",
  "mtp.0.ffn.shared_experts.w1.weight",
  "mtp.0.ffn.shared_experts.w2.weight",
  "mtp.0.ffn.shared_experts.w3.weight",
  "mtp.0.ffn_norm.weight",
  "mtp.0.hc_attn_base",
  "mtp.0.hc_attn_fn",
  "mtp.0.hc_attn_scale",
  "mtp.0.hc_ffn_base",
  "mtp.0.hc_ffn_fn",
  "mtp.0.hc_ffn_scale",
  "mtp.0.main_norm.weight",
  "mtp.0.main_proj.weight",
  "mtp.1.attn.attn_sink",
  "mtp.1.attn.kv_norm.weight",
  "mtp.1.attn.q_norm.weight",
  "mtp.1.attn.wkv.weight",
  "mtp.1.attn.wo_a.weight",
  "mtp.1.attn.wo_b.weight",
  "mtp.1.attn.wq_a.weight",
  "mtp.1.attn.wq_b.weight",
  "mtp.1.attn_norm.weight",
  "mtp.1.ffn.gate.bias_vl",
  "mtp.1.ffn.gate.weight",
  "mtp.1.ffn.shared_experts.w1.weight",
  "mtp.1.ffn.shared_experts.w2.weight",
  "mtp.1.ffn.shared_experts.w3.weight",
  "mtp.1.ffn_norm.weight",
  "mtp.1.hc_attn_base",
  "mtp.1.hc_attn_fn",
  "mtp.1.hc_attn_scale",
  "mtp.1.hc_ffn_base",
  "mtp.1.hc_ffn_fn",
  "mtp.1.hc_ffn_scale",
  "mtp.2.attn.attn_sink",
  "mtp.2.attn.kv_norm.weight",
  "mtp.2.attn.q_norm.weight",
  "mtp.2.attn.wkv.weight",
  "mtp.2.attn.wo_a.weight",
  "mtp.2.attn.wo_b.weight",
  "mtp.2.attn.wq_a.weight",
  "mtp.2.attn.wq_b.weight",
  "mtp.2.attn_norm.weight",
  "mtp.2.ffn.gate.bias_vl",
  "mtp.2.ffn.gate.weight",
  "mtp.2.ffn.shared_experts.w1.weight",
  "mtp.2.ffn.shared_experts.w2.weight",
  "mtp.2.ffn.shared_experts.w3.weight",
  "mtp.2.ffn_norm.weight",
  "mtp.2.hc_attn_base",
  "mtp.2.hc_attn_fn",
  "mtp.2.hc_attn_scale",
  "mtp.2.hc_ffn_base",
  "mtp.2.hc_ffn_fn",
  "mtp.2.hc_ffn_scale",
  "mtp.2.markov_head.embed.weight",
  "mtp.2.markov_head.head.weight",
  "mtp.2.norm.weight"
 ],
 "model_dir": "DeepSeek-V4.1-Flash",
 "created": "2026-09-15T09:54:17",
 "data": [
  "/home/bakeer/deepseek-v41-flash-spark/data/draft"
 ],
 "shards": 1015,
 "samples": 143953,
 "train_samples": 136314,
 "held_out_samples": 7639,
 "steps": 3000,
 "batch": 16,
 "lr": 2e-05,
 "beta": 0.6,
 "ce_weight": 1.0,
 "draft": 5,
 "window": 128,
 "groups": [
  "attn",
  "shared",
  "gate",
  "norms",
  "main",
  "markov"
 ],
 "markov_input": "self",
 "seed": 20260915,
 "accept_proxy_before": 3.8359375,
 "accept_proxy_after": 3.7578125,
 "accept_proxy_fp8": 3.869140625,
 "per_step_before": [
  0.8359375,
  0.708984375,
  0.61328125,
  0.51171875,
  0.4140625
 ],
 "per_step_after": [
  0.826171875,
  0.720703125,
  0.60546875,
  0.5078125,
  0.38671875
 ],
 "per_step_fp8": [
  0.8359375,
  0.724609375,
  0.62109375,
  0.5390625,
  0.427734375
 ],
 "eval_history": [
  {
   "step": 500,
   "per_step": [
    0.8203125,
    0.7109375,
    0.6015625,
    0.5,
    0.400390625
   ],
   "accept_proxy": 3.775390625
  },
  {
   "step": 1000,
   "per_step": [
    0.828125,
    0.708984375,
    0.6015625,
    0.50390625,
    0.37890625
   ],
   "accept_proxy": 3.7578125
  },
  {
   "step": 1500,
   "per_step": [
    0.828125,
    0.71875,
    0.603515625,
    0.51171875,
    0.3828125
   ],
   "accept_proxy": 3.810546875
  },
  {
   "step": 2000,
   "per_step": [
    0.826171875,
    0.71484375,
    0.61328125,
    0.513671875,
    0.37890625
   ],
   "accept_proxy": 3.78125
  },
  {
   "step": 2500,
   "per_step": [
    0.828125,
    0.71484375,
    0.609375,
    0.513671875,
    0.388671875
   ],
   "accept_proxy": 3.7578125
  },
  {
   "step": 3000,
   "per_step": [
    0.826171875,
    0.720703125,
    0.60546875,
    0.5078125,
    0.38671875
   ],
   "accept_proxy": 3.7578125
  }
 ],
 "trainable_params": 635811810
}```
