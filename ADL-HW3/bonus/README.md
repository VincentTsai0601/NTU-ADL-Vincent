# ADL HW3 bonus: dynamic passage selection with Gymnasium + PPO

This package adds a learned top-M controller to your existing RAG system. It reuses
`models/retriever`, `models/reranker`, `vector_database/passage_index.faiss`,
`vector_database/passage_store.db`, and your `utils.py` prompt functions. It does not
modify the supplied training/indexing/testing scripts or train Qwen.

The policy chooses M=1,3,5 after reranking the top ten candidates and before generation.
Each question is a one-step episode (a contextual bandit). PPO learns only a small MLP.

## 1. Install and run the synthetic learning demo

Use your Python 3.12 / PyTorch 2.8 HW3 environment. PyTorch 2.8+ and Python 3.10+
are supported by the pinned SB3 version. These commands run from the extracted
`ADL_HW3_RL_Bonus` directory:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v

python -m bonus.make_demo_cache --output-dir bonus/cache/demo
python -m bonus.train_ppo \
  --train-cache bonus/cache/demo/train.npz \
  --val-cache bonus/cache/demo/val.npz \
  --output-dir runs/ppo_demo --timesteps 10240
python -m bonus.evaluate --run-dir runs/ppo_demo \
  --cache bonus/cache/demo/val.npz --output results/demo_evaluation.json
```

The synthetic features deliberately expose an easy three-class pattern. Its rewards
are artificial. Demo scores are NOT assignment results and the demo policy cannot
be used by the real RAG inference script. No pretrained policy is shipped.

Use a fresh output directory for another training run, e.g. `runs/ppo_demo_v2`.

## 2. Connect to your original RAG project

Copy the `bonus/` directory into your existing `ADL-HW3` directory. Keep your existing
`utils.py`, data, models, requirements, and original inference script. Run all real
RAG commands below from that ADL-HW3 project root, in the same environment.

Your original preparation order is still valid:

```bash
# Run training only if you do not already have the desired checkpoints.
python3 ./script/train_retriever.py --precision bf16
python3 ./script/save_embeddings.py --retriever_model_path models/retriever --build_db
python3 ./script/train_reranker.py --retriever_model_path models/retriever --precision bf16
```

If those models and their matching index already exist, start at step 3 below.
The original `python3 inference_batch.py ...` remains your original fixed-M baseline.
The RL pipeline reuses its artifacts; it does not call or import that script.
Your GitHub repository currently stores these original scripts under `script/`.
If your checkout has that layout, use paths such as `script/train_retriever.py`
in the preparation commands, while keeping the project root as the working directory.

Model-backed scripts require the original HW3 packages (`transformers==4.56.1`,
`sentence-transformers==5.1.0`, a working FAISS installation, and torch). The small
`requirements.txt` only adds the controller dependencies. Do not install a second
FAISS distribution over a working one. The bonus supports the assignment's existing
FAISS GPU package; CPU FAISS also works for its index operations.

## 3. Collect real rewards with your trained RAG components

```bash
python -m bonus.cache_rewards \
  --train-path data/train.txt \
  --split-path runs/retriever/split.json \
  --retriever-model-path models/retriever \
  --reranker-model-path models/reranker \
  --index-folder vector_database \
  --utils-path utils.py \
  --train-limit 100 --val-limit 50 \
  --output-dir bonus/cache/real
```

This reads ONLY queries in the training file and your saved query split. It does NOT
use `test_open.txt` to train the controller. When the split has a `train_sha256`, it
must match your current training file. Train and validation IDs must be disjoint.

For each selected query, retrieval/reranking happens once, then Qwen answers using
M=1,3,5. At most 150 queries x 3 actions = 450 generations for these limits. Fewer
queries are used if your split is smaller. Use `--train-limit 2 --val-limit 2` and a
different output directory for a four-query smoke test first.

Outputs:
- `train.npz`, `val.npz`: controller observations and reward tables.
- `train_records.jsonl`, `val_records.jsonl`: answers, context-token counts, IDs,
  rewards, and references for manual review.
- `metadata.json`: pipeline/checkpoint fingerprints and settings.
- `selected_queries.json`: exact selected query IDs.

Feature order is query embedding, ten reranked scores, then ten passage token lengths
using the Qwen tokenizer. Missing positions are zero padded. The candidate set must
be large enough for all actions. Only observations are supplied to the policy;
gold answers and passage labels never enter the feature vector.

Reward = cosine similarity between the generated answer and reference using
`sentence-transformers/all-MiniLM-L6-v2`. There is no token-cost penalty in this
initial version. Qwen is in eval/inference mode, bf16 on a supported CUDA device,
with thinking disabled and greedy decoding. Default max_new_tokens is 1280.
Only newly generated tokens are decoded and then passed through your answer parser.
If a prompt exceeds Qwen's context limit, the script fails rather than silently
changing the action. Add a consistent truncation policy before rebuilding caches
if your data needs it.

If interrupted, JSONL records written so far remain available, but incomplete
NPZ files are not emitted. See section 7 for conversion of complete/salvaged JSONL
records. When salvaging real records pass the saved `metadata.json` so inference
can verify the pipeline fingerprint. Both splits still need unique, separate IDs.

## 4. Train the controller on CPU

```bash
python -m bonus.train_ppo \
  --train-cache bonus/cache/real/train.npz \
  --val-cache bonus/cache/real/val.npz \
  --output-dir runs/ppo_real --timesteps 10240
```

The expensive work is generating the reward cache. PPO reuses that deterministic
environment response while sampling its current policy's actions. 10,240 PPO
timesteps are NOT 10,240 new LLM generations. This is not LLM fine-tuning or RLHF.

PPO settings: 64/64 hidden layers, lr=3e-4, n_steps=256, batch_size=64,
n_epochs=10, gamma=0, gae_lambda=1, entropy coefficient=0.01, seed=42. A rollout
contains many one-step episodes. Features are normalized with training statistics
only. Validation checks choose the best deterministic policy; the initial
untrained policy is also eligible. If selected at timestep 0, do not claim PPO
training improved it. Policy loss need not decrease monotonically.

Artifacts in the run directory:
- `best_policy.zip`, `final_policy.zip` (SB3 checkpoints).
- `feature_stats.npz` (required at inference).
- `config.json` (IDs, versions, hashes, hyperparameters).
- `selection.json` (when/why best policy was chosen).
- `evaluation.csv`, `training.monitor.csv`, `validation_metrics.json`.

## 5. Run RL retrieval -> reranking -> generation

```bash
python -m bonus.inference_bonus \
  --run-dir runs/ppo_real \
  --retriever-model-path models/retriever \
  --reranker-model-path models/reranker \
  --index-folder vector_database \
  --test-data-path data/test_open.txt \
  --output results/rl_bonus.json
```

The bonus inference script chooses M from features before reading any reference
answer for scoring. It can run on unlabeled questions as well; label-dependent
metrics become null. For labeled data it reports answer similarity, average M,
context tokens, and available Recall@K/MRR@K. All query rows are processed unless
you specify `--limit`; a limit of zero means all rows. JSONL predictions are also
written incrementally as `rl_bonus.records.jsonl`.
Here Recall@K is the fraction of all positive passage IDs retrieved. Your original
script calls the any-positive hit rate Recall@K; those definitions differ when a
query has multiple positive passages. Compare metrics with the same definition.

For a fair baseline use the SAME bonus backend with a fixed M:

```bash
python -m bonus.inference_bonus \
  --fixed-m 3 \
  --retriever-model-path models/retriever \
  --reranker-model-path models/reranker \
  --index-folder vector_database \
  --test-data-path data/test_open.txt \
  --output results/fixed_m3_bonus_backend.json
```

Repeat with `--fixed-m 1` and `--fixed-m 5`, using different output files. Compare
the same query IDs, generator settings, prompts, and candidate order. Bonus scores
are development results, not a guarantee about the TA's unchanged evaluation.
The bonus backend deliberately uses completion-only parsing and greedy generation;
do not attribute differences from an original full-prompt/stochastic run solely
to RL. Recall@10 and MRR@10 should be unchanged between matched bonus runs.

Do not use `test_open` feedback to optimize the policy. If selecting checkpoints
or hyperparameters on validation, reserve a third query-disjoint split for final
evaluation. Query-disjointness does not guarantee that the same document or
paraphrased questions are absent across splits; stronger grouping is optional.

Changing checkpoints, the index, prompt helpers, generation settings, or feature
construction requires regenerating the real caches and retraining the policy.
The pipeline fingerprint check rejects synthetic or mismatched real policies.

## 6. Use only the trained selector in your own RAG loop

```python
from bonus.inference import PassageSelector
from bonus.data import build_features

selector = PassageSelector("runs/ppo_real")  # load once

# All inputs below must be aligned in descending reranker-score order.
raw_features = build_features(
    query_embedding,
    reranker_scores,
    passage_token_lengths,  # Qwen tokenizer, add_special_tokens=False
    top_k=10,
)
top_m = selector.select_top_m(raw_features)
context_list = reranked_passages[:top_m]
# Then call your usual prompt builder and generator.
```

No gold answer or reward table is required by PassageSelector. Keep its feature
schema and normalization identical to training. The cache collector also ensures
the candidate set can support every action. If writing your own inference loop,
preserve that requirement instead of silently clipping an unavailable action.

For a batch of inference observations in JSONL:
```bash
python -m bonus.inference --run-dir runs/ppo_real \
  --input query_features.jsonl --output results/top_m_decisions.jsonl
```
Each input row needs `qid` and `features`, or the three named feature fields in
section 7. This command emits passage counts only, not generated answers.

## 7. Bring your own reward cache

Existing `.pt` dictionaries with `qids`, `features`, `rewards`, `actions` are supported
for controller training. They must contain CPU tensors or primitive values readable
by `torch.load(weights_only=True)`. For a new cache prefer NPZ (no pickle).

You can also convert JSONL records:
```json
{"qid":"train-query-1","features":[0.1,0.2,0.3],"rewards":[0.4,0.8,0.5]}
```
These numbers are schema examples only, not measured rewards. Real `features` may
instead be supplied as `query_embedding`, `reranker_scores`, and
`passage_token_lengths`, with `candidate_count` optionally included. Use the exact
same feature representation at deployment. Do not mix raw-vector and named-field
representations within a cache.

```bash
python -m bonus.build_cache --input real_train_records.jsonl \
  --metadata bonus/cache/real/metadata.json --output bonus/cache/custom_train.npz
```

For another RAG implementation, `bonus.rag_integration.collect_query_record` accepts
callbacks for generation, scoring, and token counting. Your callback should close
over the query and fixed model/tokenizer. It calls the generator separately for
each top-M prefix and returns an auditable JSON-compatible record.

`bonus.evaluate` compares a trained selector with fixed-M rewards without loading
Qwen. It rejects policy-training IDs; it labels validation used for checkpoint
selection separately from a new holdout. The printed oracle uses labels to pick
the best action and is an upper-bound diagnostic, not a deployable model.

## Troubleshooting

- Run `python -m bonus...` from the folder CONTAINING `bonus/`, not from inside it.
- Missing FAISS database/index: finish the original `save_embeddings.py` command.
- Do not rebuild the SQLite/index concurrently with cache collection or inference.
- Missing split: use the `split.json` produced by your retriever training, or create
  an explicit train_qids/validation_qids split from training queries before RL.
- No CUDA: model-backed scripts support `--device cpu --precision fp32`, but real
  generation will be slow. PPO itself uses CPU regardless.
- Existing output: use a fresh run/cache directory. No deletion is performed.
- Pip may find a conflicting pin for old SB3/Gymnasium or torch; use a dedicated
  compatible environment rather than replacing the assignment's testing environment.

## References

- https://gymnasium.farama.org/api/env/
- https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
- https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

The synthetic demo and tests validate software behavior, not assignment performance.
The assignment dataset and trained checkpoints are not bundled. Actual full-model
GPU reward collection and assignment scores must be verified in your environment.
