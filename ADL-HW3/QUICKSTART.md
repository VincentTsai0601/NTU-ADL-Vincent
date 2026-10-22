# Quick start

Unzip this package. Inside `adl_hw3`, put `corpus.txt`, `train.txt`,
`test_open.txt`, and `qrels.txt` in a folder called `data`.

```bash
cd adl_hw3
pip install -r requirements.txt

# Validate the data without loading model weights.
python3 train_retriever.py --prepare_only --log_dir runs/data_check

```
# 1. Train the retriever
```bash
python3 train_retriever.py --precision bf16
```

# 2. Encode the corpus using the trained retriever
```bash
python3 save_embeddings.py \
  --retriever_model_path models/retriever \
  --build_db
```

# 3. Train the reranker
```bash
python3 train_reranker.py \
  --retriever_model_path models/retriever \
  --precision bf16
```

# 4. Run retrieval → reranking → generation
```bash
python3 inference_batch.py \
  --retriever_model_path models/retriever \
  --reranker_model_path models/reranker \
  --test_data_path data/test_open.txt
```

# Inspecting the loss function
```bash
python3 plot_loss.py \
  --csv_path ./runs/retriever/loss.csv \
  --output_path ./figures/retriever_loss.svg \
  --title "Retriever Training Loss"
```

Create `.env` with your own `hf_token="..."` before running the supplied
inference script. Read `README.md` for model selection, report logs, validation,
memory settings, and known behaviors of the supplied testing script.

Checkpoints are created by your training runs; none are included in this ZIP.



