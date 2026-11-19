"""Reuse the assignment's FAISS/SQLite index and trained RAG checkpoints.

Heavy model dependencies are imported only when RAGBackend is constructed.
The supplied assignment scripts are never imported or modified.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sqlite3
import numpy as np
from .data import build_features



def add_backend_arguments(parser):
    parser.add_argument("--retriever-model-path", default="models/retriever")
    parser.add_argument("--reranker-model-path", default="models/reranker")
    parser.add_argument("--index-folder", default="vector_database")
    parser.add_argument("--index-file", default="passage_index.faiss")
    parser.add_argument("--sqlite-file", default="passage_store.db")
    parser.add_argument("--utils-path", default="utils.py")
    parser.add_argument("--generator-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=1280)

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_passage_db(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Passage database not found: {path}; run save_embeddings.py first")
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)


def lookup_passages(connection, rowids):
    ids = [int(i) for i in rowids]
    if not ids or min(ids) < 0:
        raise ValueError("FAISS returned missing or invalid candidates")
    marks = ",".join("?" for _ in ids)
    rows = connection.execute(f"SELECT rowid, pid, text FROM passages WHERE rowid IN ({marks})", ids).fetchall()
    by_id = {rowid: (str(pid), text) for rowid, pid, text in rows}
    if any(rowid not in by_id for rowid in ids):
        raise ValueError("FAISS row IDs do not match SQLite; rebuild both from the same corpus")
    return [by_id[rowid] for rowid in ids]


def read_queries(path):
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            if qid in seen or not isinstance(row.get("rewrite"), str) or not row["rewrite"].strip():
                raise ValueError("Queries need unique IDs and nonempty rewrite strings")
            seen.add(qid)
            rows.append({**row, "qid": qid})
    if not rows:
        raise ValueError("Query file is empty")
    return rows


def load_split_queries(train_path, split_path, train_limit=100, val_limit=50, seed=42):
    if train_limit < 0 or val_limit < 0:
        raise ValueError("Limits must be nonnegative; zero means all queries")
    rows = {r["qid"]: r for r in read_queries(train_path)}
    split = json.loads(Path(split_path).read_text())
    if split.get("train_sha256") and split["train_sha256"] != file_hash(train_path):
        raise ValueError("train.txt does not match the file used to create split.json")
    training = [str(q) for q in split["train_qids"]]
    validation = [str(q) for q in split["validation_qids"]]
    if not training or not validation or set(training) & set(validation):
        raise ValueError("Split must have separate, nonempty training and validation query sets")
    if len(set(training)) != len(training) or len(set(validation)) != len(validation):
        raise ValueError("Split contains duplicate query IDs")
    if (set(training) | set(validation)) - rows.keys():
        raise ValueError("Split contains queries absent from the training file")
    rng = random.Random(seed)
    rng.shuffle(training)
    rng.shuffle(validation)
    return ([rows[q] for q in (training[:train_limit] if train_limit else training)],
            [rows[q] for q in (validation[:val_limit] if val_limit else validation)])





def backend_from_args(args):
    return RAGBackend(**{name: getattr(args, name) for name in [
        "retriever_model_path", "reranker_model_path", "index_folder", "index_file", "sqlite_file",
        "utils_path", "generator_model", "precision", "device", "top_k", "max_new_tokens"]})


class RAGBackend:
    def __init__(self, retriever_model_path="models/retriever", reranker_model_path="models/reranker",
                 index_folder="vector_database", index_file="passage_index.faiss", sqlite_file="passage_store.db",
                 utils_path="utils.py", generator_model="Qwen/Qwen3-1.7B", precision="bf16",
                 device="cuda", top_k=10, max_new_tokens=1280):
        import torch
        import faiss
        from sentence_transformers import SentenceTransformer, CrossEncoder
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if top_k < 1 or max_new_tokens < 1:
            raise ValueError("top_k and max_new_tokens must be positive")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; for debugging use --device cpu --precision fp32")
        if precision == "bf16" and (not device.startswith("cuda") or not torch.cuda.is_bf16_supported()):
            raise ValueError("bf16 requires a supported CUDA GPU; otherwise use --precision fp32")
        self.torch, self.device = torch, device
        self.top_k, self.max_new_tokens = top_k, max_new_tokens
        utils_path = Path(utils_path)
        if not utils_path.is_file():
            raise FileNotFoundError(f"Prompt helper file not found: {utils_path}")
        spec = importlib.util.spec_from_file_location("adl_bonus_user_prompts", utils_path)
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)
        self.system_prompt = helpers.get_inference_system_prompt
        self.user_prompt = helpers.get_inference_user_prompt
        self.parse_answer = helpers.parse_generated_answer
        index_path = Path(index_folder) / index_file
        self.connection = open_passage_db(Path(index_folder) / sqlite_file)
        self.index = faiss.read_index(str(index_path))
        if self.index.ntotal < top_k:
            raise ValueError("The FAISS index has fewer passages than top_k")
        self.retriever = SentenceTransformer(retriever_model_path, device=device)
        self.retriever.eval()
        self.reranker = CrossEncoder(reranker_model_path, device=device)
        self.reranker.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        self.generator = AutoModelForCausalLM.from_pretrained(generator_model, dtype=dtype).to(device)
        self.generator.eval()
        self.scorer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        self.scorer.eval()
        fingerprints = {}
        for name, checkpoint in (("retriever", retriever_model_path), ("reranker", reranker_model_path)):
            folder = Path(checkpoint)
            files = sorted([p for p in folder.glob("*") if p.is_file()
                            and (p.suffix in {".safetensors", ".bin"} or p.name == "config.json")])
            fingerprints[name] = ({p.name: file_hash(p) for p in files} if files else str(checkpoint))
        identity = {"checkpoints": fingerprints, "index_sha256": file_hash(index_path),
                    "utils_sha256": file_hash(utils_path), "generator": generator_model,
                    "generator_revision": getattr(self.generator.config, "_commit_hash", None),
                    "precision": precision, "top_k": top_k, "max_new_tokens": max_new_tokens,
                    "do_sample": False, "enable_thinking": False,
                    "scorer": "sentence-transformers/all-MiniLM-L6-v2"}
        self.metadata = {"synthetic_demo": False,
                         "feature_schema": f"embedding-scores-lengths-k{top_k}-v1",
                         "pipeline_id": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                         "pipeline": identity}

    def retrieve(self, query):
        with self.torch.inference_mode():
            embedding = self.retriever.encode(["query: " + query], normalize_embeddings=True,
                                              convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            if embedding.shape[1] != self.index.d:
                raise ValueError("Retriever dimension does not match FAISS index")
            _, ids = self.index.search(embedding, self.top_k)
            passages = lookup_passages(self.connection, ids[0].tolist())
            scores = np.asarray(self.reranker.predict([(query, text) for _, text in passages],
                                show_progress_bar=False), dtype=np.float32).reshape(-1)
        order = np.argsort(-scores, kind="stable")
        pids = [passages[i][0] for i in order]
        texts = [passages[i][1] for i in order]
        ranked_scores = scores[order]
        lengths = [len(self.tokenizer.encode(text, add_special_tokens=False)) for text in texts]
        return {"query_embedding": embedding[0], "pids": pids, "texts": texts,
                "scores": ranked_scores, "token_lengths": lengths,
                "retrieved_pids": [pid for pid, _ in passages],
                "features": build_features(embedding[0], ranked_scores, lengths, self.top_k)}

    def generate(self, query, contexts):
        messages = [{"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": self.user_prompt(query, contexts)}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        width = inputs["input_ids"].shape[1]
        context_limit = getattr(self.generator.config, "max_position_embeddings", None)
        if context_limit and width + self.max_new_tokens > context_limit:
            raise ValueError("Prompt exceeds the generator context budget; use a consistent truncation policy and regenerate caches")
        with self.torch.inference_mode():
            output = self.generator.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        # Decode only the completion, never the question and retrieved context.
        completion = self.tokenizer.decode(output[0, width:], skip_special_tokens=True)
        return str(self.parse_answer(completion.strip()))

    def score(self, generated, reference):
        vectors = self.scorer.encode([generated, reference], normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False)
        return float(np.dot(vectors[0], vectors[1]))

    def close(self):
        self.connection.close()
