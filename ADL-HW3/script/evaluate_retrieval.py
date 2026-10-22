"""Full-corpus retrieval/reranking evaluation without loading the generator.

This is a development tool. Use the unmodified TA scripts for official tests.
"""
import argparse
import hashlib
import json
from pathlib import Path

from rag_data import load_data, ranking_metrics


class CorpusIndex:
    def __init__(self, retriever, corpus, batch_size=16):
        import faiss
        from tqdm import tqdm
        self.ids = list(corpus)
        self.corpus = corpus
        self.index = None
        for start in tqdm(range(0, len(self.ids), batch_size), desc='Indexing corpus for validation'):
            ids = self.ids[start:start + batch_size]
            vectors = retriever.encode(['passage: ' + corpus[pid] for pid in ids],
                                       normalize_embeddings=True, convert_to_numpy=True,
                                       batch_size=batch_size, show_progress_bar=False).astype('float32')
            if self.index is None:
                self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)

    def search(self, retriever, queries, batch_size):
        vectors = retriever.encode(['query: ' + query for query in queries],
                                   normalize_embeddings=True, convert_to_numpy=True,
                                   batch_size=batch_size, show_progress_bar=False).astype('float32')
        _, indices = self.index.search(vectors, min(10, len(self.ids)))
        return [[self.ids[int(index)] for index in row if index >= 0] for row in indices]


def evaluate_models(retriever, reranker, index, examples, batch_size=16):
    import numpy as np
    import torch
    if not examples:
        raise ValueError('Evaluation requires at least one query')
    retriever.eval()
    if reranker is not None:
        reranker.model.eval()
    recall, raw_mrr, reranked_mrr = 0.0, 0.0, 0.0
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            rankings = index.search(retriever, [item.query for item in batch], batch_size)
            for item, ids in zip(batch, rankings):
                hit, reciprocal = ranking_metrics(ids, set(item.positive_ids))
                recall += hit
                raw_mrr += reciprocal
                if reranker is not None:
                    scores = np.asarray(reranker.predict(
                        [(item.query, index.corpus[pid]) for pid in ids],
                        batch_size=batch_size, show_progress_bar=False)).reshape(-1)
                    reranked = [pid for _, pid in sorted(zip(scores.tolist(), ids),
                                                         key=lambda pair: pair[0], reverse=True)]
                    reranked_mrr += ranking_metrics(reranked, set(item.positive_ids))[1]
    metrics = {'queries': len(examples), 'recall@10': recall / len(examples),
               'retriever_mrr@10': raw_mrr / len(examples)}
    if reranker is not None:
        metrics['reranker_mrr@10'] = reranked_mrr / len(examples)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus_path', default='data/corpus.txt')
    parser.add_argument('--train_path', default='data/train.txt')
    parser.add_argument('--qrels_path', default='data/qrels.txt')
    parser.add_argument('--split_path', default='runs/retriever/split.json')
    parser.add_argument('--retriever_model_path', default='models/retriever')
    parser.add_argument('--reranker_model_path', default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_queries', type=int, default=0)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output_path', default='results/heldout_ranking.json')
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_queries < 0:
        raise ValueError('batch_size must be positive; max_queries must be nonnegative')
    corpus, rows, _ = load_data(args.corpus_path, args.train_path, args.qrels_path)
    split = json.loads(Path(args.split_path).read_text(encoding='utf-8'))
    if hashlib.sha256(Path(args.train_path).read_bytes()).hexdigest() != split['train_sha256']:
        raise ValueError('train.txt differs from the file used to create this split')
    by_qid = {row.qid: row for row in rows}
    wanted = split['validation_qids']
    missing = set(wanted) - by_qid.keys()
    if missing:
        raise ValueError(f'Validation queries missing from data: {sorted(missing)[:10]}')
    valid = [by_qid[qid] for qid in wanted]
    if args.max_queries:
        valid = valid[:args.max_queries]
    from sentence_transformers import SentenceTransformer, CrossEncoder
    retriever = SentenceTransformer(args.retriever_model_path, device=args.device)
    reranker = (CrossEncoder(args.reranker_model_path, device=args.device)
                if args.reranker_model_path else None)
    index = CorpusIndex(retriever, corpus, args.batch_size)
    metrics = evaluate_models(retriever, reranker, index, valid, args.batch_size)
    result = {'retriever': args.retriever_model_path, 'reranker': args.reranker_model_path,
              'split_path': args.split_path, **metrics}
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_path).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
