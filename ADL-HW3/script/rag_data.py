"""Assignment data preparation; deliberately has no ML dependencies."""
from dataclasses import dataclass
import json
from pathlib import Path
import random
import warnings

RETRIEVER = 'intfloat/multilingual-e5-small'
RERANKER = 'cross-encoder/ms-marco-MiniLM-L-12-v2'

@dataclass(frozen=True)
class QueryExample:
    qid: str
    query: str
    positive_ids: tuple[str, ...]
    negative_ids: tuple[str, ...]

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f'{path}:{line_no}: invalid JSON') from error

def load_corpus(path):
    corpus = {}
    for row in read_jsonl(path):
        pid = str(row['id'])
        if pid in corpus:
            raise ValueError(f'Duplicate corpus passage id: {pid}')
        if not isinstance(row['text'], str) or not row['text'].strip():
            raise ValueError(f'Empty/non-string text for passage {pid}')
        corpus[pid] = row['text']
    if not corpus:
        raise ValueError('Corpus is empty')
    return corpus


def load_data(corpus_path, train_path, qrels_path):
    """Only queries in train_path become examples, regardless of qrels contents.

    Evidence may be a corpus id, passage text, title-plus-passage text, or a
    dict containing id/text. Whitespace differences are normalized for lookup
    only; original corpus text and the evaluation corpus remain unchanged.
    Unmatched negatives are counted and omitted; sample_negatives fills gaps
    from the corpus. Unmatched positive evidence requires another known positive.
    CANNOTANSWER is a generation label; it does not erase retrieval positives.
    """
    corpus = load_corpus(corpus_path)
    normalize = lambda text: ' '.join(text.split())
    text_to_id = {normalize(text): pid for pid, text in corpus.items()}
    title_text_to_id = {}
    for record in read_jsonl(corpus_path):
        title = record.get('title')
        if isinstance(title, str) and title.strip():
            title_text_to_id[normalize(title + ' ' + record['text'])] = str(record['id'])

    def resolve_text(text):
        if not isinstance(text, str):
            return None
        key = normalize(text)
        return text_to_id.get(key, title_text_to_id.get(key))
    with Path(qrels_path).open(encoding='utf-8') as stream:
        qrels = json.load(stream)
    examples, seen = [], set()
    stats = {'queries_read': 0, 'queries_without_positive': 0,
             'queries_without_bm25_negative': 0,
             'unmatched_negative_evidences': 0,
             'unmatched_positive_evidences': 0}
    for row in read_jsonl(train_path):
        stats['queries_read'] += 1
        qid = str(row['qid'])
        if qid in seen:
            raise ValueError(f'Duplicate query id: {qid}; split would leak queries')
        seen.add(qid)
        query = row['rewrite']
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f'{qid}: rewrite must be a nonempty string')
        positive_ids = {str(pid) for pid, label in qrels.get(qid, {}).items()
                        if float(label) > 0}
        missing = positive_ids - corpus.keys()
        if missing:
            raise ValueError(f'{qid}: positive passages missing from corpus: {sorted(missing)}')
        evidences = row.get('evidences', [])
        labels = row.get('retrieval_labels', [])
        if len(evidences) != len(labels):
            raise ValueError(f'{qid}: evidences and retrieval_labels lengths differ')
        candidates = []
        unmatched_positive = False
        for evidence, label in zip(evidences, labels):
            if isinstance(evidence, str):
                pid = evidence if evidence in corpus else resolve_text(evidence)
            elif isinstance(evidence, dict):
                value = evidence.get('id', evidence.get('pid', evidence.get('passage_id')))
                if value is None and not isinstance(evidence.get('text'), str):
                    raise ValueError(f'{qid}: evidence dict requires an id or text field')
                pid = str(value) if value is not None else None
                if pid not in corpus:
                    pid = resolve_text(evidence.get('text'))
            else:
                raise ValueError(f'{qid}: unsupported evidence type {type(evidence).__name__}')
            if pid is None:
                if float(label) > 0:
                    stats['unmatched_positive_evidences'] += 1
                    unmatched_positive = True
                else:
                    stats['unmatched_negative_evidences'] += 1
                continue
            if float(label) > 0:
                positive_ids.add(pid)
            else:
                candidates.append(pid)
        if not positive_ids:
            if unmatched_positive:
                raise ValueError(f'{qid}: unmatched positive evidence and no positive qrels '
                                 'or resolvable positive evidence; check corpus/qrels files')
            stats['queries_without_positive'] += 1
            continue
        positive_texts = {corpus[pid] for pid in positive_ids}
        negatives = tuple(dict.fromkeys(pid for pid in candidates
                                        if corpus[pid] not in positive_texts))
        if not negatives:
            stats['queries_without_bm25_negative'] += 1
        examples.append(QueryExample(qid, query, tuple(sorted(positive_ids)), negatives))
    if not examples:
        raise ValueError('No queries with positives; check train/corpus/qrels formats')
    if stats['unmatched_negative_evidences'] or stats['unmatched_positive_evidences']:
        warnings.warn(
            f"Unmatched evidence: {stats['unmatched_negative_evidences']} negative(s) omitted "
            f"and {stats['unmatched_positive_evidences']} positive(s) unresolved. "
            'Queries retain known qrels/resolved positives; negative sampling fills '
            'missing negatives from the corpus. See data_stats for counts.',
            UserWarning, stacklevel=2)
    return corpus, examples, stats

def split_examples(examples, validation_ratio, seed):
    if not 0 < validation_ratio < 1:
        raise ValueError('validation_ratio must be strictly between 0 and 1')
    if len(examples) < 2:
        raise ValueError('Need at least two queries for a training/validation split')
    rows = list(examples)
    random.Random(seed).shuffle(rows)
    size = min(len(rows) - 1, max(1, round(len(rows) * validation_ratio)))
    return rows[size:], rows[:size]

def sample_negatives(example, corpus, count, rng, corpus_ids=None):
    """Prefer supplied BM25 negatives, then fill with random corpus passages.

    Deduplicate by text and exclude every known positive text. Return fewer than
    count when the corpus cannot provide enough valid distinct negatives.
    """
    positive_texts = {corpus[pid] for pid in example.positive_ids}
    selected, seen_texts = [], set(positive_texts)
    hard = list(example.negative_ids)
    rng.shuffle(hard)
    for pid in hard:
        if len(selected) == count:
            break
        if corpus[pid] not in seen_texts:
            selected.append(pid)
            seen_texts.add(corpus[pid])
    ids = corpus_ids if corpus_ids is not None else tuple(corpus)
    # Rejection sampling avoids copying/shuffling the entire corpus per query.
    for _ in range(max(20, count * 20)):
        if len(selected) >= count:
            break
        pid = rng.choice(ids)
        if corpus[pid] not in seen_texts:
            selected.append(pid)
            seen_texts.add(corpus[pid])
    if len(selected) < count:
        for pid in ids:
            if len(selected) >= count:
                break
            if corpus[pid] not in seen_texts:
                selected.append(pid)
                seen_texts.add(corpus[pid])
    return selected

def ranking_metrics(ranked_ids, gold_ids, k=10):
    """Match the TA's hit-style Recall@k, even with multiple gold passages."""
    rank = next((i for i, pid in enumerate(ranked_ids[:k], 1) if pid in gold_ids), None)
    return (1.0, 1.0 / rank) if rank else (0.0, 0.0)
