"""Shared CLI, data split, and optimizer loop for the two allowed models."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random

from rag_data import RETRIEVER, RERANKER, load_data, sample_negatives, split_examples


def parser_for(task):
    parser = argparse.ArgumentParser(description=f'Fine-tune the HW3 {task}')
    parser.add_argument('--corpus_path', default='data/corpus.txt')
    parser.add_argument('--train_path', default='data/train.txt')
    parser.add_argument('--qrels_path', default='data/qrels.txt')
    parser.add_argument('--output_dir', default=f'models/{task}')
    parser.add_argument('--log_dir', default=f'runs/{task}')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4 if task == 'retriever' else 16)
    parser.add_argument('--grad_accum_steps', type=int, default=4 if task == 'retriever' else 1)
    parser.add_argument('--negatives_per_query', type=int, default=2 if task == 'retriever' else 4)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--validation_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_queries', type=int, default=200,
                        help='Limit held-out evaluation cost; 0 evaluates every held-out query')
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--device', default=None, help='Default: CUDA when available, otherwise CPU')
    parser.add_argument('--prepare_only', action='store_true',
                        help='Validate data and save split without importing ML packages')
    if task == 'reranker':
        parser.add_argument('--retriever_model_path', default='models/retriever',
                            help='Fixed retriever used for full-corpus validation')
        
    return parser


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def positive_mask(examples, candidate_ids, corpus):
    """Known positives and text-identical passages are positive, even in-batch."""
    masks = []
    for example in examples:
        gold_texts = {corpus[pid] for pid in example.positive_ids}
        masks.append([corpus[pid] in gold_texts for pid in candidate_ids])
    return masks


def candidate_batch(examples, corpus, corpus_ids, negatives_per_query, rng):
    candidates, seen_texts = [], set()
    for example in examples:
        ids = [rng.choice(example.positive_ids)]
        ids += sample_negatives(example, corpus, negatives_per_query, rng, corpus_ids)
        for pid in ids:
            if corpus[pid] not in seen_texts:
                candidates.append(pid)
                seen_texts.add(corpus[pid])
    return candidates, positive_mask(examples, candidates, corpus)


def pair_examples(examples, corpus, corpus_ids, negatives_per_query, rng):
    pairs = []
    for example in examples:
        positive_id = rng.choice(example.positive_ids)
        pairs.append((example.query, corpus[positive_id], 1.0))
        negatives = sample_negatives(example, corpus, negatives_per_query, rng, corpus_ids)
        pairs.extend((example.query, corpus[pid], 0.0) for pid in negatives)
    rng.shuffle(pairs)
    return pairs


def retriever_loss(model, examples, corpus, corpus_ids, args, rng, device):
    import torch
    import torch.nn.functional as functional
    ids, masks = candidate_batch(examples, corpus, corpus_ids, args.negatives_per_query, rng)
    if not any(not value for mask in masks for value in mask):
        raise ValueError('Contrastive batch has no negative passages; increase corpus/batch size')

    def encode(texts):
        features = {key: value.to(device) for key, value in model.tokenize(texts).items()}
        return functional.normalize(model(features)['sentence_embedding'].float(), dim=-1)

    queries = encode(['query: ' + example.query for example in examples])
    passages = encode(['passage: ' + corpus[pid] for pid in ids])
    logits = (queries @ passages.T).float() / args.temperature
    mask = torch.tensor(masks, dtype=torch.bool, device=device)
    # Multi-positive InfoNCE. No known positive is forced to be a negative.
    positive_logits = logits.masked_fill(~mask, float('-inf'))
    return (torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()


def reranker_loss(model, batch, args, device):
    import torch
    import torch.nn.functional as functional
    features = model.tokenizer([item[0] for item in batch], [item[1] for item in batch],
                               padding=True, truncation=True, max_length=args.max_length,
                               return_tensors='pt')
    features = {key: value.to(device) for key, value in features.items()}
    logits = model.model(**features).logits.view(-1).float()
    labels = torch.tensor([item[2] for item in batch], device=device)
    return functional.binary_cross_entropy_with_logits(logits, labels)


def main(task):
    args = parser_for(task).parse_args()
    for name in ['epochs', 'batch_size', 'grad_accum_steps', 'negatives_per_query',
                 'eval_batch_size', 'max_length']:
        if getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive')
    if args.max_length > 512:
        raise ValueError('Both assignment checkpoints support at most 512 tokens')
    if args.temperature <= 0 or args.learning_rate <= 0:
        raise ValueError('temperature and learning_rate must be positive')
    if not 0 <= args.warmup_ratio <= 1 or args.eval_queries < 0:
        raise ValueError('Invalid warmup_ratio or eval_queries')
    output, logs = Path(args.output_dir), Path(args.log_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} is nonempty; use a fresh output directory')
    if logs.exists() and any(logs.iterdir()):
        raise ValueError(f'{logs} is nonempty; use a fresh log directory')
    corpus, examples, stats = load_data(args.corpus_path, args.train_path, args.qrels_path)
    train, valid = split_examples(examples, args.validation_ratio, args.seed)
    logs.mkdir(parents=True, exist_ok=True)
    train_hash = hashlib.sha256(Path(args.train_path).read_bytes()).hexdigest()
    save_json(logs / 'split.json', {'seed': args.seed, 'train_sha256': train_hash,
                                  'train_qids': [x.qid for x in train],
                                  'validation_qids': [x.qid for x in valid]})
    save_json(logs / 'config.json', {'task': task, 'base_checkpoint': RETRIEVER if task == 'retriever' else RERANKER,
                                   **vars(args), 'data_stats': stats,
                                   'training_queries': len(train), 'validation_queries': len(valid)})
    print(json.dumps({**stats, 'train': len(train), 'validation': len(valid)}, indent=2))
    if args.prepare_only:
        print(f'Data validated. Split saved to {logs / "split.json"}. No models trained.')
        return

    import torch
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from transformers import get_linear_schedule_with_warmup
    from tqdm import tqdm
    from evaluate_retrieval import CorpusIndex, evaluate_models

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if args.precision != 'fp32' and device.type != 'cuda':
        raise ValueError('Use fp32 on CPU; mixed precision here is supported on CUDA only')
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support bf16; use fp16 or fp32')
    if task == 'retriever':
        model = SentenceTransformer(RETRIEVER, device=str(device))
        model.max_seq_length = args.max_length
        retriever, reranker = model, None
        parameters = model.named_parameters()
    else:
        model = CrossEncoder(RERANKER, num_labels=1, max_length=args.max_length, device=str(device))
        retriever = SentenceTransformer(args.retriever_model_path, device=str(device))
        reranker = model
        parameters = model.model.named_parameters()
    decay, no_decay = [], []
    for name, parameter in parameters:
        if parameter.requires_grad:
            (no_decay if name.endswith('bias') or 'LayerNorm' in name else decay).append(parameter)
    optimizer = torch.optim.AdamW([
        {'params': decay, 'weight_decay': args.weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}], lr=args.learning_rate)
    corpus_ids = tuple(corpus)
    epoch_size = len(train) if task == 'retriever' else sum(
        1 + len(sample_negatives(x, corpus, args.negatives_per_query,
                                 random.Random(args.seed), corpus_ids)) for x in train)
    micro_steps = math.ceil(epoch_size / args.batch_size)
    steps_per_epoch = math.ceil(micro_steps / args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                    num_warmup_steps=int(total_steps * args.warmup_ratio),
                    num_training_steps=total_steps)
    use_amp = args.precision != 'fp32'
    amp_dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=args.precision == 'fp16')
    valid = valid[:args.eval_queries] if args.eval_queries else valid
    index = None
    metrics_history = []
    best_score = None
    optimizer_step = 0
    with (logs / 'loss.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['step', 'epoch', 'loss', 'learning_rate'])
        writer.writeheader()
        # Evaluate and save baseline too: fine-tuning is not guaranteed to help.
        for epoch in range(args.epochs + 1):
            if epoch:
                rng = random.Random(args.seed + epoch)
                batches = list(train) if task == 'retriever' else pair_examples(
                    train, corpus, corpus_ids, args.negatives_per_query, rng)
                rng.shuffle(batches)
                model.train() if task == 'retriever' else model.model.train()
                optimizer.zero_grad(set_to_none=True)
                epoch_micro_steps = math.ceil(len(batches) / args.batch_size)
                rolling_loss = 0.0
                for batch_no, start in enumerate(tqdm(range(0, len(batches), args.batch_size),
                                                     desc=f'{task} epoch {epoch}')):
                    batch = batches[start:start + args.batch_size]
                    group_start = (batch_no // args.grad_accum_steps) * args.grad_accum_steps
                    group_size = min(args.grad_accum_steps, epoch_micro_steps - group_start)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        loss = (retriever_loss(model, batch, corpus, corpus_ids, args, rng, device)
                                if task == 'retriever' else reranker_loss(model, batch, args, device))
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite loss; retry with fp32 or a lower learning rate')
                    rolling_loss += loss.detach().item() / group_size
                    scaler.scale(loss / group_size).backward()
                    if (batch_no + 1) % args.grad_accum_steps == 0 or batch_no + 1 == epoch_micro_steps:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters() if task == 'retriever' else model.model.parameters(), 1.0)
                        previous_scale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        # Do not advance the LR schedule on an fp16-overflow skipped step.
                        if scaler.get_scale() >= previous_scale:
                            scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_step += 1
                        writer.writerow({'step': optimizer_step, 'epoch': epoch,
                                         'loss': rolling_loss,
                                         'learning_rate': optimizer.param_groups[0]['lr']})
                        stream.flush()
                        rolling_loss = 0.0
            if task == 'retriever' or index is None:
                index = CorpusIndex(retriever, corpus, args.eval_batch_size)
            metrics = evaluate_models(retriever, reranker, index, valid, args.eval_batch_size)
            score = ((metrics['recall@10'], metrics['retriever_mrr@10']) if task == 'retriever'
                     else (metrics['reranker_mrr@10'],))
            improved = best_score is None or score > best_score
            if improved:
                best_score = score
                output.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(output))
                save_json(output / 'training_selection.json',
                          {'selected_epoch': epoch, 'metrics': metrics,
                           'base_checkpoint': RETRIEVER if task == 'retriever' else RERANKER,
                           'note': 'epoch 0 is the unmodified baseline'})
            metrics_history.append({'epoch': epoch, **metrics, 'saved_as_best': improved})
            save_json(logs / 'metrics.json', metrics_history)
            print(json.dumps(metrics_history[-1], indent=2))
    print(f'Best checkpoint saved at {output}; actual optimizer-step losses: {logs / "loss.csv"}')

