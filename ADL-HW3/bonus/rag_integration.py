"""Small adapter for an existing RAG implementation with reusable callbacks."""
from .data import build_features


def collect_query_record(qid, query_embedding, reranked_passages, reranker_scores,
                         *, count_tokens, generate_answer, score_answer,
                         reference_answer, actions=(1, 3, 5), top_k=10):
    if len(reranked_passages) != len(reranker_scores):
        raise ValueError("Passages and reranker scores must be aligned")
    if len(reranked_passages) < max(actions) or max(actions) > top_k:
        raise ValueError("Not enough retrieved candidates for all passage-count actions")
    lengths = [int(count_tokens(p)) for p in reranked_passages]
    features = build_features(query_embedding, reranker_scores, lengths, top_k)
    answers = [str(generate_answer(reranked_passages[:m])) for m in actions]
    rewards = [float(score_answer(answer, reference_answer)) for answer in answers]
    return {"qid": str(qid), "features": features.tolist(), "rewards": rewards,
            "candidate_count": len(reranked_passages), "answers": answers,
            "context_tokens": [sum(lengths[:m]) for m in actions]}
