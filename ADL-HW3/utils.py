from typing import List
import re
    
# def get_inference_system_prompt() -> str:
#     """get system prompt for generation"""
#     prompt ="You are a retrieval-augmented question-answering assistant. "
            
#     return prompt


# def get_inference_user_prompt(query: str, context_list: List[str]) -> str:
#     """Create the user prompt for generation given a query and a list of context passages."""
    
#     prompt =""
#     return prompt

# def parse_generated_answer(pred_ans: str) -> str:
#     """Extract the actual answer from the model's generated text."""
#     parsed_ans = pred_ans
#     return parsed_ans

def get_inference_system_prompt() -> str:
    """Get system prompt for generation."""
    prompt = (
        "You are a retrieval-augmented question-answering assistant. "
        "Answer the user's question using only the provided context passages. "
        "Do not use outside knowledge. "
        "If the answer cannot be found in the context, say: I don't know. "
        "Keep the answer concise and factual."
    )
    return prompt


def get_inference_user_prompt(query: str, context_list: List[str]) -> str:
    """Create the user prompt for generation given a query and a list of context passages."""

    context_text = ""
    for i, context in enumerate(context_list, start=1):
        context_text += f"[Passage {i}]\n{context}\n\n"

    prompt = f"""Use the following context passages to answer the question.

Context:
{context_text}

Question:
{query}

Answer:"""

    return prompt


def parse_generated_answer(pred_ans: str) -> str:
    """Extract the actual answer from the model's generated text."""

    parsed_ans = pred_ans.strip()

    # If the model repeats the prompt and includes "Answer:", keep only the answer part.
    if "Answer:" in parsed_ans:
        parsed_ans = parsed_ans.split("Answer:")[-1].strip()

    # Remove common unwanted endings if needed.
    stop_words = [
        "\nQuestion:",
        "\nContext:",
        "\nPassage",
    ]

    for stop in stop_words:
        if stop in parsed_ans:
            parsed_ans = parsed_ans.split(stop)[0].strip()

    return parsed_ans