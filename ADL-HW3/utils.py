from typing import List
import re
    
# def get_inference_system_prompt() -> str:
#     """get system prompt for generation"""
#     prompt = "" 
#     return prompt

# def get_inference_user_prompt(query : str, context_list : List[str]) -> str:
#     """Create the user prompt for generation given a query and a list of context passages."""
#     prompt = f""""""
#     return prompt

# def parse_generated_answer(pred_ans: str) -> str:
#     """Extract the actual answer from the model's generated text."""
#     parsed_ans = pred_ans
#     return parsed_ans

def get_inference_system_prompt():
    return (
        "You are a precise QA assistant.\n"
        "Rules:\n"
        "1. Always output ONLY the short final answer.\n"
        "2. Do NOT output explanations, reasoning, or extra sentences.\n"
        "3. If no answer is available, output exactly 'CANNOTANSWER'.\n"
        "4. Match the style of the gold answer (usually short word/phrase).\n"
    )

def get_inference_user_prompt(query, contexts):
    # contexts = list of retrieved passages (top K)
    context_text = "\n".join([f"Passage: {c}" for c in contexts])
    return (
        f"Question: {query}\n"
        f"Context:\n{context_text}\n\n"
        f"Answer (short and concise, no extra text):"
    )

def parse_generated_answer(ans: str) -> str:
    # 後處理：確保結果乾淨
    ans = ans.strip()
    # 移除開頭的 "Answer:" 之類字樣
    for prefix in ["Answer:", "答案：", "Final Answer:", "A:", "Ans:"]:
        if ans.startswith(prefix):
            ans = ans[len(prefix):].strip()
    # 如果模型輸出空字串，就回傳 "CANNOTANSWER"
    if not ans:
        return "CANNOTANSWER"
    return ans