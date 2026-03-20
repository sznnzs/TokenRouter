# utils.py
"""
Utility functions for entropy calculation, answer string processing,
mathematical validation, and CUDA device management.
"""
import os
import re
import numpy as np
from contextlib import contextmanager
from collections import deque
from typing import Dict, List, Optional

from .math_parsing import strip_answer_string, math_equal

# ===================================================================
# CUDA Environment Utility
# ===================================================================

@contextmanager
def set_cuda_devices(device_ids: List[int]):
    """Context manager to temporarily set CUDA_VISIBLE_DEVICES."""
    old_device = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))
    try:
        yield
    finally:
        if old_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_device
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]

# ===================================================================
# Lightweight Entropy Utilities
# ===================================================================

class OnlineEntropyPeakDetector:
    """Online Z-Score based peak detection for entropy monitoring."""
    def __init__(self, lag: int, threshold: float, influence: float = 0.2):
        self.lag = lag
        self.threshold = threshold
        self.influence = influence
        self.history = deque(maxlen=lag * 2)
        self.filtered_y = deque(maxlen=lag)
        
    def add_datapoint(self, value: float) -> str:
        self.history.append(value)
        if len(self.history) < self.lag:
            self.filtered_y.append(value)
            return "NORMAL"
        
        window = list(self.history)[-self.lag:]
        mean = np.mean(window)
        std_dev = np.std(window)
        
        if std_dev == 0: return "NORMAL"
        
        z_score = abs((value - mean) / std_dev)
        
        if z_score > self.threshold:
            self.filtered_y.append(self.influence * value + (1 - self.influence) * self.filtered_y[-1])
            return "PEAK"
        else:
            self.filtered_y.append(value)
            return "NORMAL"

def calculate_shannon_entropy(logprobs: Dict[int, float], vocab_size: int) -> float:
    """Calculate Shannon entropy from a logprobs dictionary."""
    probs = np.zeros(vocab_size)
    for token_id, logprob_obj in logprobs.items():
        # --- ADD THIS CHECK ---
        # If the token_id is outside the vocabulary of the target model, skip it.
        if token_id >= vocab_size:
            continue
        logprob_val = logprob_obj.logprob 
        probs[token_id] = np.exp(logprob_val)
    
    probs_sum = probs.sum()
    if probs_sum > 1e-10:
        probs = probs / probs_sum
    
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return entropy

# ===================================================================
# Answer Parsing and Validation Utilities (Updated)
# ===================================================================

def extract_answer(pred_str: str, use_last_number: bool = True) -> Optional[str]:
    """Parses out the final expression or numeric value from a typical LLM chain-of-thought."""
    pred_str = pred_str.replace("\u043a\u0438", "")
    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        # minerva_math
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "答案是" in pred_str:
        # Handle Chinese few-shot multiple choice problem answer extraction
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    elif "**Final Answer:**" in pred_str or "Final Answer:" in pred_str:
        if "**Final Answer:**" in pred_str:
            answer_text = pred_str.split("**Final Answer:**")[1]
        else:
            answer_text = pred_str.split("Final Answer:")[1]
            
        for marker in ["</think>", "\n\n"]:
            if marker in answer_text:
                answer_text = answer_text.split(marker)[0]
            
        pattern = r"\$?(\d+(?:\.\d+)?)R"
        numbers = re.findall(pattern, answer_text.replace(",", ""))
        if numbers:
            pred = numbers[0]
        else:
            words = [w for w in answer_text.split() if w]
            pred = words[0] if words else ""
    else:  # use the last number
        if use_last_number:
            pattern = r"-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
        else:
            pred = ""

    # multiple line
    # pred = pred.split("\n")[0]
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_answer_string(pred)

    return pred


def is_correct_answer(model_answer_str: Optional[str], ground_truth_str: str) -> bool:
    
    cleaned_ans = strip_answer_string(ground_truth_str)
    cleaned_pred = strip_answer_string(model_answer_str)
    # Check correctness
    correct = math_equal(cleaned_pred, cleaned_ans)
    return correct
