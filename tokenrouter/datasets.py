# data.py
"""Public dataset loaders and prompt construction for TokenRouter."""
import os
import re
from typing import Dict, List, Optional
from datasets import load_dataset

from .utils import extract_answer


DATASET_SPECS = {
    "GSM8K": {
        "path": os.getenv("TOKENROUTER_GSM8K_DATASET", "openai/gsm8k"),
        "name": "main",
        "split": "validation",
    },
    "MATH": {
        "path": os.getenv("TOKENROUTER_MATH_DATASET", "HuggingFaceH4/MATH-500"),
        "name": None,
        "split": "test",
    },
    "AIME": {
        "path": os.getenv("TOKENROUTER_AIME_DATASET", "HuggingFaceH4/aime_2024"),
        "name": None,
        "split": "train",
    },
}

def build_math_prompt(problem: str) -> str:
    """Creates a standardized instruction prompt for math problems."""
    return (
        "Solve the following math problem step-by-step. "
        "The final answer must be enclosed in a single \\boxed{} block. "
        f"Question: {problem}\n\n"
    )

class DatasetManager:
    """Manages loading and preprocessing of public reasoning datasets."""

    @staticmethod
    def _load_dataset(dataset_name: str):
        spec = DATASET_SPECS[dataset_name]
        if spec["name"] is None:
            return load_dataset(spec["path"], split=spec["split"])
        return load_dataset(spec["path"], spec["name"], split=spec["split"])

    @staticmethod
    def load_GSM8K(num_samples: Optional[int] = None) -> List[Dict]:
        print("Loading GSM8K dataset...")
        dataset = DatasetManager._load_dataset("GSM8K")
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        problems = []
        for item in dataset:
            # Extract final number from the solution string
            answer = re.findall(r'####\s*([0-9,]+)', item['answer'])
            answer = answer[0].replace(',', '') if answer else ""
            problems.append({'question': item['question'], 'answer': answer})
        return problems
    
    @staticmethod
    def load_MATH(num_samples: Optional[int] = None) -> List[Dict]:
        print("Loading MATH dataset...")
        dataset = DatasetManager._load_dataset("MATH")
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        problems = []
        for item in dataset:
            problems.append({
                'question': item['problem'], 
                'answer': extract_answer(item['solution']) or item['solution']
            })
        return problems
    
    @staticmethod
    def load_AIME(num_samples: Optional[int] = None) -> List[Dict]:
        """Loads the AIME 2024 dataset from Hugging Face."""
        print("Loading AIME dataset...")
        dataset = DatasetManager._load_dataset("AIME")
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        problems = []
        for item in dataset:
            answer_source = item.get('answer', item.get('Answer', item.get('solution', item.get('Solution', ""))))
            problems.append({
                'question': item.get('problem', item.get('Problem', "")),
                'answer': extract_answer(str(answer_source)) or str(answer_source),
            })
        return problems
    
