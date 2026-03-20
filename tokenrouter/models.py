"""vLLM-backed model wrappers used by TokenRouter experiments."""
from typing import Dict, List, Tuple, Union

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from .config import ModelConfig, GLOBAL_MAX_TOKENS
from .utils import set_cuda_devices

class VLLMModel:
    """vLLM-based language model for local inference."""

    def __init__(
        self,
        model_path: str,
        config: ModelConfig,
        gpu_ids: Union[int, List[int]],
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
    ):
        self.model_path = model_path
        self.config = config
        self.gpu_ids = gpu_ids if isinstance(gpu_ids, list) else [gpu_ids]

        with set_cuda_devices(self.gpu_ids):
            print(f"Initializing model on GPU(s): {self.gpu_ids} with tensor_parallel_size={tensor_parallel_size}")
            self.model = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=GLOBAL_MAX_TOKENS,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype="bfloat16",
                trust_remote_code=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.vocab_size = len(self.tokenizer)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token
        print(f"Model {model_path.split('/')[-1]} uses EOS token ID: '{self.eos_token_id}'")

    def generate(
        self,
        prompts: Union[str, List[str]] = None,
        prompt_token_ids: List[List[int]] = None,
        max_tokens: int = 50,
    ) -> List[str]:
        """Generate text given prompts or token_ids."""
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be provided.")

        sampling_params = SamplingParams(temperature=self.config.temperature, max_tokens=max_tokens)
        outputs = self.model.generate(prompts, sampling_params, prompt_token_ids=prompt_token_ids, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]

    def generate_one_token(self, prompts: Union[str, List[str]] = None, prompt_token_ids: List[List[int]] = None) -> List[Tuple[str, int, Dict[int, float]]]:
        """Generate a single token and return logprobs."""
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be provided.")

        sampling_params = SamplingParams(temperature=self.config.temperature, max_tokens=1, logprobs=10)
        outputs = self.model.generate(prompts, sampling_params, prompt_token_ids=prompt_token_ids, use_tqdm=False)

        results = []
        for output in outputs:
            token_text = output.outputs[0].text
            token_id = output.outputs[0].token_ids[0]
            logprobs = output.outputs[0].logprobs[0] if output.outputs[0].logprobs else {0: 0.0}
            results.append((token_text, token_id, logprobs))
        return results
