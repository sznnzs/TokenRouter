"""Experiment orchestration for TokenRouter."""
from __future__ import annotations

import csv
import json
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Sequence

import numpy as np

from .config import DE_CASCADE_BATCH_SIZE, GLOBAL_MAX_TOKENS, InferenceState, ModelConfig
from .datasets import build_math_prompt
from .routing import FixedThresholdRouter, PeriodicRouter, RandomRouter, TokenRouter
from .utils import calculate_shannon_entropy, extract_answer, is_correct_answer

if TYPE_CHECKING:
    from .models import VLLMModel


class TokenRouterExperimentSuite:
    """Runs model-only baselines, routing baselines, and TokenRouter experiments."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.results: Dict[str, Dict[str, Dict]] = {}
        self.llm: VLLMModel | None = None
        self.slm: VLLMModel | None = None

    def initialize_models(self) -> None:
        """Initialize the paired LLM and SLM."""
        from .models import VLLMModel

        print("Initializing models with vLLM...")
        self.llm = VLLMModel(
            self.config.llm_path,
            self.config,
            self.config.llm_gpu_ids,
            self.config.llm_tensor_parallel_size,
            self.config.llm_gpu_memory_utilization,
        )
        self.slm = VLLMModel(
            self.config.slm_path,
            self.config,
            self.config.slm_gpu_id,
            self.config.slm_tensor_parallel_size,
            self.config.slm_gpu_memory_utilization,
        )

    @staticmethod
    def default_router_config(dataset_name: str, target_llm_ratio: float) -> Dict:
        """Return the default TokenRouter hyperparameters used in this release."""
        return {
            "target_llm_ratio": target_llm_ratio,
            "kalman_process_var": 0.1,
            "kalman_measurement_var": 0.5,
            "ewma_lambda": 0.05,
            "beta_gain": 1.0,
            "k_base": 20 if dataset_name == "GSM8K" else 50,
            "alpha_severity": 2.0,
            "L_min": 1.0,
            "L_max": 4.0,
            "initial_L": 2.0,
        }

    def run_model_only(self, dataset_name: str, problems: List[Dict], model_type: str) -> None:
        """Run a pure LLM or pure SLM baseline."""
        assert self.llm is not None and self.slm is not None
        method_name = f"{model_type}-Only"
        model = self.llm if model_type == "LLM" else self.slm
        prompts = [build_math_prompt(problem["question"]) for problem in problems]

        current = self._init_result_bucket(dataset_name, method_name, total=len(problems))
        start_time = time.time()
        responses = model.generate(prompts, max_tokens=GLOBAL_MAX_TOKENS)
        total_wall_time = time.time() - start_time
        avg_time = total_wall_time / len(problems) if problems else 0.0

        for problem, prompt_text, response in zip(problems, prompts, responses):
            predicted_answer = extract_answer(response)
            is_correct = is_correct_answer(predicted_answer, problem["answer"])
            if is_correct:
                current["correct"] += 1

            token_count = len(model.tokenizer.encode(prompt_text + response))
            metrics = {
                "total_tokens": token_count,
                "llm_tokens": token_count if model_type == "LLM" else 0,
                "slm_tokens": token_count if model_type == "SLM" else 0,
                "llm_interventions": 0,
                "entropy_peaks": 0,
                "wall_time": avg_time,
                "entropy_history": [],
            }
            current["metrics"].append(metrics)
            current["predictions"].append(
                {
                    "question": problem["question"],
                    "predicted": response,
                    "predicted_answer": predicted_answer,
                    "ground_truth": problem["answer"],
                    "correct": is_correct,
                }
            )

    def run_policy(
        self,
        dataset_name: str,
        problems: List[Dict],
        policy,
        method_name: str,
        batch_size: int = DE_CASCADE_BATCH_SIZE,
    ) -> None:
        """Run a token-level routing policy over a dataset."""
        assert self.llm is not None and self.slm is not None
        print(f"\nRunning {method_name} on {dataset_name}\n{'=' * 50}")
        current = self._init_result_bucket(dataset_name, method_name)
        start_time = time.time()

        total_batches = (len(problems) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(problems))
            batch_problems = problems[batch_start:batch_end]
            print(f"\nProcessing batch {batch_idx + 1}/{total_batches} ({len(batch_problems)} problems)")

            active_states = []
            for problem in batch_problems:
                prompt_text = build_math_prompt(problem["question"])
                prompt_token_ids = self.slm.tokenizer.encode(prompt_text)
                if isinstance(policy, TokenRouter):
                    state = policy.create_state(
                        request_id=str(uuid.uuid4()),
                        problem=problem,
                        prompt=prompt_text,
                        initial_token_ids=prompt_token_ids,
                    )
                else:
                    state = InferenceState(
                        request_id=str(uuid.uuid4()),
                        problem=problem,
                        prompt=prompt_text,
                        token_ids=prompt_token_ids,
                    )
                    state.metrics["slm_tokens"] = len(prompt_token_ids)
                state.current_model = "slm"
                active_states.append(state)

            iteration = 0
            while active_states:
                iteration += 1
                if iteration > GLOBAL_MAX_TOKENS:
                    print("Reached the global max token limit for this batch.")
                    break

                llm_states = [state for state in active_states if state.current_model == "llm"]
                slm_states = [state for state in active_states if state.current_model == "slm"]

                if llm_states:
                    self._advance_states(llm_states, self.llm, policy, use_llm=True)
                if slm_states:
                    self._advance_states(slm_states, self.slm, policy, use_llm=False)

                remaining_states = []
                for state in active_states:
                    total_tokens = state.metrics["llm_tokens"] + state.metrics["slm_tokens"]
                    if state.is_finished or total_tokens >= GLOBAL_MAX_TOKENS:
                        self._finalize_state(current, state, total_tokens)
                    else:
                        remaining_states.append(state)
                active_states = remaining_states

        total_wall_time = time.time() - start_time
        avg_time = total_wall_time / len(problems) if problems else 0.0
        for metric_record in current["metrics"]:
            metric_record["wall_time"] = avg_time

        accuracy = current["correct"] / current["total"] if current["total"] else 0.0
        llm_ratio = self._average_llm_ratio(current["metrics"])
        print(
            f"\n{method_name} finished in {total_wall_time:.2f}s. "
            f"Accuracy={accuracy:.2%}, LLM ratio={llm_ratio:.2%}"
        )

    def run_tokenrouter(
        self,
        dataset_name: str,
        problems: List[Dict],
        target_ratios: Sequence[float],
    ) -> None:
        """Run TokenRouter at one or more target budgets."""
        for ratio in target_ratios:
            router = TokenRouter(**self.default_router_config(dataset_name, ratio))
            method_name = f"TokenRouter-R{int(ratio * 100)}"
            self.run_policy(dataset_name, problems, router, method_name)

    def run_baselines(
        self,
        dataset_name: str,
        problems: List[Dict],
        include_model_only: bool = True,
        random_probabilities: Sequence[float] = (0.5, 0.6, 0.7),
        fixed_thresholds: Sequence[float] = (2.0, 3.0),
        periodic_steps: Sequence[int] = (5, 10),
    ) -> None:
        """Run lightweight baselines used for comparison."""
        if include_model_only:
            self.run_model_only(dataset_name, problems, "LLM")
            self.run_model_only(dataset_name, problems, "SLM")
        for probability in random_probabilities:
            self.run_policy(
                dataset_name,
                problems,
                RandomRouter(llm_probability=probability),
                f"Random-P{int(probability * 100)}",
            )
        for threshold in fixed_thresholds:
            self.run_policy(
                dataset_name,
                problems,
                FixedThresholdRouter(threshold=threshold),
                f"FixedThreshold-T{threshold}",
            )
        for period in periodic_steps:
            self.run_policy(
                dataset_name,
                problems,
                PeriodicRouter(period=period),
                f"Periodic-N{period}",
            )

    def run_ablation(self, dataset_name: str, problems: List[Dict]) -> None:
        """Run the main TokenRouter ablations."""
        base_ratio = 0.7 if dataset_name == "GSM8K" else 0.8
        base_config = self.default_router_config(dataset_name, base_ratio)
        ablations = {
            "TokenRouter-Full": TokenRouter(**base_config),
            "TokenRouter-NoKalman": TokenRouter(
                **{**base_config, "kalman_process_var": 1e10, "kalman_measurement_var": 1e-10}
            ),
            "TokenRouter-NoAdaptive": TokenRouter(**{**base_config, "beta_gain": 0.0}),
            "TokenRouter-NoCommitment": TokenRouter(
                **{**base_config, "k_base": 1, "alpha_severity": 0.0}
            ),
        }
        for method_name, router in ablations.items():
            self.run_policy(dataset_name, problems, router, method_name)

    def save_results_bundle(self, output_dir: str | os.PathLike[str]) -> Dict[str, str]:
        """Save raw results, aggregate analysis, and a summary CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_path = output_dir / "results.json"
        analysis_path = output_dir / "results_analysis.json"
        summary_path = output_dir / "results_summary.csv"

        with raw_path.open("w", encoding="utf-8") as file_obj:
            json.dump(self.results, file_obj, indent=2, ensure_ascii=False)

        analysis = self.build_analysis()
        with analysis_path.open("w", encoding="utf-8") as file_obj:
            json.dump(analysis, file_obj, indent=2)

        with summary_path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(
                [
                    "Dataset",
                    "Method",
                    "Accuracy",
                    "LLM_Token_Ratio",
                    "Avg_Total_Tokens",
                    "Avg_LLM_Interventions",
                    "Avg_Wall_Time",
                ]
            )
            for dataset_name, methods in analysis.items():
                for method_name, stats in methods.items():
                    writer.writerow(
                        [
                            dataset_name,
                            method_name,
                            f"{stats['accuracy']:.4f}",
                            f"{stats['llm_token_ratio']:.4f}",
                            f"{stats['avg_total_tokens']:.1f}",
                            f"{stats['avg_llm_interventions']:.2f}",
                            f"{stats['avg_wall_time']:.2f}",
                        ]
                    )

        return {
            "results": str(raw_path),
            "analysis": str(analysis_path),
            "summary": str(summary_path),
        }

    def build_analysis(self) -> Dict[str, Dict[str, Dict]]:
        """Aggregate experiment metrics into a compact summary."""
        analysis: Dict[str, Dict[str, Dict]] = {}
        for dataset_name, methods in self.results.items():
            analysis[dataset_name] = {}
            for method_name, data in methods.items():
                if not data["metrics"] or data["total"] == 0:
                    continue
                metrics = data["metrics"]
                analysis[dataset_name][method_name] = {
                    "accuracy": data["correct"] / data["total"],
                    "total_problems": data["total"],
                    "avg_total_tokens": float(np.mean([m["total_tokens"] for m in metrics])),
                    "avg_llm_tokens": float(np.mean([m["llm_tokens"] for m in metrics])),
                    "avg_slm_tokens": float(np.mean([m["slm_tokens"] for m in metrics])),
                    "llm_token_ratio": self._average_llm_ratio(metrics),
                    "avg_llm_interventions": float(
                        np.mean([m.get("llm_interventions", 0) for m in metrics])
                    ),
                    "avg_wall_time": float(np.mean([m["wall_time"] for m in metrics])),
                }
        return analysis

    def print_summary(self) -> None:
        """Print a concise summary of all completed runs."""
        analysis = self.build_analysis()
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        for dataset_name, methods in analysis.items():
            print(f"\n{dataset_name}")
            print("-" * len(dataset_name))
            for method_name, stats in methods.items():
                print(
                    f"{method_name}: acc={stats['accuracy']:.2%}, "
                    f"llm_ratio={stats['llm_token_ratio']:.2%}, "
                    f"tokens={stats['avg_total_tokens']:.1f}, "
                    f"interventions={stats['avg_llm_interventions']:.2f}"
                )

    def _advance_states(self, states: List[InferenceState], model: VLLMModel, policy, use_llm: bool) -> None:
        prompts = [state.prompt + state.full_generation for state in states]
        generation_results = model.generate_one_token(prompts=prompts)
        peer_model = self.slm if use_llm else self.llm
        assert peer_model is not None

        for state, (token, token_id, logprobs) in zip(states, generation_results):
            if token_id in [model.eos_token_id, peer_model.eos_token_id]:
                state.is_finished = True
                continue

            if use_llm:
                state.full_generation += token
                state.metrics["llm_tokens"] += 1
                entropy = calculate_shannon_entropy(logprobs, model.vocab_size)
            else:
                state.full_generation += token
                state.metrics["slm_tokens"] += 1
                entropy = calculate_shannon_entropy(logprobs, model.vocab_size)

            state.metrics["entropy_history"].append(entropy)
            state.current_model = policy.make_routing_decision(state, entropy)

    def _finalize_state(self, current: Dict, state: InferenceState, total_tokens: int) -> None:
        predicted_answer = extract_answer(state.full_generation)
        is_correct = is_correct_answer(predicted_answer, state.problem["answer"])
        current["total"] += 1
        if is_correct:
            current["correct"] += 1

        state.metrics["total_tokens"] = total_tokens
        current["metrics"].append(state.metrics)
        current["predictions"].append(
            {
                "question": state.problem["question"],
                "predicted": state.full_generation,
                "predicted_answer": predicted_answer,
                "ground_truth": state.problem["answer"],
                "correct": is_correct,
            }
        )

    def _init_result_bucket(self, dataset_name: str, method_name: str, total: int = 0) -> Dict:
        self.results.setdefault(dataset_name, {})
        bucket = {"correct": 0, "total": total, "metrics": [], "predictions": []}
        self.results[dataset_name][method_name] = bucket
        return bucket

    @staticmethod
    def _average_llm_ratio(metrics: List[Dict]) -> float:
        if not metrics:
            return 0.0
        ratios = []
        for metric in metrics:
            total_tokens = metric.get("total_tokens", 0)
            if total_tokens > 0:
                ratios.append(metric.get("llm_tokens", 0) / total_tokens)
        return float(np.mean(ratios)) if ratios else 0.0
