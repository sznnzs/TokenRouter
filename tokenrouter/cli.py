"""Command-line entry points for TokenRouter."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .benchmark import TokenRouterExperimentSuite
from .config import ModelConfig
from .datasets import DatasetManager
from .plotting import main as plot_results_main


DEFAULT_TARGET_RATIOS = {
    "GSM8K": [0.7],
    "MATH": [0.8],
    "AIME": [0.8],
}


def parse_run_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TokenRouter experiments.")
    parser.add_argument("--dataset", choices=["GSM8K", "MATH", "AIME", "all"], default="GSM8K")
    parser.add_argument(
        "--mode",
        choices=["main", "baselines", "ablation", "all"],
        default="main",
        help="Which experiments to execute.",
    )
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--llm_path", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    parser.add_argument("--slm_path", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--llm_gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--slm_gpu", type=int, default=0)
    parser.add_argument("--target_ratios", type=float, nargs="+", default=None)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory. Defaults to results/<timestamp>/.",
    )
    return parser.parse_args()


def load_datasets(data_manager: DatasetManager, dataset_name: str, num_samples: int | None) -> dict:
    if dataset_name == "all":
        return {
            "GSM8K": data_manager.load_GSM8K(num_samples),
            "MATH": data_manager.load_MATH(num_samples),
            "AIME": data_manager.load_AIME(num_samples),
        }
    if dataset_name == "GSM8K":
        return {"GSM8K": data_manager.load_GSM8K(num_samples)}
    if dataset_name == "MATH":
        return {"MATH": data_manager.load_MATH(num_samples)}
    return {"AIME": data_manager.load_AIME(num_samples)}


def run_experiments_main() -> None:
    args = parse_run_args()
    config = ModelConfig(
        llm_path=args.llm_path,
        slm_path=args.slm_path,
        llm_gpu_ids=args.llm_gpus,
        slm_gpu_id=args.slm_gpu,
        llm_tensor_parallel_size=len(args.llm_gpus),
        slm_tensor_parallel_size=1,
    )

    suite = TokenRouterExperimentSuite(config)
    suite.initialize_models()
    datasets = load_datasets(DatasetManager(), args.dataset, args.num_samples)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"results/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, problems in datasets.items():
        print(f"\n{'=' * 72}\nDataset: {dataset_name}\n{'=' * 72}")
        target_ratios = args.target_ratios or DEFAULT_TARGET_RATIOS[dataset_name]

        if args.mode in {"main", "all"}:
            suite.run_tokenrouter(dataset_name, problems, target_ratios)
        if args.mode in {"baselines", "all"}:
            suite.run_baselines(dataset_name, problems)
        if args.mode in {"ablation", "all"}:
            suite.run_ablation(dataset_name, problems)

    saved_paths = suite.save_results_bundle(output_dir)
    suite.print_summary()
    print("\nSaved files:")
    for key, path in saved_paths.items():
        print(f"  {key}: {path}")


__all__ = ["plot_results_main", "run_experiments_main"]
