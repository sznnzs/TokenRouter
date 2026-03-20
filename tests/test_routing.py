"""Lightweight unit tests for routing components."""
from tokenrouter.benchmark import TokenRouterExperimentSuite
from tokenrouter.config import ModelConfig
from tokenrouter.routing import KalmanFilter1D, TokenRouter


def test_kalman_filter_returns_finite_values():
    kalman = KalmanFilter1D(process_variance=0.1, measurement_variance=0.5)
    outputs = [kalman.filter(value) for value in [0.2, 0.4, 0.9, 0.3]]
    assert len(outputs) == 4
    assert all(output == output for output in outputs)


def test_commitment_duration_grows_with_severity():
    router = TokenRouter(k_base=5, alpha_severity=2.0)
    mild = router.compute_commitment_duration(z_score=2.5, threshold=2.0)
    severe = router.compute_commitment_duration(z_score=4.5, threshold=2.0)
    assert mild >= 5
    assert severe > mild


def test_default_config_matches_dataset_scale():
    gsm8k = TokenRouterExperimentSuite.default_router_config("GSM8K", 0.7)
    aime = TokenRouterExperimentSuite.default_router_config("AIME", 0.8)
    assert gsm8k["k_base"] == 20
    assert aime["k_base"] == 50
    assert gsm8k["ewma_lambda"] == 0.05
    assert aime["alpha_severity"] == 2.0


def test_model_config_uses_public_defaults():
    config = ModelConfig()
    assert "DeepSeek-R1-Distill-Qwen" in config.llm_path
    assert "DeepSeek-R1-Distill-Qwen" in config.slm_path
