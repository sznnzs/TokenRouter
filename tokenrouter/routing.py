"""Routing policies used by TokenRouter experiments."""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .config import InferenceState

# ===================================================================
# Kalman Filter Implementation for Complexity Tracking
# ===================================================================

class KalmanFilter1D:
    """One-dimensional Kalman filter for latent complexity estimation"""
    def __init__(self, process_variance: float = 1e-4, measurement_variance: float = 0.5):
        # State estimate
        self.x = 0.0  # Initial state estimate
        self.P = 1.0  # Initial error covariance
        
        # Model parameters
        self.F = 1.0  # State transition (random walk)
        self.H = 1.0  # Observation model (direct observation)
        self.Q = process_variance  # Process noise variance
        self.R = measurement_variance  # Measurement noise variance
        
    def predict(self):
        """Prediction step"""
        self.x = self.F * self.x
        self.P = self.F * self.P * self.F + self.Q
        
    def update(self, z: float):
        """Update step with new measurement z"""
        # Kalman gain
        K = self.P * self.H / (self.H * self.P * self.H + self.R)
        
        # State update
        self.x = self.x + K * (z - self.H * self.x)
        
        # Covariance update
        self.P = (1 - K * self.H) * self.P
        
    def filter(self, measurement: float) -> float:
        """Complete filtering step: predict then update"""
        self.predict()
        self.update(measurement)
        return self.x

# ===================================================================
# EWMA Control Chart for Anomaly Detection
# ===================================================================

class EWMAControlChart:
    """Exponentially Weighted Moving Average control chart for anomaly detection"""
    def __init__(self, lambda_param: float = 0.2, initial_mean: float = 0.0, 
                 process_std: float = 1.0):
        self.lambda_param = lambda_param
        self.mu = initial_mean  # EWMA statistic
        self.process_std = process_std
        self.t = 0  # Time step counter
        
    def update(self, value: float) -> Tuple[float, float]:
        """Update EWMA statistic and return (statistic, std_dev)"""
        self.t += 1
        self.mu = self.lambda_param * value + (1 - self.lambda_param) * self.mu
        
        # EWMA variance (converges to steady state)
        if self.t > 30:  # Use steady-state formula after convergence
            variance = (self.lambda_param / (2 - self.lambda_param)) * (self.process_std ** 2)
        else:
            # Time-varying variance for initial observations
            variance = (self.lambda_param / (2 - self.lambda_param)) * \
                      (self.process_std ** 2) * (1 - (1 - self.lambda_param) ** (2 * self.t))
        
        std_dev = np.sqrt(variance)
        return self.mu, std_dev
    
    def get_z_score(self, value: float, prev_mu: float) -> float:
        """Calculate Z-score for anomaly detection"""
        _, std_dev = self.update(value)
        if std_dev < 1e-6:
            return 0.0
        return (value - prev_mu) / std_dev

# ===================================================================
# Complete TokenRouter Implementation
# ===================================================================

@dataclass
class TokenRouterState(InferenceState):
    """Extended state for TokenRouter with additional tracking."""
    # Kalman filter for complexity tracking
    kalman_filter: KalmanFilter1D = None
    # EWMA control chart
    ewma_chart: EWMAControlChart = None
    # Control parameters
    L_threshold: float = 3.0  # Adaptive threshold
    commitment_remaining: int = 0  # Remaining commitment steps
    # Budget tracking
    target_llm_ratio: float = 0.3
    # Entropy tracking
    filtered_complexity: List[float] = field(default_factory=list)
    z_scores: List[float] = field(default_factory=list)

class TokenRouter:
    """Complete implementation of the TokenRouter algorithm."""
    
    def __init__(self, 
                 # Signal processing parameters
                 kalman_process_var: float = 1e-4,
                 kalman_measurement_var: float = 0.5,
                 ewma_lambda: float = 0.2,
                 # Control policy parameters
                 target_llm_ratio: float = 0.3,
                 beta_gain: float = 0.01,
                 k_base: int = 3,
                 alpha_severity: float = 0.5,
                 L_min: float = 1.0,
                 L_max: float = 4.0,
                 # Initial values
                 initial_L: float = 3.0):
        
        # Store parameters
        self.kalman_process_var = kalman_process_var
        self.kalman_measurement_var = kalman_measurement_var
        self.ewma_lambda = ewma_lambda
        self.target_llm_ratio = target_llm_ratio
        self.beta_gain = beta_gain
        self.k_base = k_base
        self.alpha_severity = alpha_severity
        self.L_min = L_min
        self.L_max = L_max
        self.initial_L = initial_L
        
    def create_state(self, request_id: str, problem: Dict, prompt: str, 
                    initial_token_ids: List[int]) -> TokenRouterState:
        """Create a new TokenRouter state for a problem."""
        state = TokenRouterState(
            request_id=request_id,
            problem=problem,
            prompt=prompt,
            token_ids=initial_token_ids,
            kalman_filter=KalmanFilter1D(self.kalman_process_var, self.kalman_measurement_var),
            ewma_chart=EWMAControlChart(self.ewma_lambda),
            L_threshold=self.initial_L,
            target_llm_ratio=self.target_llm_ratio
        )
        state.metrics['slm_tokens'] = len(initial_token_ids)
        return state
    
    def process_entropy(self, state: TokenRouterState, entropy: float) -> Tuple[float, float]:
        """Process raw entropy through Kalman filtering and EWMA."""
        # Stage 1: Kalman filtering
        filtered_value = state.kalman_filter.filter(entropy)
        state.filtered_complexity.append(filtered_value)
        
        # Stage 2: EWMA anomaly detection
        prev_mu = state.ewma_chart.mu
        z_score = state.ewma_chart.get_z_score(filtered_value, prev_mu)
        state.z_scores.append(z_score)
        
        return filtered_value, z_score
    
    def update_threshold(self, state: TokenRouterState):
        """Self-calibrating threshold update based on budget."""
        total_tokens = state.metrics['llm_tokens'] + state.metrics['slm_tokens']
        if total_tokens == 0:
            return
            
        actual_ratio = state.metrics['llm_tokens'] / total_tokens
        error = actual_ratio - state.target_llm_ratio
        
        # Proportional control
        state.L_threshold += self.beta_gain * error
        state.L_threshold = np.clip(state.L_threshold, self.L_min, self.L_max)
    
    def compute_commitment_duration(self, z_score: float, threshold: float) -> int:
        """Compute intervention commitment duration based on anomaly severity"""
        severity = max(0, z_score - threshold)
        duration = max(self.k_base, int(np.ceil(self.k_base + self.alpha_severity * severity)))
        return duration
    
    def make_routing_decision(self, state: TokenRouterState, entropy: float) -> str:
        """Main routing decision logic."""
        # Check if we're in commitment mode
        if state.commitment_remaining > 0:
            state.commitment_remaining -= 1
            return 'llm'
        
        # Process entropy signal
        filtered_value, z_score = self.process_entropy(state, entropy)
        
        # Update adaptive threshold
        self.update_threshold(state)
        
        # Check for anomaly
        if z_score > state.L_threshold and state.current_model == 'slm':
            # Trigger LLM intervention
            commitment = self.compute_commitment_duration(z_score, state.L_threshold)
            state.commitment_remaining = commitment - 1  # -1 because current step uses LLM
            state.metrics['llm_interventions'] += 1
            state.metrics['entropy_peaks'] += 1
            return 'llm'
        
        return 'slm'

# ===================================================================
# Baseline Routing Strategies
# ===================================================================

class RandomRouter:
    """Random routing baseline with configurable probability"""
    def __init__(self, llm_probability: float = 0.5):
        self.llm_probability = llm_probability
        
    def make_routing_decision(self, state: InferenceState, entropy: float) -> str:
        return 'llm' if np.random.random() < self.llm_probability else 'slm'

class FixedThresholdRouter:
    """Fixed threshold routing without adaptation"""
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.ewma_chart = EWMAControlChart()
        
    def make_routing_decision(self, state: InferenceState, entropy: float) -> str:
        prev_mu = self.ewma_chart.mu
        z_score = self.ewma_chart.get_z_score(entropy, prev_mu)
        return 'llm' if z_score > self.threshold else 'slm'

class PeriodicRouter:
    """Periodic switching between models"""
    def __init__(self, period: int = 10):
        self.period = period
        self.counter = 0
        
    def make_routing_decision(self, state: InferenceState, entropy: float) -> str:
        self.counter += 1
        # Use LLM every 'period' tokens
        return 'llm' if (self.counter % self.period) == 0 else 'slm'

# ===================================================================
