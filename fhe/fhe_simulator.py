"""
FHE Multiplicative Depth Tracker and Latency Simulator.

Tracks FHE multiplicative depth through network layers,
estimates inference latency, and simulates bootstrapping placement.

Based on CKKS parameters from Section 4 of the paper:
  - Ring degree: 2^16
  - Hamming weight: 192
  - Default modulus: 46 bits
  - Special modulus: 51 bits
  - Security: 128 bits
"""

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class OperationCost:
    """
    Latency (in seconds) for FHE operations.
    Based on Table 2 and paper's Amazon AWS r5.24xlarge benchmarks.
    """
    # Latency ratios from paper (Table 2, CryptoFaceNet4 analysis)
    conv_per_layer: float = 66.0       # seconds per conv layer at high level
    bn_per_layer: float = 1.0          # seconds per BN layer
    avgpool_per_block: float = 6.5     # seconds for pooling
    linear_per_layer: float = 27.7     # seconds per linear
    activation_per_block: float = 1.3  # seconds per HerPN block
    l2_norm: float = 2.0               # seconds for polynomial L2 norm
    match: float = 0.3                 # seconds for match computation
    bootstrap_per_op: float = 141.0    # seconds per bootstrap (1 boot total)
    parallel_overhead_ratio: float = 0.0326  # 3.26% overhead for parallelism


@dataclass
class LayerDepthInfo:
    """Tracks FHE depth through each layer."""
    layer_name: str
    layer_type: str
    input_level: int
    output_level: int
    depth_consumed: int
    requires_bootstrap: bool = False


class FHEDepthTracker:
    """
    Tracks multiplicative depth through CryptoFaceNet layers.

    In CKKS FHE:
    - Each ciphertext starts at level L
    - Each multiplication consumes 1 level
    - When level hits 0, bootstrapping is needed (resets to L_eff)
    - Bootstrapping consumes L_boot levels

    FHE level costs:
      Conv2d:        2 levels (includes rescaling)
      BatchNorm:     0 levels (fused with conv in FHE; add + scale = free)
      HerPN(fused):  1 level  (x^2 + bx + c, 'a' absorbed into conv)
      HerPN(full):   2 levels (ax^2 + bx + c)
      AvgPool:       0 levels (pure addition)
      Linear:        0 levels (rotation + add, no multiplication)
      L2 Norm poly:  2 levels (degree-2 polynomial)
      Add (residual): 0 levels
    """

    LEVEL_COSTS = {
        'conv2d': 2,
        'batchnorm': 0,      # fused with conv
        'herp_fused': 1,     # CryptoFaceNet optimization
        'herp_full': 2,      # standard AESPA
        'relu': 14,          # Chebyshev composite [15, 15, 27]
        'silu': 7,           # degree-63 Chebyshev
        'avgpool': 0,
        'linear': 0,         # rotation-based, no mult
        'l2norm_poly': 2,    # distribution-aware polynomial
        'add': 0,
        'bootstrap': 0,      # resets level (costs L_boot levels but resets)
    }

    def __init__(self, initial_level: int = 30, l_eff: int = 10,
                 l_boot: int = 15):
        """
        Args:
            initial_level: Starting level L of fresh ciphertext
            l_eff:         Effective level after bootstrapping (L - L_boot)
            l_boot:        Levels consumed by bootstrapping
        """
        self.initial_level = initial_level
        self.l_eff = l_eff
        self.l_boot = l_boot
        self.current_level = initial_level
        self.bootstrap_count = 0
        self.layer_history: List[LayerDepthInfo] = []
        self.total_depth = 0

    def reset(self):
        self.current_level = self.initial_level
        self.bootstrap_count = 0
        self.layer_history = []
        self.total_depth = 0

    def consume(self, layer_name: str, layer_type: str,
                force_bootstrap: bool = False) -> bool:
        """
        Consume levels for a layer, bootstrap if needed.

        Args:
            layer_name: Human-readable layer name
            layer_type: Type from LEVEL_COSTS keys
            force_bootstrap: Force bootstrap before this layer

        Returns:
            bootstrapped: Whether bootstrapping was triggered
        """
        cost = self.LEVEL_COSTS.get(layer_type, 0)
        bootstrapped = False

        # Bootstrap if level insufficient or forced
        if force_bootstrap or (cost > 0 and self.current_level < cost):
            self.bootstrap_count += 1
            bootstrapped = True
            old_level = self.current_level
            self.current_level = self.l_eff
            info = LayerDepthInfo(
                layer_name=f'[Bootstrap #{self.bootstrap_count}]',
                layer_type='bootstrap',
                input_level=old_level,
                output_level=self.current_level,
                depth_consumed=0,
                requires_bootstrap=True,
            )
            self.layer_history.append(info)

        # Record layer
        input_level = self.current_level
        self.current_level -= cost
        self.total_depth += cost

        info = LayerDepthInfo(
            layer_name=layer_name,
            layer_type=layer_type,
            input_level=input_level,
            output_level=self.current_level,
            depth_consumed=cost,
            requires_bootstrap=bootstrapped,
        )
        self.layer_history.append(info)

        return bootstrapped

    def simulate_cryptofacenet_pcnn(self, n_blocks: int = 5) -> int:
        """
        Simulate depth consumption for one PCNN in CryptoFaceNet.

        Block topology (CryptoFaceNet block, depth=6 each):
          HerPN(1) → Conv(2) → [residual BN] → HerPN(1) → Conv(2) → Add(0)

        Plus head:
          HerPN(1) → AvgPool(0) → Linear(0) → BN(0)

        Returns:
            Number of bootstraps needed
        """
        self.reset()

        # Stem conv
        self.consume('stem_conv', 'conv2d')

        # Backbone blocks
        for i in range(n_blocks):
            stride = 2 if i in [1, 3] else 1
            block_name = f'block{i+1}'

            self.consume(f'{block_name}.act1', 'herp_fused')
            self.consume(f'{block_name}.conv1', 'conv2d')
            self.consume(f'{block_name}.act2', 'herp_fused')
            self.consume(f'{block_name}.conv2', 'conv2d')
            self.consume(f'{block_name}.add', 'add')

        # Head
        self.consume('head_act', 'herp_fused')
        self.consume('head_pool', 'avgpool')
        self.consume('head_linear', 'linear')

        return self.bootstrap_count

    def print_depth_summary(self):
        """Print depth consumption table."""
        print(f'\n{"FHE Depth Analysis":=^60}')
        print(f'{"Layer":<35} {"Level In":>8} {"Cost":>6} {"Level Out":>9}')
        print('-' * 60)

        for info in self.layer_history:
            marker = ' ← BOOTSTRAP' if info.requires_bootstrap else ''
            print(f'{info.layer_name:<35} {info.input_level:>8} '
                  f'{info.depth_consumed:>6} {info.output_level:>9}{marker}')

        print('=' * 60)
        print(f'Total depth consumed: {self.total_depth}')
        print(f'Bootstrapping operations: {self.bootstrap_count}')
        print(f'Remaining level: {self.current_level}')


class LatencySimulator:
    """
    Simulates FHE inference latency based on paper's Table 2.

    Estimates latency for CryptoFaceNet vs baselines.

    Note: Real latency requires actual FHE library (SEAL/OpenFHE with C++).
    This is a simulation based on paper's reported breakdown.
    """

    # Reported latencies from Table 2 (seconds) for 64×64 images
    REPORTED_LATENCIES = {
        'MPCNN_ResNet32':  {'total': 7367, 'boot_pct': 67.76},
        'MPCNN_ResNet44':  {'total': 9845, 'boot_pct': 70.27},
        'AutoFHE_ResNet32': {'total': 4001, 'boot_pct': 31.84},
        'CryptoFaceNet4':  {'total': 1364, 'boot_pct': 10.34},
    }

    def __init__(self, costs: OperationCost = None):
        self.costs = costs or OperationCost()

    def estimate_cryptofacenet_latency(self, n_patches: int,
                                       resolution: int = 64) -> Dict[str, float]:
        """
        Estimate total FHE latency for CryptoFaceNet.

        Computation model:
        - L PCNNs run in PARALLEL → latency ≈ single PCNN latency
        - Fusion (add) is cheap
        - Only 1 bootstrap needed

        Args:
            n_patches: Number of patches (4, 9, or 16)
            resolution: Input resolution

        Returns:
            Dict with latency breakdown
        """
        c = self.costs
        n_blocks = 5

        # Per-PCNN latency (runs in parallel)
        conv_latency = n_blocks * 2 * c.conv_per_layer  # 2 convs per block
        activation_latency = (n_blocks * 2 + 1) * c.activation_per_block  # HerPN
        bn_latency = n_blocks * c.bn_per_layer
        pool_latency = c.avgpool_per_block
        linear_latency = c.linear_per_layer

        # FHE-specific
        bootstrap_latency = c.bootstrap_per_op  # only 1!
        l2_norm_latency = c.l2_norm
        match_latency = c.match

        # Parallel overhead (3.26% from paper)
        parallel_overhead = (conv_latency + activation_latency) * c.parallel_overhead_ratio

        total = (conv_latency + activation_latency + bn_latency +
                 pool_latency + linear_latency + bootstrap_latency +
                 l2_norm_latency + match_latency + parallel_overhead)

        return {
            'Conv': conv_latency,
            'BN': bn_latency,
            'Residual': 0.1,
            'AvgPool': pool_latency,
            'Linear': linear_latency,
            'Activation': activation_latency,
            'L2Norm': l2_norm_latency,
            'Match': match_latency,
            'Bootstrapping': bootstrap_latency,
            'Other': parallel_overhead,
            'Total': total,
            'n_patches': n_patches,
            'resolution': resolution,
        }

    def speedup_vs_baseline(self, our_latency: float,
                             baseline: str = 'MPCNN_ResNet44') -> float:
        """Compute speedup over a baseline."""
        baseline_lat = self.REPORTED_LATENCIES[baseline]['total']
        return baseline_lat / our_latency

    def print_latency_table(self, results: Dict[str, Dict]):
        """Print Table 2 style latency breakdown."""
        print(f'\n{"Latency Breakdown (Table 2 style)":=^80}')
        headers = ['Method', 'Conv', 'BN', 'AvgPool', 'Linear',
                   'Activation', 'Bootstrap', 'Total(s)']
        print(f'{"Method":<20}', end='')
        for h in headers[1:]:
            print(f' {h:>12}', end='')
        print()
        print('-' * 80)

        for name, r in results.items():
            print(f'{name:<20}', end='')
            for key in ['Conv', 'BN', 'AvgPool', 'Linear',
                        'Activation', 'Bootstrapping', 'Total']:
                v = r.get(key, 0)
                print(f' {v:>12.1f}', end='')
            print()


def analyze_cryptofacenet_depth() -> Dict[str, int]:
    """
    Analyze multiplicative depth of all CryptoFaceNet variants.
    Returns depth per PCNN and total bootstrap count.
    """
    tracker = FHEDepthTracker(initial_level=30, l_eff=10)
    n_boots = tracker.simulate_cryptofacenet_pcnn(n_blocks=5)
    tracker.print_depth_summary()

    return {
        'total_depth': tracker.total_depth,
        'n_bootstraps': n_boots,
        'remaining_level': tracker.current_level,
    }
