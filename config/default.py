"""
Default configuration for CryptoFace.
All hyperparameters from the paper are documented here.
"""

import argparse


def get_config():
    parser = argparse.ArgumentParser(description='CryptoFace Configuration')

    # ── Dataset ──────────────────────────────────────────────────────────────
    parser.add_argument('--data_root', type=str, default='./data/webface4m',
                        help='Root directory of WebFace4M training data')
    parser.add_argument('--test_data_root', type=str, default='./data/eval',
                        help='Root directory of test datasets')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of DataLoader workers')

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument('--resolution', type=int, default=64,
                        choices=[64, 96, 128],
                        help='Input face image resolution. '
                             '64→CryptoFaceNet4, 96→CryptoFaceNet9, '
                             '128→CryptoFaceNet16')
    parser.add_argument('--patch_size', type=int, default=32,
                        help='Patch size (P×P). Default 32 from paper.')
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='Feature embedding dimension (256 from paper)')
    parser.add_argument('--num_blocks', type=int, default=5,
                        help='Number of convolutional blocks per PCNN')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base channel count')
    parser.add_argument('--use_herp_activation', action='store_true',
                        default=True,
                        help='Use HerPN (Hermite polynomial) activation')

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training (256 from paper)')
    parser.add_argument('--epochs', type=int, default=26,
                        help='Total training epochs (26 from paper)')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Initial learning rate (0.05 from paper)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (0.9 from paper)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (0.0005 from paper)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm (1.0 from paper)')
    parser.add_argument('--lr_steps', type=int, nargs='+',
                        default=[12, 20, 24],
                        help='Epoch milestones for LR decay (×0.1)')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='LR decay factor at each milestone')

    # ── ArcFace Loss ──────────────────────────────────────────────────────────
    parser.add_argument('--arcface_s', type=float, default=64.0,
                        help='ArcFace scale factor')
    parser.add_argument('--arcface_m', type=float, default=0.5,
                        help='ArcFace margin (additive angular margin)')

    # ── Jigsaw Loss ───────────────────────────────────────────────────────────
    parser.add_argument('--jigsaw_alpha', type=float, default=0.005,
                        help='Jigsaw puzzle loss weight (α=0.005 from paper)')

    # ── Evaluation ────────────────────────────────────────────────────────────
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['lfw', 'agedb', 'calfw', 'cplfw', 'cfpfp'],
                        help='Test datasets to evaluate on')
    parser.add_argument('--eval_batch_size', type=int, default=512,
                        help='Batch size for evaluation')
    parser.add_argument('--num_folds', type=int, default=10,
                        help='Number of cross-validation folds')

    # ── FHE Simulation ────────────────────────────────────────────────────────
    parser.add_argument('--ring_degree_log2', type=int, default=16,
                        help='log2 of CKKS ring degree (2^16 from paper)')
    parser.add_argument('--hamming_weight', type=int, default=192,
                        help='Hamming weight for CKKS key (192 from paper)')
    parser.add_argument('--default_modulus_bits', type=int, default=46,
                        help='Default modulus bits (46 from paper)')
    parser.add_argument('--special_modulus_bits', type=int, default=51,
                        help='Special modulus bits (51 from paper)')

    # ── I/O ───────────────────────────────────────────────────────────────────
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to load')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='TensorBoard log directory')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_freq', type=int, default=100,
                        help='Log training stats every N iterations')

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, cuda:0, ...')
    parser.add_argument('--amp', action='store_true',
                        help='Use Automatic Mixed Precision')

    return parser


# ── Paper hyper-parameter constants (do not change) ─────────────────────────
PAPER_PARAMS = {
    # Training
    'batch_size': 256,
    'epochs': 26,
    'lr': 0.05,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'grad_clip': 1.0,
    'lr_steps': [12, 20, 24],
    'lr_gamma': 0.1,
    'patch_size': 32,
    'feature_dim': 256,
    'jigsaw_alpha': 0.005,
    'arcface_s': 64.0,
    'arcface_m': 0.5,

    # FHE
    'ring_degree': 2 ** 16,
    'hamming_weight': 192,
    'default_modulus_bits': 46,
    'special_modulus_bits': 51,
    'security_bits': 128,

    # CryptoFaceNet variants
    'variants': {
        'CryptoFaceNet4':  {'resolution': 64,  'n_patches': 4,  'patch_size': 32},
        'CryptoFaceNet9':  {'resolution': 96,  'n_patches': 9,  'patch_size': 32},
        'CryptoFaceNet16': {'resolution': 128, 'n_patches': 16, 'patch_size': 32},
    },

    # Paper reported results (Table 1)
    'paper_results': {
        'MPCNN_ResNet32': {
            'LFW': 97.02, 'AgeDB': 83.02, 'CALFW': 87.00,
            'CPLFW': 78.90, 'CFP-FP': 82.07, 'Avg': 85.60,
            'Latency': 7367, 'RAM': '286G', '#Boot': 31,
        },
        'MPCNN_ResNet44': {
            'LFW': 98.27, 'AgeDB': 87.45, 'CALFW': 90.85,
            'CPLFW': 83.72, 'CFP-FP': 87.90, 'Avg': 89.64,
            'Latency': 9845, 'RAM': '286G', '#Boot': 43,
        },
        'AutoFHE_ResNet32': {
            'LFW': 93.53, 'AgeDB': 80.88, 'CALFW': 85.40,
            'CPLFW': 75.67, 'CFP-FP': 77.96, 'Avg': 82.69,
            'Latency': 4001, 'RAM': '286G', '#Boot': 8,
        },
        'CryptoFaceNet4': {
            'LFW': 98.87, 'AgeDB': 89.45, 'CALFW': 91.60,
            'CPLFW': 81.98, 'CFP-FP': 85.21, 'Avg': 89.42,
            'Latency': 1364, 'RAM': '269G', '#Boot': 1,
            'Params': '0.94M', 'Resolution': 64,
        },
        'CryptoFaceNet9': {
            'LFW': 99.18, 'AgeDB': 91.38, 'CALFW': 93.32,
            'CPLFW': 84.23, 'CFP-FP': 86.81, 'Avg': 90.99,
            'Latency': 1395, 'RAM': '276G', '#Boot': 1,
            'Params': '2.12M', 'Resolution': 96,
        },
        'CryptoFaceNet16': {
            'LFW': 98.78, 'AgeDB': 92.90, 'CALFW': 93.73,
            'CPLFW': 83.95, 'CFP-FP': 87.94, 'Avg': 91.46,
            'Latency': 1446, 'RAM': '277G', '#Boot': 1,
            'Params': '3.78M', 'Resolution': 128,
        },
    },

    # Latency breakdown (Table 2) for CryptoFaceNet4 on 64×64
    'latency_breakdown': {
        'MPCNN_ResNet32': {
            'Conv': (896, 12.17), 'BN': (28, 0.38), 'Residual': (0.5, 0.01),
            'AvgPool': (46, 0.62), 'Linear': (807, 10.96),
            'Activation': (583, 7.92), 'L2Norm': (5, 0.07),
            'Match': (2, 0.02), 'Bootstrapping': (4991, 67.76), 'Other': (7, 0.09),
            'Total': 7367,
        },
        'MPCNN_ResNet44': {
            'Conv': (1214, 12.33), 'BN': (39, 0.39), 'Residual': (0.7, 0.01),
            'AvgPool': (46, 0.46), 'Linear': (807, 8.19),
            'Activation': (807, 8.20), 'L2Norm': (5, 0.05),
            'Match': (2, 0.02), 'Bootstrapping': (6917, 70.27), 'Other': (7, 0.08),
            'Total': 9845,
        },
        'AutoFHE_ResNet32': {
            'Conv': (1966, 49.14), 'BN': (28, 0.69), 'Residual': (1.7, 0.04),
            'AvgPool': (38, 0.95), 'Linear': (658, 16.43),
            'Activation': (17, 0.43), 'L2Norm': (4, 0.10),
            'Match': (1, 0.03), 'Bootstrapping': (1274, 31.84), 'Other': (14, 0.34),
            'Total': 4001,
        },
        'CryptoFaceNet4': {
            'Conv': (858, 62.93), 'BN': (2, 0.16), 'Residual': (0.1, 0.01),
            'AvgPool': (26, 1.94), 'Linear': (277, 20.30),
            'Activation': (13, 0.94), 'L2Norm': (2, 0.11),
            'Match': (0.3, 0.02), 'Bootstrapping': (141, 10.34), 'Other': (44, 3.26),
            'Total': 1364,
        },
    },
}
