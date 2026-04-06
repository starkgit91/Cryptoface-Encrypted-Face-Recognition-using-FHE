#!/usr/bin/env python3
"""
Generate all paper figures and tables.

Usage:
  python scripts/plot_results.py --output results/
  python scripts/plot_results.py --checkpoint checkpoints/best.pth
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='./results',
                        help='Output directory for figures')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Optional model checkpoint for custom eval results')
    parser.add_argument('--results_json', type=str, default=None,
                        help='Optional JSON file with evaluation results')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load results if provided
    eval_results = None
    if args.results_json and os.path.exists(args.results_json):
        with open(args.results_json) as f:
            eval_results = json.load(f)

    from utils.visualization import generate_all_figures
    generate_all_figures(output_dir=args.output, eval_results=eval_results)

    print(f'\nAll figures saved to: {args.output}')
    print('\nGenerated figures:')
    for f in sorted(os.listdir(args.output)):
        if f.endswith('.png'):
            print(f'  {f}')


if __name__ == '__main__':
    main()
