#!/usr/bin/env python3
"""
Generate all paper result tables in the terminal and as CSV.
Reproduces Table 1, Table 2, Table 3 from the paper.
No PyTorch required.
"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path

OUT = Path('./results')
OUT.mkdir(exist_ok=True)

# ── Paper data ────────────────────────────────────────────────────────────────
PAPER_RESULTS = {
    'MPCNN (ResNet32)':   {'Params':'0.53M','#Boot':31,'Res':64,'LFW':97.02,'AgeDB':83.02,'CALFW':87.00,'CPLFW':78.90,'CFP-FP':82.07,'Avg':85.60,'Latency':7367,'RAM':'286G'},
    'MPCNN (ResNet44)':   {'Params':'0.72M','#Boot':43,'Res':64,'LFW':98.27,'AgeDB':87.45,'CALFW':90.85,'CPLFW':83.72,'CFP-FP':87.90,'Avg':89.64,'Latency':9845,'RAM':'286G'},
    'AutoFHE (ResNet32)': {'Params':'0.53M','#Boot':8, 'Res':64,'LFW':93.53,'AgeDB':80.88,'CALFW':85.40,'CPLFW':75.67,'CFP-FP':77.96,'Avg':82.69,'Latency':4001,'RAM':'286G'},
    'CryptoFaceNet4 (Ours)': {'Params':'0.94M','#Boot':1,'Res':64,'LFW':98.87,'AgeDB':89.45,'CALFW':91.60,'CPLFW':81.98,'CFP-FP':85.21,'Avg':89.42,'Latency':1364,'RAM':'269G'},
    'CryptoFaceNet9 (Ours)': {'Params':'2.12M','#Boot':1,'Res':96,'LFW':99.18,'AgeDB':91.38,'CALFW':93.32,'CPLFW':84.23,'CFP-FP':86.81,'Avg':90.99,'Latency':1395,'RAM':'276G'},
    'CryptoFaceNet16 (Ours)':{'Params':'3.78M','#Boot':1,'Res':128,'LFW':98.78,'AgeDB':92.90,'CALFW':93.73,'CPLFW':83.95,'CFP-FP':87.94,'Avg':91.46,'Latency':1446,'RAM':'277G'},
}

LATENCY_BREAKDOWN = {
    'MPCNN (ResNet32)':   {'Conv':(896,12.17),'BN':(28,0.38),'Residual':(0.5,0.01),'AvgPool':(46,0.62),'Linear':(807,10.96),'Activation':(583,7.92),'L2Norm':(5,0.07),'Match':(2,0.02),'Bootstrap':(4991,67.76),'Other':(7,0.09)},
    'MPCNN (ResNet44)':   {'Conv':(1214,12.33),'BN':(39,0.39),'Residual':(0.7,0.01),'AvgPool':(46,0.46),'Linear':(807,8.19),'Activation':(807,8.20),'L2Norm':(5,0.05),'Match':(2,0.02),'Bootstrap':(6917,70.27),'Other':(7,0.08)},
    'AutoFHE (ResNet32)': {'Conv':(1966,49.14),'BN':(28,0.69),'Residual':(1.7,0.04),'AvgPool':(38,0.95),'Linear':(658,16.43),'Activation':(17,0.43),'L2Norm':(4,0.10),'Match':(1,0.03),'Bootstrap':(1274,31.84),'Other':(14,0.34)},
    'CryptoFaceNet4 (Ours)':{'Conv':(858,62.93),'BN':(2,0.16),'Residual':(0.1,0.01),'AvgPool':(26,1.94),'Linear':(277,20.30),'Activation':(13,0.94),'L2Norm':(2,0.11),'Match':(0.3,0.02),'Bootstrap':(141,10.34),'Other':(44,3.26)},
}

IDENT_RESULTS = {
    'MPCNN':       {'Rank1':88.28,'Rank5':97.66},
    'CryptoFace':  {'Rank1':92.19,'Rank5':98.44},
}

IJB_RESULTS = {
    'IJB-B': {'AutoFHE':{'Time':4001,'AUC':0.825},'CryptoFace':{'Time':1364,'AUC':0.926}},
    'IJB-C': {'MPCNN':  {'Time':7367,'AUC':0.927},'CryptoFace':{'Time':1364,'AUC':0.946}},
}

SEP = '─' * 115

def print_table1():
    print('\n' + '═'*115)
    print('TABLE 1 – End-to-End Face Recognition on FHE')
    print('═'*115)
    hdr = f"{'Method':<28} {'Params':>7} {'#Boot':>5} {'Res':>4} {'LFW':>7} {'AgeDB':>7} {'CALFW':>7} {'CPLFW':>7} {'CFP-FP':>7} {'Avg':>7} {'Latency(s)':>11} {'RAM':>6}"
    print(hdr); print(SEP)
    rows = []
    for method, d in PAPER_RESULTS.items():
        marker = ' ◄' if 'Ours' in method else ''
        print(f"{method:<28} {d['Params']:>7} {d['#Boot']:>5} {d['Res']:>4} "
              f"{d['LFW']:>7.2f} {d['AgeDB']:>7.2f} {d['CALFW']:>7.2f} "
              f"{d['CPLFW']:>7.2f} {d['CFP-FP']:>7.2f} {d['Avg']:>7.2f} "
              f"{d['Latency']:>11,} {d['RAM']:>6}{marker}")
        rows.append([method]+[str(d[k]) for k in ['Params','#Boot','Res','LFW','AgeDB','CALFW','CPLFW','CFP-FP','Avg','Latency','RAM']])
    print(SEP)
    # Best per column
    ds_cols = ['LFW','AgeDB','CALFW','CPLFW','CFP-FP','Avg']
    best_vals = {c: max(d[c] for d in PAPER_RESULTS.values()) for c in ds_cols}
    print('\nBest values per benchmark:')
    for c, v in best_vals.items():
        winners = [m for m,d in PAPER_RESULTS.items() if d[c] == v]
        print(f"  {c:<8}: {v:.2f}%  ← {', '.join(winners)}")

    # Save CSV
    with open(OUT/'table1_main_results.csv','w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['Method','Params','#Boot','Res','LFW','AgeDB','CALFW','CPLFW','CFP-FP','Avg','Latency(s)','RAM'])
        w.writerows(rows)
    print(f'\n  ✓ Saved: {OUT}/table1_main_results.csv')

def print_table2():
    print('\n' + '═'*115)
    print('TABLE 2 – Latency Breakdown of FHE Operations (64×64 encrypted face images)')
    print('═'*115)
    comps = ['Conv','BN','Residual','AvgPool','Linear','Activation','L2Norm','Match','Bootstrap','Other']
    hdr = f"{'Method':<28}" + ''.join(f" {c[:8]:>9}" for c in comps) + f" {'Total':>9}"
    print(hdr); print(SEP)
    rows = []
    for method, d in LATENCY_BREAKDOWN.items():
        total = sum(v[0] for v in d.values())
        row = [method] + [f"{d[c][0]:.1f}s({d[c][1]:.1f}%)" for c in comps] + [f"{total:.0f}s"]
        line = f"{method:<28}" + ''.join(f" {d[c][0]:>6.1f}s" for c in comps) + f" {total:>7.0f}s"
        print(line)
        rows.append(row)
    print(SEP)
    print('\nKey insight: Bootstrapping dominates MPCNN (67-70%) but only 10.3% in CryptoFaceNet4.')

    with open(OUT/'table2_latency_breakdown.csv','w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['Method'] + comps + ['Total'])
        w.writerows(rows)
    print(f'  ✓ Saved: {OUT}/table2_latency_breakdown.csv')

def print_table3():
    print('\n' + '═'*60)
    print('TABLE 3 – 1:128 Closed-Set Retrieval (Face Identification)')
    print('═'*60)
    print(f"{'Method':<20} {'Rank-1 Acc':>12} {'Rank-5 Acc':>12}")
    print('─'*46)
    for method, d in IDENT_RESULTS.items():
        marker = ' ◄' if method == 'CryptoFace' else ''
        if method == 'CryptoFace':
            delta1 = d['Rank1'] - IDENT_RESULTS['MPCNN']['Rank1']
            delta5 = d['Rank5'] - IDENT_RESULTS['MPCNN']['Rank5']
            print(f"{method:<20} {d['Rank1']:>11.2f}% {d['Rank5']:>11.2f}%"
                  f"  (↑{delta1:.2f}%, ↑{delta5:.2f}%){marker}")
        else:
            print(f"{method:<20} {d['Rank1']:>11.2f}% {d['Rank5']:>11.2f}%")
    print('─'*46)

def print_ijb():
    print('\n' + '═'*60)
    print('FIGURE 7 – IJB-B and IJB-C Results')
    print('═'*60)
    for ds, methods in IJB_RESULTS.items():
        print(f'\n{ds}:')
        print(f"  {'Method':<15} {'Latency(s)':>12} {'AUC':>8}")
        print('  ' + '─'*37)
        for method, d in methods.items():
            marker = ' ◄' if method == 'CryptoFace' else ''
            print(f"  {method:<15} {d['Time']:>12,} {d['AUC']:>8.3f}{marker}")
        # deltas
        our = methods['CryptoFace']
        other = {k:v for k,v in methods.items() if k!='CryptoFace'}
        for bname, bd in other.items():
            dt = bd['Time'] - our['Time']
            da = our['AUC'] - bd['AUC']
            print(f"\n  vs {bname}: time ↓{dt:,}s ({bd['Time']/our['Time']:.2f}× faster), AUC ↑{da:.3f}")

def print_speedup_summary():
    print('\n' + '═'*70)
    print('SPEEDUP SUMMARY')
    print('═'*70)
    our = PAPER_RESULTS['CryptoFaceNet4 (Ours)']
    print(f"\nCryptoFaceNet4 (64×64) vs baselines:")
    for method in ['MPCNN (ResNet32)','MPCNN (ResNet44)','AutoFHE (ResNet32)']:
        bl = PAPER_RESULTS[method]
        speedup  = bl['Latency'] / our['Latency']
        acc_diff = our['Avg'] - bl['Avg']
        sign = '+' if acc_diff >= 0 else ''
        print(f"  vs {method:<28}: {speedup:.1f}× faster | {sign}{acc_diff:.2f}% avg accuracy")

    print(f"\nResolution scaling (CryptoFaceNet):")
    v4 = PAPER_RESULTS['CryptoFaceNet4 (Ours)']
    for vname in ['CryptoFaceNet9 (Ours)','CryptoFaceNet16 (Ours)']:
        v = PAPER_RESULTS[vname]
        lat_ratio = v['Latency'] / v4['Latency']
        acc_gain  = v['Avg'] - v4['Avg']
        print(f"  {vname:<32}: latency ×{lat_ratio:.2f} | acc +{acc_gain:.2f}%")

    print(f"\nBootstrapping reduction vs MPCNN ResNet44:")
    print(f"  Operations: 43 → 1  ({43/1:.0f}× reduction)")
    print(f"  % of latency: 70.27% → 10.34%")
    print(f"  RAM saving: 286G → 269G  (17G saved)")

if __name__ == '__main__':
    print_table1()
    print_table2()
    print_table3()
    print_ijb()
    print_speedup_summary()
    print(f'\n{"="*70}')
    print(f'All CSV tables saved to: {OUT.resolve()}')
    print(f'{"="*70}')
