#!/usr/bin/env python3
"""
JNS論文用：3症例の縦断的解析と比較

各症例について術前アンカー方式で患側判定を統一し、
Tonic/Clonic成分の経時変化を比較する。

Case 1: 術後自覚症状悪化→最終的に改善（6時点）
Case 2: 肉眼で分からないが客観的指標で改善を検出（5時点）
Case 3: 著明改善だが術後顔面麻痺でアルゴリズム限界あり（4時点）

出力:
  - コンソール: 各症例の比較表
  - output/jns_paper/jns_all_cases_comparison.png: 3症例の縦断グラフ
  - output/jns_paper/jns_summary_table.csv: 論文用サマリーテーブル
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic',
                                    'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.append(str(Path(__file__).parent))
from compare_case2_longitudinal import analyze_comprehensive, check_side_consistency


# ============================================================
# 症例定義
# ============================================================

CASES = {
    'Case 1': {
        'description': '術後一過性悪化→最終改善',
        'timepoints': [
            ('output/comprehensive_metrics/IMG_2340_comprehensive_metrics.csv', '術前', 0),
            ('output/comprehensive_metrics/IMG_2348_comprehensive_metrics.csv', '術後1日', 1),
            ('output/comprehensive_metrics/IMG_2357_comprehensive_metrics.csv', '術後3日', 3),
            ('output/comprehensive_metrics/IMG_2359_comprehensive_metrics.csv', '術後4日', 4),
            ('output/comprehensive_metrics/IMG_2370_comprehensive_metrics.csv', '術後7日', 7),
            ('output/comprehensive_metrics/IMG_2489_comprehensive_metrics.csv', '術後1ヶ月', 30),
        ],
    },
    'Case 2': {
        'description': '肉眼で不明だが指標で改善を検出',
        'timepoints': [
            ('output/comprehensive_metrics/IMG_2369_comprehensive_metrics.csv', '術前', 0),
            ('output/comprehensive_metrics/IMG_2371_comprehensive_metrics.csv', '術後1日', 1),
            ('output/comprehensive_metrics/IMG_2377_comprehensive_metrics.csv', '術後3日', 3),
            ('output/comprehensive_metrics/IMG_2404_comprehensive_metrics.csv', '術後7日', 7),
            ('output/comprehensive_metrics/IMG_2772_comprehensive_metrics.csv', '術後1ヶ月', 30),
        ],
    },
    'Case 3': {
        'description': '著明改善・術後顔面麻痺あり',
        'timepoints': [
            ('output/comprehensive_metrics/IMG_2407_comprehensive_metrics.csv', '術前', 0),
            ('output/comprehensive_metrics/IMG_2499_comprehensive_metrics.csv', '術後1日', 1),
            ('output/comprehensive_metrics/IMG_2516_comprehensive_metrics.csv', '術後4日', 4),
            ('output/comprehensive_metrics/IMG_2616_comprehensive_metrics.csv', '術後2週', 14),
        ],
        # 麻痺によりClonic指標が無効な時点（R/L比 > 1.0 = 患側が開いている）
        'paresis_timepoints': ['術後1日', '術後2週'],
    },
}


def detect_paresis(result: dict) -> bool:
    """
    術後顔面麻痺の有無を判定する。

    RDの平均が大きく負（患側が健側より開いている）の場合、
    顔面麻痺と判定する。

    引数:
        result: analyze_comprehensiveの結果dict

    戻り値:
        bool: 麻痺ありならTrue
    """
    rd_tonic_mean = result['tonic_mean']
    # Tonic平均が-0.1以下 = 患側が健側より10%以上開いている → 麻痺
    return rd_tonic_mean < -0.1


def analyze_single_case(case_name: str, case_config: dict) -> list:
    """
    1症例の全時点を解析する（術前アンカー方式）。

    引数:
        case_name: 症例名（例: 'Case 1'）
        case_config: CASES辞書の1エントリ

    戻り値:
        list: 各時点のanalyze_comprehensive結果のリスト
              各要素に 'day', 'csv_file', 'has_paresis' を追加
    """
    timepoints = case_config['timepoints']
    paresis_labels = case_config.get('paresis_timepoints', [])

    print(f"\n{'#'*80}")
    print(f"# {case_name}: {case_config['description']}")
    print(f"# 時点数: {len(timepoints)}")
    print(f"{'#'*80}")

    # Phase 1: 全時点を自動判定で解析
    print(f"\n--- Phase 1: 各時点の独立解析（自動患側判定）---")
    results = []
    for csv_file, label, day in timepoints:
        if not Path(csv_file).exists():
            print(f"  [SKIP] {csv_file} が見つかりません")
            continue
        result = analyze_comprehensive(csv_file, label)
        result['day'] = day
        result['csv_file'] = csv_file
        results.append(result)

    if len(results) == 0:
        print(f"  {case_name}: 解析可能なデータがありません")
        return []

    # Phase 2: 一貫性チェック
    consistency = check_side_consistency(results)
    print(consistency['summary'])

    # Phase 3: 反転時点のみ再解析
    if not consistency['consistent']:
        print(f"--- Phase 2: 反転時点の再解析（患側補正適用）---")
        for idx in consistency['flipped_indices']:
            tp = timepoints[idx]
            csv_file, label, day = tp
            result = analyze_comprehensive(csv_file, label,
                                           forced_side=consistency['consensus_side'])
            result['day'] = day
            result['csv_file'] = csv_file
            results[idx] = result

    # 麻痺フラグを追加
    for r in results:
        has_paresis = detect_paresis(r) or r['label'] in paresis_labels
        r['has_paresis'] = has_paresis

    return results


def print_case_table(case_name: str, results: list):
    """
    1症例の比較表をコンソールに出力する。

    引数:
        case_name: 症例名
        results: analyze_single_caseの結果リスト
    """
    print(f"\n{'='*120}")
    print(f"{case_name} 術前術後比較（{len(results)}時点）")
    print(f"{'='*120}\n")

    # ヘッダー
    header = f"{'指標':<30}"
    for r in results:
        paresis_mark = " [麻痺]" if r.get('has_paresis') else ""
        header += f" {r['label'] + paresis_mark:>15}"
    print(header)
    print("-" * 120)

    # 患側判定行
    side_row = f"{'患側判定':<30}"
    for r in results:
        mark = " *補正*" if r['side_corrected'] else ""
        side_row += f" {r['used_side'] + mark:>15}"
    print(side_row)

    cv_row = f"{'CV比（自動判定時）':<30}"
    for r in results:
        cv_row += f" {r['auto_cv_ratio']:>15.2f}"
    print(cv_row)
    print("-" * 120)

    # メトリクス行
    metrics = [
        ('Tonic平均値', 'tonic_mean', ''),
        ('Tonic比率', 'tonic_ratio', '%'),
        ('Tonic刺激時上昇', 'tonic_elevation', ''),
        ('Clonic痙攣回数', 'clonic_episodes', '回'),
        ('Clonic発生率', 'clonic_rate', '/秒'),
        ('Clonicカバレッジ', 'clonic_coverage', '%'),
        ('Clonic平均振幅', 'clonic_amplitude', ''),
        ('Clonic平均持続時間', 'clonic_duration', 'ms'),
    ]

    for name, key, unit in metrics:
        row = f"{name:<30}"
        for r in results:
            val = r[key]
            if unit == '%':
                row += f" {val:>14.1f}%"
            elif unit == 'ms':
                row += f" {val:>14.0f}{unit}"
            elif unit == '回':
                row += f" {val:>14.0f}{unit}"
            elif unit == '/秒':
                row += f" {val:>14.3f}{unit}"
            else:
                row += f" {val:>15.3f}"
        print(row)

    # 術前比の変化率（最終時点 vs 術前）
    if len(results) >= 2:
        pre = results[0]
        # 最終有効時点を探す（麻痺なしの最後の時点）
        valid_results = [r for r in results if not r.get('has_paresis')]
        if len(valid_results) >= 2:
            post = valid_results[-1]
        else:
            post = results[-1]

        print(f"\n  術前 vs {post['label']} の変化率:")
        if pre['tonic_mean'] != 0:
            tonic_change = (post['tonic_mean'] - pre['tonic_mean']) / abs(pre['tonic_mean']) * 100
            print(f"    Tonic平均: {tonic_change:+.1f}%")
        if pre['clonic_rate'] != 0:
            clonic_change = (post['clonic_rate'] - pre['clonic_rate']) / abs(pre['clonic_rate']) * 100
            print(f"    Clonic発生率: {clonic_change:+.1f}%")
        if pre['tonic_elevation'] != 0:
            elev_change = (post['tonic_elevation'] - pre['tonic_elevation']) / abs(pre['tonic_elevation']) * 100
            print(f"    Tonic刺激時上昇: {elev_change:+.1f}%")


def create_summary_csv(all_results: dict, output_path: str):
    """
    論文用サマリーCSVを出力する。

    引数:
        all_results: {'Case 1': [results], 'Case 2': [results], ...}
        output_path: CSVファイルの出力パス
    """
    rows = []
    for case_name, results in all_results.items():
        for r in results:
            rows.append({
                'Case': case_name,
                'Timepoint': r['label'],
                'Day': r['day'],
                'Affected_Side': r['used_side'],
                'Side_Corrected': r['side_corrected'],
                'CV_Ratio': round(r['auto_cv_ratio'], 3),
                'Has_Paresis': r.get('has_paresis', False),
                'Tonic_Mean': round(r['tonic_mean'], 4),
                'Tonic_Ratio_pct': round(r['tonic_ratio'], 1),
                'Tonic_Elevation': round(r['tonic_elevation'], 4),
                'Clonic_Episodes': r['clonic_episodes'],
                'Clonic_Rate_per_sec': round(r['clonic_rate'], 4),
                'Clonic_Coverage_pct': round(r['clonic_coverage'], 1),
                'Clonic_Amplitude': round(r['clonic_amplitude'], 4),
                'Clonic_Duration_ms': round(r['clonic_duration'], 1),
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nサマリーCSV保存: {output_path}")


def create_comparison_figure(all_results: dict, output_path: str):
    """
    3症例の縦断比較グラフを作成する（論文Figure用）。

    4パネル構成:
      (1) Tonic平均の経時変化
      (2) Tonic刺激時上昇の経時変化
      (3) Clonic発生率の経時変化
      (4) Clonic振幅の経時変化

    引数:
        all_results: {'Case 1': [results], 'Case 2': [results], ...}
        output_path: 画像ファイルの出力パス
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = {'Case 1': '#1f77b4', 'Case 2': '#ff7f0e', 'Case 3': '#2ca02c'}
    markers = {'Case 1': 'o', 'Case 2': 's', 'Case 3': '^'}

    for case_name, results in all_results.items():
        color = colors[case_name]
        marker = markers[case_name]
        days = [r['day'] for r in results]

        # 麻痺時点を区別してプロットする
        valid_mask = [not r.get('has_paresis', False) for r in results]
        paresis_mask = [r.get('has_paresis', False) for r in results]

        # --- Panel 1: Tonic平均 ---
        ax1 = axes[0, 0]
        vals = [r['tonic_mean'] for r in results]
        # 有効な点のみを実線でつなぐ
        valid_days = [d for d, v in zip(days, valid_mask) if v]
        valid_vals = [v for v, m in zip(vals, valid_mask) if m]
        ax1.plot(valid_days, valid_vals, f'{marker}-', color=color,
                 linewidth=2, markersize=8, label=case_name)
        # 麻痺時点は白抜きマーカーで
        paresis_days = [d for d, m in zip(days, paresis_mask) if m]
        paresis_vals = [v for v, m in zip(vals, paresis_mask) if m]
        if paresis_days:
            ax1.plot(paresis_days, paresis_vals, marker, color=color,
                     markersize=8, markerfacecolor='white', markeredgewidth=2)

        # --- Panel 2: Tonic刺激時上昇 ---
        ax2 = axes[0, 1]
        vals = [r['tonic_elevation'] for r in results]
        valid_vals = [v for v, m in zip(vals, valid_mask) if m]
        ax2.plot(valid_days, valid_vals, f'{marker}-', color=color,
                 linewidth=2, markersize=8, label=case_name)
        paresis_vals = [v for v, m in zip(vals, paresis_mask) if m]
        if paresis_days:
            ax2.plot(paresis_days, paresis_vals, marker, color=color,
                     markersize=8, markerfacecolor='white', markeredgewidth=2)

        # --- Panel 3: Clonic発生率 ---
        ax3 = axes[1, 0]
        vals = [r['clonic_rate'] for r in results]
        valid_vals = [v for v, m in zip(vals, valid_mask) if m]
        ax3.plot(valid_days, valid_vals, f'{marker}-', color=color,
                 linewidth=2, markersize=8, label=case_name)
        paresis_vals = [v for v, m in zip(vals, paresis_mask) if m]
        if paresis_days:
            ax3.plot(paresis_days, paresis_vals, marker, color=color,
                     markersize=8, markerfacecolor='white', markeredgewidth=2)

        # --- Panel 4: Clonic振幅 ---
        ax4 = axes[1, 1]
        vals = [r['clonic_amplitude'] for r in results]
        valid_vals = [v for v, m in zip(vals, valid_mask) if m]
        ax4.plot(valid_days, valid_vals, f'{marker}-', color=color,
                 linewidth=2, markersize=8, label=case_name)
        paresis_vals = [v for v, m in zip(vals, paresis_mask) if m]
        if paresis_days:
            ax4.plot(paresis_days, paresis_vals, marker, color=color,
                     markersize=8, markerfacecolor='white', markeredgewidth=2)

    # 軸設定
    for ax, title, ylabel in [
        (axes[0, 0], 'Tonic Component (Mean RD)', 'Relative Difference'),
        (axes[0, 1], 'Tonic Elevation During Provocation', 'RD Elevation'),
        (axes[1, 0], 'Clonic Spasm Rate During Provocation', 'Spasms / sec'),
        (axes[1, 1], 'Clonic Spasm Amplitude', 'RD Amplitude'),
    ]:
        ax.set_xlabel('Post-operative Day', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 白抜きマーカーの凡例注記
    fig.text(0.5, 0.01,
             'Open markers indicate timepoints with post-operative facial paresis '
             '(Clonic metrics unreliable)',
             ha='center', fontsize=10, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n比較グラフ保存: {output_path}")


def print_paper_summary(all_results: dict):
    """
    論文用のサマリーテーブル（術前 vs 最終有効時点）を出力する。

    引数:
        all_results: {'Case 1': [results], ...}
    """
    print(f"\n{'='*100}")
    print("論文用サマリー: 術前 vs 最終有効時点")
    print(f"{'='*100}\n")

    header = f"{'指標':<30} {'Case 1':>20} {'Case 2':>20} {'Case 3':>20}"
    print(header)
    print("-" * 100)

    summaries = {}
    for case_name, results in all_results.items():
        pre = results[0]
        # 麻痺なしの最終時点を選択
        valid = [r for r in results if not r.get('has_paresis')]
        post = valid[-1] if len(valid) >= 2 else results[-1]

        summaries[case_name] = {
            'pre': pre,
            'post': post,
            'post_label': post['label'],
        }

    # 最終評価時点
    row = f"{'最終評価時点':<30}"
    for cn in ['Case 1', 'Case 2', 'Case 3']:
        s = summaries[cn]
        row += f" {s['post_label']:>20}"
    print(row)
    print("-" * 100)

    # メトリクスと変化率
    metrics_def = [
        ('Tonic平均', 'tonic_mean'),
        ('Tonic刺激時上昇', 'tonic_elevation'),
        ('Clonic発生率 (/秒)', 'clonic_rate'),
        ('Clonicカバレッジ (%)', 'clonic_coverage'),
        ('Clonic振幅', 'clonic_amplitude'),
    ]

    for name, key in metrics_def:
        # 術前値
        row_pre = f"{'  ' + name + '（術前）':<30}"
        for cn in ['Case 1', 'Case 2', 'Case 3']:
            val = summaries[cn]['pre'][key]
            row_pre += f" {val:>20.3f}"
        print(row_pre)

        # 術後値
        row_post = f"{'  ' + name + '（術後）':<30}"
        for cn in ['Case 1', 'Case 2', 'Case 3']:
            val = summaries[cn]['post'][key]
            row_post += f" {val:>20.3f}"
        print(row_post)

        # 変化率
        row_change = f"{'  変化率':<30}"
        for cn in ['Case 1', 'Case 2', 'Case 3']:
            pre_val = summaries[cn]['pre'][key]
            post_val = summaries[cn]['post'][key]
            if abs(pre_val) > 0.001:
                change = (post_val - pre_val) / abs(pre_val) * 100
                row_change += f" {change:>19.1f}%"
            else:
                row_change += f" {'N/A':>20}"
        print(row_change)
        print()


def main():
    """3症例すべてを解析し、比較結果を出力する。"""

    all_results = {}

    # 各症例を順次解析
    for case_name, case_config in CASES.items():
        results = analyze_single_case(case_name, case_config)
        if results:
            all_results[case_name] = results

    # 各症例の詳細テーブル
    for case_name, results in all_results.items():
        print_case_table(case_name, results)

    # 論文用サマリー
    if len(all_results) == 3:
        print_paper_summary(all_results)

    # CSV出力
    create_summary_csv(all_results, 'output/jns_paper/jns_summary_table.csv')

    # グラフ作成
    create_comparison_figure(all_results, 'output/jns_paper/jns_all_cases_comparison.png')

    print(f"\n{'='*80}")
    print("解析完了")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
