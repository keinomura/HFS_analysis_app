#!/usr/bin/env python3
"""
Case 2 縦断的解析：術前・術後1日・術後3日・術後7日・術後1ヶ月
IMG_2369（術前）→ IMG_2371（術後1日）→ IMG_2377（術後3日）→ IMG_2404（術後7日）→ IMG_2772（術後1ヶ月）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.append(str(Path(__file__).parent))
from analyze_eye_asymmetry import (
    detect_affected_side,
    compute_asymmetry_metrics,
    detect_blinks_traditional,
    extend_blink_mask
)
from separate_tonic_clonic import extract_tonic_baseline
from analyze_hfs_final import detect_spasm_episodes_final
from analyze_tonic_clonic_metrics import analyze_tonic_metrics, analyze_clonic_metrics


def create_transition_mask(pursing_mask: np.ndarray, fps: float = 60.0,
                           transition_sec: float = 1.0) -> np.ndarray:
    """
    口すぼめ遷移区間の除外マスクを生成。

    Savitzky-Golayフィルタ（3秒窓）によるTonic抽出は、口すぼめ開始・終了時の
    急峻なRD変化を追従できず、Clonic残差にフィルタ漏洩アーチファクトを生む。
    この関数は口すぼめ境界の±transition_sec秒間をマスクし、
    Clonic検出の偽陽性を抑制する。

    引数:
        pursing_mask: 口すぼめ刺激マスク（boolean配列）
        fps: フレームレート
        transition_sec: 遷移区間の片側幅（秒）。デフォルト1.0秒。

    戻り値:
        np.ndarray: 遷移区間マスク（boolean配列、Trueが除外対象）
    """
    if pursing_mask is None:
        return None

    n = len(pursing_mask)
    transition_frames = int(transition_sec * fps)
    mask = np.zeros(n, dtype=bool)

    # 口すぼめの開始・終了境界を検出
    diff = np.diff(pursing_mask.astype(int))
    onsets = np.where(diff == 1)[0] + 1   # 0→1 遷移（開始）
    offsets = np.where(diff == -1)[0] + 1  # 1→0 遷移（終了）

    # 各境界の±transition_framesをマスク
    for boundary in np.concatenate([onsets, offsets]):
        start = max(0, boundary - transition_frames)
        end = min(n, boundary + transition_frames)
        mask[start:end] = True

    return mask


def analyze_comprehensive(csv_file: str, label: str, fps: float = 60.0,
                          forced_side: str = None) -> dict:
    """
    包括的解析（単一時点）

    引数:
        csv_file: comprehensive_metricsのCSVファイルパス
        label: 時点のラベル（例: '術前', '術後1日'）
        fps: 動画のフレームレート（デフォルト60fps）
        forced_side: 患側を強制指定する場合 'left' or 'right'。
                     Noneの場合はCV基準で自動判定。

    戻り値:
        dict: 解析結果（自動判定情報、Tonic/Clonic指標、時系列データを含む）
    """
    TONIC_WINDOW_SEC = 3.0
    BLINK_EXTENSION_FRAMES = 3
    THRESHOLD_STD = 0.5
    MIN_DURATION_FRAMES = 4
    TRANSITION_SEC = 1.0  # 口すぼめ遷移区間除外幅（片側、秒）

    print(f"\n{'='*80}")
    print(f"{label}の解析")
    print(f"{'='*80}")

    df = pd.read_csv(csv_file)
    time = df['time_s'].values
    right_aperture = df['right_aperture_px'].values
    left_aperture = df['left_aperture_px'].values

    pursing_mask = None
    if 'mouth_pursing' in df.columns:
        pursing_mask = df['mouth_pursing'].values.astype(bool)

    # 患側判定: まず自動判定を実行し、結果を記録
    auto_side_info = detect_affected_side(right_aperture, left_aperture)
    auto_detected_side = auto_side_info['affected_side']
    auto_cv_ratio = auto_side_info['cv_ratio']

    # 実際に使う患側を決定
    side_corrected = False
    if forced_side is not None and forced_side != auto_detected_side:
        # 強制指定と自動判定が異なる → 補正適用
        side_corrected = True
        if forced_side == 'left':
            healthy_aperture = right_aperture
            affected_aperture = left_aperture
        else:
            healthy_aperture = left_aperture
            affected_aperture = right_aperture
        used_side = forced_side
        print(f"\n自動判定: {auto_detected_side} (CV比: {auto_cv_ratio:.2f})")
        print(f"  >>> 他時点との一貫性チェックにより補正: {forced_side} を患側として使用")
    else:
        healthy_aperture = auto_side_info['healthy_aperture']
        affected_aperture = auto_side_info['affected_aperture']
        used_side = auto_detected_side
        print(f"\n患側: {used_side} (CV比: {auto_cv_ratio:.2f})")

    # 非対称性メトリクス
    metrics = compute_asymmetry_metrics(healthy_aperture, affected_aperture)
    rd = metrics['relative_diff']

    # 瞬き検出
    blink_mask_core = detect_blinks_traditional(healthy_aperture)
    blink_mask = extend_blink_mask(blink_mask_core, before_frames=BLINK_EXTENSION_FRAMES,
                                   after_frames=BLINK_EXTENSION_FRAMES)

    # Tonic/Clonic分離
    rd_tonic = extract_tonic_baseline(rd, window_seconds=TONIC_WINDOW_SEC, fps=fps)
    rd_clonic = rd - rd_tonic

    # 遷移区間マスクを生成
    transition_mask = create_transition_mask(pursing_mask, fps=fps,
                                             transition_sec=TRANSITION_SEC)

    # 瞬き + 遷移区間の統合除外マスク
    if transition_mask is not None:
        combined_exclude = blink_mask | transition_mask
    else:
        combined_exclude = blink_mask

    # Tonic統計
    tonic_mean = np.mean(rd_tonic)
    tonic_std = np.std(rd_tonic)
    tonic_power = np.sum(np.abs(rd_tonic))
    clonic_power = np.sum(np.abs(rd_clonic))
    tonic_ratio = tonic_power / (tonic_power + clonic_power) * 100

    print(f"\nTonic成分:")
    print(f"  平均: {tonic_mean:.3f}")
    print(f"  SD: {tonic_std:.3f}")
    print(f"  Tonic比率: {tonic_ratio:.1f}%")

    if transition_mask is not None:
        trans_frames = np.sum(transition_mask)
        print(f"  遷移区間除外: {trans_frames}フレーム ({trans_frames/len(rd)*100:.1f}%)")

    # 痙攣検出（瞬き＋遷移区間を除外）
    result = detect_spasm_episodes_final(rd_clonic, threshold_std=THRESHOLD_STD,
                                        min_duration_frames=MIN_DURATION_FRAMES,
                                        exclude_mask=combined_exclude)

    print(f"\nClonic痙攣検出:")
    print(f"  エピソード数: {result['episode_count']}回")
    print(f"  痙攣フレーム: {np.sum(result['spasm_mask'])} ({np.sum(result['spasm_mask'])/len(result['spasm_mask'])*100:.1f}%)")

    if result['episode_count'] > 0:
        mean_duration_ms = np.mean(result['durations']) / fps * 1000
        print(f"  平均持続時間: {mean_duration_ms:.0f} ms")

    # 口すぼめ刺激時のメトリクス
    tonic_metrics = None
    clonic_metrics = None
    if pursing_mask is not None:
        tonic_metrics = analyze_tonic_metrics(rd_tonic, pursing_mask)
        clonic_metrics = analyze_clonic_metrics(rd_clonic, result['spasm_mask'], pursing_mask, fps,
                                                exclude_mask=combined_exclude)

        print(f"\n口すぼめ刺激時（誘発時）:")
        print(f"  Tonic刺激時上昇: {tonic_metrics['baseline_elevation']:.3f}")
        print(f"  Clonic発生率: {clonic_metrics['spasm_rate_per_sec']:.3f} /秒")
        print(f"  Clonicカバレッジ: {clonic_metrics['spasm_coverage']:.1f}%")
        print(f"  Clonic平均振幅: {clonic_metrics['mean_amplitude']:.3f}")
        if clonic_metrics['mean_duration_ms'] > 0:
            print(f"  Clonic平均持続時間: {clonic_metrics['mean_duration_ms']:.0f} ms")

    return {
        'label': label,
        # 患側判定情報（人間が確認するため両方保持）
        'auto_detected_side': auto_detected_side,
        'auto_cv_ratio': auto_cv_ratio,
        'used_side': used_side,
        'side_corrected': side_corrected,
        # 解析指標
        'cv_ratio': auto_cv_ratio,
        'tonic_mean': tonic_mean,
        'tonic_std': tonic_std,
        'tonic_ratio': tonic_ratio,
        'tonic_elevation': tonic_metrics['baseline_elevation'] if tonic_metrics else 0,
        'clonic_episodes': result['episode_count'],
        'clonic_rate': clonic_metrics['spasm_rate_per_sec'] if clonic_metrics else 0,
        'clonic_coverage': clonic_metrics['spasm_coverage'] if clonic_metrics else 0,
        'clonic_amplitude': clonic_metrics['mean_amplitude'] if clonic_metrics else 0,
        'clonic_duration': clonic_metrics['mean_duration_ms'] if clonic_metrics else 0,
        'rd_tonic': rd_tonic,
        'rd_clonic': rd_clonic,
        'time': time,
        'spasm_mask': result['spasm_mask'],
        'pursing_mask': pursing_mask
    }


def check_side_consistency(results: list) -> dict:
    """
    全時点の患側判定の一貫性をチェックする（術前アンカー方式）。

    縦断解析では、術前（最初の時点）の患側判定を基準（アンカー）として使用する。
    治療効果によりCV値が変化し自動判定が反転することがあるため、
    術前の判定を全時点に適用し、不一致があれば補正・フラグ表示する。

    引数:
        results: analyze_comprehensiveの結果リスト（auto_detected_side含む）
                 results[0] が術前（アンカー時点）であること

    戻り値:
        dict: {
            'consistent': 全時点一致ならTrue,
            'anchor_side': 術前の患側判定（基準）,
            'consensus_side': anchor_sideと同一（後方互換のため保持）,
            'flipped_indices': 反転している時点のインデックスリスト,
            'summary': 人間確認用のサマリー文字列
        }
    """
    # 術前（最初の時点）をアンカーとして使用
    anchor_side = results[0]['auto_detected_side']

    flipped_indices = [i for i, r in enumerate(results)
                       if r['auto_detected_side'] != anchor_side]
    consistent = len(flipped_indices) == 0

    # 確認用サマリーを作成
    lines = []
    lines.append("=" * 80)
    lines.append("患側判定の一貫性チェック（術前アンカー方式）")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  基準: 術前（{results[0]['label']}）の自動判定 = {anchor_side}")
    lines.append(f"  方針: 術前の患側判定を全時点に適用（治療による反転を考慮）")
    lines.append("")
    lines.append(f"  {'時点':<15} {'自動判定':>10} {'CV比':>10} {'状態':>15}")
    lines.append("  " + "-" * 55)
    for i, r in enumerate(results):
        if i == 0:
            status = "アンカー（基準）"
        elif i not in flipped_indices:
            status = "一致"
        else:
            status = "反転 → 補正対象"
        lines.append(f"  {r['label']:<15} {r['auto_detected_side']:>10} {r['auto_cv_ratio']:>10.2f} {status:>15}")
    lines.append("")
    if not consistent:
        flipped_labels = [results[i]['label'] for i in flipped_indices]
        lines.append(f"  補正対象: {', '.join(flipped_labels)}")
        lines.append(f"  → 術前判定 ({anchor_side}) に統一して再解析します")
    else:
        lines.append("  全時点で患側判定が一致しています。補正不要。")
    lines.append("")

    summary = "\n".join(lines)
    return {
        'consistent': consistent,
        'anchor_side': anchor_side,
        'consensus_side': anchor_side,
        'flipped_indices': flipped_indices,
        'summary': summary
    }


def main():
    timepoints = [
        ('output/comprehensive_metrics/IMG_2369_comprehensive_metrics.csv', '術前', 0),
        ('output/comprehensive_metrics/IMG_2371_comprehensive_metrics.csv', '術後1日', 1),
        ('output/comprehensive_metrics/IMG_2377_comprehensive_metrics.csv', '術後3日', 3),
        ('output/comprehensive_metrics/IMG_2404_comprehensive_metrics.csv', '術後7日', 7),
        ('output/comprehensive_metrics/IMG_2772_comprehensive_metrics.csv', '術後1ヶ月', 30),
    ]

    # --- Phase 1: 全時点を自動判定で解析 ---
    print("\n" + "#" * 80)
    print("# Phase 1: 各時点の独立解析（自動患側判定）")
    print("#" * 80)

    results = []
    for csv_file, label, day in timepoints:
        result = analyze_comprehensive(csv_file, label)
        result['day'] = day
        result['csv_file'] = csv_file
        results.append(result)

    # --- Phase 2: 一貫性チェック ---
    consistency = check_side_consistency(results)
    print(consistency['summary'])

    # --- Phase 3: 反転時点のみ再解析 ---
    if not consistency['consistent']:
        print("#" * 80)
        print("# Phase 2: 反転時点の再解析（患側補正適用）")
        print("#" * 80)

        for idx in consistency['flipped_indices']:
            tp = timepoints[idx]
            csv_file, label, day = tp
            result = analyze_comprehensive(csv_file, label,
                                           forced_side=consistency['consensus_side'])
            result['day'] = day
            result['csv_file'] = csv_file
            results[idx] = result

    # 比較表
    print(f"\n{'='*120}")
    print("Case 2 術前術後比較：客観的指標（5時点）")
    print(f"{'='*120}\n")

    # 患側判定の最終状態を表示
    header = f"{'指標':<30}"
    for r in results:
        header += f" {r['label']:>15}"
    print(header)
    print("-" * 120)

    # 患側判定行（補正があった場合 * 印付き）
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

    metrics = [
        ('CV比', 'cv_ratio', ''),
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
    
    # グラフ作成
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    days = [r['day'] for r in results]
    labels = [r['label'] for r in results]
    
    # Graph 1: Tonic/Clonic比率の推移
    ax1 = axes[0, 0]
    tonic_ratios = [r['tonic_ratio'] for r in results]
    clonic_ratios = [100 - r['tonic_ratio'] for r in results]
    ax1.plot(days, tonic_ratios, 'o-', color='purple', linewidth=2.5, markersize=10, label='Tonic比率')
    ax1.plot(days, clonic_ratios, 's-', color='green', linewidth=2.5, markersize=10, label='Clonic比率')
    ax1.set_xlabel('術後日数', fontsize=12)
    ax1.set_ylabel('比率 (%)', fontsize=12)
    ax1.set_title('Tonic/Clonic比率の推移', fontsize=13, fontweight='bold')
    ax1.set_xticks(days)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graph 2: Clonic発生率とカバレッジ
    ax2 = axes[0, 1]
    rates = [r['clonic_rate'] for r in results]
    coverages = [r['clonic_coverage'] for r in results]
    ax2_twin = ax2.twinx()
    l1 = ax2.plot(days, rates, 'o-', color='red', linewidth=2.5, markersize=10, label='発生率')
    l2 = ax2_twin.plot(days, coverages, 's-', color='orange', linewidth=2.5, markersize=10, label='カバレッジ')
    ax2.set_xlabel('術後日数', fontsize=12)
    ax2.set_ylabel('発生率（回/秒）', fontsize=12, color='red')
    ax2_twin.set_ylabel('カバレッジ（%）', fontsize=12, color='orange')
    ax2.set_title('Clonic成分の推移', fontsize=13, fontweight='bold')
    ax2.set_xticks(days)
    ax2.set_xticklabels(labels)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    ax2.grid(True, alpha=0.3)
    lines = l1 + l2
    labels_leg = [l.get_label() for l in lines]
    ax2.legend(lines, labels_leg, loc='upper right')
    
    # Graph 3: Tonic成分の推移
    ax3 = axes[1, 0]
    tonic_means = [r['tonic_mean'] for r in results]
    tonic_elevs = [r['tonic_elevation'] for r in results]
    ax3.plot(days, tonic_means, 'o-', color='purple', linewidth=2.5, markersize=10, label='平均ベースライン')
    ax3.plot(days, tonic_elevs, 's-', color='red', linewidth=2.5, markersize=10, label='刺激時上昇')
    ax3.set_xlabel('術後日数', fontsize=12)
    ax3.set_ylabel('Tonic成分', fontsize=12)
    ax3.set_title('Tonic成分の推移', fontsize=13, fontweight='bold')
    ax3.set_xticks(days)
    ax3.set_xticklabels([r['label'] for r in results])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Graph 4: 痙攣特性（振幅・持続時間）
    ax4 = axes[1, 1]
    amps = [r['clonic_amplitude'] for r in results]
    durs = [r['clonic_duration'] for r in results]
    ax4_twin = ax4.twinx()
    l1 = ax4.plot(days, amps, 'o-', color='blue', linewidth=2.5, markersize=10, label='振幅')
    l2 = ax4_twin.plot(days, durs, '^-', color='darkblue', linewidth=2.5, markersize=10, label='持続時間')
    ax4.set_xlabel('術後日数', fontsize=12)
    ax4.set_ylabel('振幅', fontsize=12, color='blue')
    ax4_twin.set_ylabel('持続時間（ms）', fontsize=12, color='darkblue')
    ax4.set_title('痙攣特性の推移', fontsize=13, fontweight='bold')
    ax4.set_xticks(days)
    ax4.set_xticklabels([r['label'] for r in results])
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4_twin.tick_params(axis='y', labelcolor='darkblue')
    ax4.grid(True, alpha=0.3)
    lines = l1 + l2
    labels_leg = [l.get_label() for l in lines]
    ax4.legend(lines, labels_leg, loc='upper right')
    
    plt.tight_layout()
    output_path = 'output/case2_longitudinal_5timepoints.png'
    plt.savefig(output_path, dpi=300)
    print(f"\nグラフ保存: {output_path}")


if __name__ == '__main__':
    main()
