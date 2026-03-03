#!/usr/bin/env python3
"""
包括的眼計測スクリプト - 5つの異なる手法で眼周囲を評価

1. 眼裂測定（Eye Aperture）: EAR 6点法
2. 眼裂面積（Eye Area）: 8点ポリゴン面積
3. 眼瞼リング面積（Eyelid Ring Area）: 16点ポリゴン面積
4. 眼輪筋近似領域（Orbicularis Approximation）: 16点を拡張
5. 眼窩周囲領域（Periorbital Region）: 眉・眼窩・頬骨を含む広範囲

左右分割なし、全顔面からランドマークを取得し、口すぼめ検出も同時実施。
"""

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import logging
from pathlib import Path
from typing import Tuple, List, Dict
import argparse
from scipy.spatial import ConvexHull
from scipy.signal import medfilt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MediaPipe FaceMesh ランドマークインデックス定義

# 1. 眼裂測定用（EAR 6点法）
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]

# 2. 眼裂面積用（8点ポリゴン - Phase2.3互換）
RIGHT_EYE_POLY = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POLY = [362, 385, 386, 387, 263, 373, 374, 380]

# 3. 眼瞼リング（16点 - 眼瞼縁全体）
RIGHT_EYELID_RING = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYELID_RING = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# 4. 眼輪筋近似領域（眼瞼リングを拡張）
# スケール係数で調整（1.6〜2.5倍推奨）
ORBICULARIS_SCALE = 1.6

# 5. 眼窩周囲領域（広範囲ランドマーク）
# 眉・眼窩上縁・眼窩外側・頬骨上部を含む
RIGHT_PERIORBITAL = [
    # 眉（右）
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
    # 眼窩上縁
    189, 221, 222, 223, 224, 225, 113,
    # 眼瞼周囲
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    # 眼窩外側〜頬骨
    130, 25, 110, 24, 23, 22, 26, 112, 243, 244, 245,
    # 頬骨上部
    116, 117, 118, 119, 120, 121, 128, 129, 203, 205, 206, 207, 50, 101, 36, 205,
    # 鼻側
    198, 217, 126, 142, 97, 98, 99, 100, 49, 48, 64,
    # 追加（眼窩下縁）
    229, 230, 231, 232, 233, 233, 111, 117, 118, 101, 36, 205, 123, 147, 187
]

LEFT_PERIORBITAL = [
    # 眉（左）
    300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
    # 眼窩上縁
    413, 441, 442, 443, 444, 445, 342,
    # 眼瞼周囲
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    # 眼窩外側〜頬骨
    359, 255, 339, 254, 253, 252, 256, 341, 463, 464, 465,
    # 頬骨上部
    345, 346, 347, 348, 349, 350, 357, 358, 423, 425, 426, 427, 280, 330, 266, 425,
    # 鼻側
    419, 437, 355, 371, 326, 327, 328, 329, 279, 278, 294
]

# 口すぼめ検出用
MOUTH_TOP = 0
MOUTH_BOTTOM = 17
MOUTH_LEFT = 61
MOUTH_RIGHT = 291


def calculate_ear(eye_points: np.ndarray) -> float:
    """
    EAR (Eye Aspect Ratio) を計算
    6点法: [P1=外眼角, P2=上瞼外, P3=上瞼内, P4=内眼角, P5=下瞼内, P6=下瞼外]
    EAR = (|P2-P6| + |P3-P5|) / 2

    Returns
    -------
    ear : float
        眼裂高さ（ピクセル）
    """
    if len(eye_points) != 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_points

    # 垂直方向の2つの距離
    dist1 = np.linalg.norm(p2 - p6)
    dist2 = np.linalg.norm(p3 - p5)

    # 平均
    ear = (dist1 + dist2) / 2.0

    return ear


def calculate_polygon_area(points: np.ndarray) -> float:
    """
    ポリゴンの面積を計算（Shoelace formula）

    Parameters
    ----------
    points : np.ndarray
        ポリゴンの頂点座標 (N, 2)

    Returns
    -------
    area : float
        面積（平方ピクセル）
    """
    if len(points) < 3:
        return 0.0

    x = points[:, 0]
    y = points[:, 1]
    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j]
        area -= x[j] * y[i]
    area = abs(area) / 2.0

    return area


def calculate_convex_hull_area(points: np.ndarray) -> float:
    """
    点群の凸包面積を計算

    Parameters
    ----------
    points : np.ndarray
        点群の座標 (N, 2)

    Returns
    -------
    area : float
        凸包面積（平方ピクセル）
    """
    if len(points) < 3:
        return 0.0

    try:
        hull = ConvexHull(points)
        return hull.volume  # 2Dの場合はvolume=area
    except:
        return 0.0


def scale_points_from_center(points: np.ndarray, scale: float) -> np.ndarray:
    """
    点群を中心から拡大縮小

    Parameters
    ----------
    points : np.ndarray
        点群の座標 (N, 2)
    scale : float
        スケール係数（1.0 = 元のまま）

    Returns
    -------
    scaled_points : np.ndarray
        拡大縮小後の座標 (N, 2)
    """
    center = np.mean(points, axis=0)
    vectors = points - center
    scaled_points = center + vectors * scale
    return scaled_points


def calculate_mar(mouth_landmarks: Dict[str, np.ndarray]) -> float:
    """
    MAR (Mouth Aspect Ratio) を計算

    MAR = width / height

    Parameters
    ----------
    mouth_landmarks : dict
        {'top': (x,y), 'bottom': (x,y), 'left': (x,y), 'right': (x,y)}

    Returns
    -------
    mar : float
        口のアスペクト比
    """
    width = np.linalg.norm(mouth_landmarks['right'] - mouth_landmarks['left'])
    height = np.linalg.norm(mouth_landmarks['top'] - mouth_landmarks['bottom'])

    if height < 1e-6:
        return 0.0

    mar = width / height
    return mar


def extract_landmarks_from_video(video_path: str) -> Dict:
    """
    動画から全フレームのランドマークを抽出し、5つの計測を実施

    Returns
    -------
    results : dict
        各計測手法の時系列データ
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"動画を開けません: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # データ保存用
    data = {
        'frame_numbers': [],
        'timestamps': [],
        # 1. 眼裂測定
        'right_aperture': [],
        'left_aperture': [],
        # 2. 眼裂面積
        'right_eye_area': [],
        'left_eye_area': [],
        # 3. 眼瞼リング面積
        'right_eyelid_area': [],
        'left_eyelid_area': [],
        # 4. 眼輪筋近似領域
        'right_orbicularis_area': [],
        'left_orbicularis_area': [],
        # 5. 眼窩周囲領域
        'right_periorbital_area': [],
        'left_periorbital_area': [],
        # 口すぼめ検出
        'mar': [],
        'mouth_pursing': []
    }

    frame_count = 0

    logger.info(f"動画解析開始: {video_path}")
    logger.info(f"  解像度: {width}x{height} @ {fps:.2f}fps")
    logger.info(f"  総フレーム数: {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = frame_count / fps

        # RGB変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            # 検出失敗時はNaN
            data['frame_numbers'].append(frame_count)
            data['timestamps'].append(timestamp)
            for key in data.keys():
                if key not in ['frame_numbers', 'timestamps']:
                    data[key].append(np.nan)
            continue

        face_landmarks = results.multi_face_landmarks[0]

        # ランドマーク座標を取得
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks.append((x, y))

        landmarks = np.array(landmarks)

        # === 1. 眼裂測定（EAR） ===
        right_ear_points = landmarks[RIGHT_EYE_EAR]
        left_ear_points = landmarks[LEFT_EYE_EAR]

        right_aperture = calculate_ear(right_ear_points)
        left_aperture = calculate_ear(left_ear_points)

        # === 2. 眼裂面積（8点ポリゴン） ===
        right_poly_points = landmarks[RIGHT_EYE_POLY]
        left_poly_points = landmarks[LEFT_EYE_POLY]

        right_eye_area = calculate_polygon_area(right_poly_points)
        left_eye_area = calculate_polygon_area(left_poly_points)

        # === 3. 眼瞼リング面積（16点） ===
        right_ring_points = landmarks[RIGHT_EYELID_RING]
        left_ring_points = landmarks[LEFT_EYELID_RING]

        right_eyelid_area = calculate_polygon_area(right_ring_points)
        left_eyelid_area = calculate_polygon_area(left_ring_points)

        # === 4. 眼輪筋近似領域（16点を拡張） ===
        right_orbicularis_points = scale_points_from_center(right_ring_points, ORBICULARIS_SCALE)
        left_orbicularis_points = scale_points_from_center(left_ring_points, ORBICULARIS_SCALE)

        right_orbicularis_area = calculate_convex_hull_area(right_orbicularis_points)
        left_orbicularis_area = calculate_convex_hull_area(left_orbicularis_points)

        # === 5. 眼窩周囲領域（広範囲凸包） ===
        right_periorbital_points = landmarks[RIGHT_PERIORBITAL]
        left_periorbital_points = landmarks[LEFT_PERIORBITAL]

        right_periorbital_area = calculate_convex_hull_area(right_periorbital_points)
        left_periorbital_area = calculate_convex_hull_area(left_periorbital_points)

        # === 口すぼめ検出 ===
        mouth_landmarks = {
            'top': landmarks[MOUTH_TOP],
            'bottom': landmarks[MOUTH_BOTTOM],
            'left': landmarks[MOUTH_LEFT],
            'right': landmarks[MOUTH_RIGHT]
        }

        mar = calculate_mar(mouth_landmarks)
        # 口すぼめ判定は後で一括処理（2パスアルゴリズム）
        mouth_pursing = 0  # 仮の値

        # データ保存
        data['frame_numbers'].append(frame_count)
        data['timestamps'].append(timestamp)
        data['right_aperture'].append(right_aperture)
        data['left_aperture'].append(left_aperture)
        data['right_eye_area'].append(right_eye_area)
        data['left_eye_area'].append(left_eye_area)
        data['right_eyelid_area'].append(right_eyelid_area)
        data['left_eyelid_area'].append(left_eyelid_area)
        data['right_orbicularis_area'].append(right_orbicularis_area)
        data['left_orbicularis_area'].append(left_orbicularis_area)
        data['right_periorbital_area'].append(right_periorbital_area)
        data['left_periorbital_area'].append(left_periorbital_area)
        data['mar'].append(mar)
        data['mouth_pursing'].append(mouth_pursing)

        if frame_count % 100 == 0:
            logger.info(f"  処理済み: {frame_count}/{total_frames} フレーム")

    cap.release()
    face_mesh.close()

    # NumPy配列に変換
    for key in data.keys():
        data[key] = np.array(data[key])

    data['fps'] = fps
    data['width'] = width
    data['height'] = height

    logger.info(f"解析完了: {frame_count} フレーム処理")

    # === 変化量ベースの口すぼめ検出（改良版v3）===
    # v3変更点: ギャップ結合を1秒に拡大、最小持続を2秒に延長
    # 根拠: 口すぼめ刺激は約5秒間を意図しており、MAR信号の短い途切れを
    #        結合し、ノイズ由来の短い検出を除去することで検出精度を改善
    from scipy.ndimage import uniform_filter1d

    mar_array = data['mar']
    if len(mar_array) > 0:
        # 移動平均でベースライン推定（10秒窓）
        window_frames = int(10.0 * fps)
        mar_baseline = uniform_filter1d(mar_array, size=window_frames, mode='nearest')

        # ベースラインからの減少率を計算
        mar_reduction = (mar_baseline - mar_array) / mar_baseline

        # 閾値：ベースラインから7%以上減少した区間を口すぼめと判定
        threshold_reduction = 0.07
        mouth_pursing_raw = (mar_reduction > threshold_reduction).astype(int)

        # ステップ1: ギャップ結合（2秒以内の空白を埋める）
        # 根拠: 口すぼめ中のMAR信号は変動しやすく、短い途切れが生じうる
        # 臨床検証（n=8）で1.0→2.0秒に変更: 弱い口すぼめの途切れを正しく結合
        max_gap = int(2.0 * fps)  # 120フレーム（2秒）
        mouth_pursing_merged = np.copy(mouth_pursing_raw)
        diff = np.diff(np.concatenate(([0], mouth_pursing_raw, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for i in range(len(ends) - 1):
            gap = starts[i + 1] - ends[i]
            if gap <= max_gap:
                mouth_pursing_merged[ends[i]:starts[i + 1]] = 1

        # ステップ2: 最小持続時間フィルター（2秒以上）
        # 根拠: 口すぼめ刺激は約5秒間であり、2秒未満の検出はノイズと判断
        min_duration = int(2.0 * fps)  # 120フレーム（2秒）
        mouth_pursing_final = np.copy(mouth_pursing_merged)
        diff = np.diff(np.concatenate(([0], mouth_pursing_merged, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for start, end in zip(starts, ends):
            if (end - start) < min_duration:
                mouth_pursing_final[start:end] = 0

        data['mouth_pursing'] = mouth_pursing_final

        # 統計情報
        diff_final = np.diff(np.concatenate(([0], mouth_pursing_final, [0])))
        final_episodes = np.sum(diff_final == 1)

        logger.info(f"口すぼめ検出（変化量ベース、改良版v3）:")
        logger.info(f"  ベースラインMAR（10秒移動平均）: 平均 {np.mean(mar_baseline):.2f}")
        logger.info(f"  検出閾値: ベースラインから{threshold_reduction*100:.0f}%以上減少")
        logger.info(f"  ギャップ結合: {max_gap}フレーム（{max_gap/fps:.1f}秒）、最小持続: {min_duration}フレーム（{min_duration/fps:.1f}秒）")
        logger.info(f"  生検出フレーム: {np.sum(mouth_pursing_raw)} → 結合後: {np.sum(mouth_pursing_merged)} → 最終: {np.sum(mouth_pursing_final)}")
        logger.info(f"  最終エピソード: {final_episodes}回")
        logger.info(f"  検出フレーム: {np.sum(mouth_pursing_final)} ({np.sum(mouth_pursing_final)/len(mouth_pursing_final)*100:.1f}%)")

    return data


def smooth_and_interpolate(signal: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    中央値フィルタで平滑化し、NaNを線形補間
    """
    # NaN以外のインデックス
    valid_idx = ~np.isnan(signal)

    if np.sum(valid_idx) < 2:
        return signal

    # 線形補間
    x = np.arange(len(signal))
    signal_interp = np.interp(x, x[valid_idx], signal[valid_idx])

    # 中央値フィルタ
    if kernel_size > 0:
        signal_smooth = medfilt(signal_interp, kernel_size=kernel_size)
    else:
        signal_smooth = signal_interp

    return signal_smooth


def create_comparison_plot(data: Dict, output_path: str, patient_side: str = "right"):
    """
    5つの計測手法の比較グラフを作成

    Parameters
    ----------
    data : dict
        計測データ
    output_path : str
        出力画像パス
    patient_side : str
        "right" または "left" (患側の指定)
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(6, 1, height_ratios=[1, 1, 1, 1, 1, 0.5], hspace=0.3)

    timestamps = data['timestamps']

    # 患側と健側の色
    if patient_side == "right":
        affected_color = 'red'
        healthy_color = 'blue'
        affected_label = 'Right (Affected)'
        healthy_label = 'Left (Healthy)'
    else:
        affected_color = 'blue'
        healthy_color = 'red'
        affected_label = 'Left (Affected)'
        healthy_label = 'Right (Healthy)'

    # 1. 眼裂測定（EAR）
    ax1 = fig.add_subplot(gs[0])
    right_aperture_smooth = smooth_and_interpolate(data['right_aperture'])
    left_aperture_smooth = smooth_and_interpolate(data['left_aperture'])

    if patient_side == "right":
        ax1.plot(timestamps, right_aperture_smooth, color=affected_color, label=affected_label, linewidth=1.5)
        ax1.plot(timestamps, left_aperture_smooth, color=healthy_color, label=healthy_label, linewidth=1.5)
    else:
        ax1.plot(timestamps, left_aperture_smooth, color=affected_color, label=affected_label, linewidth=1.5)
        ax1.plot(timestamps, right_aperture_smooth, color=healthy_color, label=healthy_label, linewidth=1.5)

    ax1.set_ylabel('Aperture (px)', fontsize=10)
    ax1.set_title('1. Eye Aperture (EAR 6-point)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. 眼裂面積
    ax2 = fig.add_subplot(gs[1])
    right_eye_smooth = smooth_and_interpolate(data['right_eye_area'])
    left_eye_smooth = smooth_and_interpolate(data['left_eye_area'])

    if patient_side == "right":
        ax2.plot(timestamps, right_eye_smooth, color=affected_color, label=affected_label, linewidth=1.5)
        ax2.plot(timestamps, left_eye_smooth, color=healthy_color, label=healthy_label, linewidth=1.5)
    else:
        ax2.plot(timestamps, left_eye_smooth, color=affected_color, label=affected_label, linewidth=1.5)
        ax2.plot(timestamps, right_eye_smooth, color=healthy_color, label=healthy_label, linewidth=1.5)

    ax2.set_ylabel('Area (px²)', fontsize=10)
    ax2.set_title('2. Eye Area (8-point polygon)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # 3. 眼瞼リング面積
    ax3 = fig.add_subplot(gs[2])
    right_eyelid_smooth = smooth_and_interpolate(data['right_eyelid_area'])
    left_eyelid_smooth = smooth_and_interpolate(data['left_eyelid_area'])

    if patient_side == "right":
        ax3.plot(timestamps, right_eyelid_smooth, color=affected_color, label=affected_label, linewidth=1.5)
        ax3.plot(timestamps, left_eyelid_smooth, color=healthy_color, label=healthy_label, linewidth=1.5)
    else:
        ax3.plot(timestamps, left_eyelid_smooth, color=affected_color, label=affected_label, linewidth=1.5)
        ax3.plot(timestamps, right_eyelid_smooth, color=healthy_color, label=healthy_label, linewidth=1.5)

    ax3.set_ylabel('Area (px²)', fontsize=10)
    ax3.set_title('3. Eyelid Ring Area (16-point polygon)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # 4. 眼輪筋近似領域
    ax4 = fig.add_subplot(gs[3])
    right_orb_smooth = smooth_and_interpolate(data['right_orbicularis_area'])
    left_orb_smooth = smooth_and_interpolate(data['left_orbicularis_area'])

    if patient_side == "right":
        ax4.plot(timestamps, right_orb_smooth, color=affected_color, label=affected_label, linewidth=1.5)
        ax4.plot(timestamps, left_orb_smooth, color=healthy_color, label=healthy_label, linewidth=1.5)
    else:
        ax4.plot(timestamps, left_orb_smooth, color=affected_color, label=affected_label, linewidth=1.5)
        ax4.plot(timestamps, right_orb_smooth, color=healthy_color, label=healthy_label, linewidth=1.5)

    ax4.set_ylabel('Area (px²)', fontsize=10)
    ax4.set_title('4. Orbicularis Approximation (16-point scaled convex hull)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    # 5. 眼窩周囲領域
    ax5 = fig.add_subplot(gs[4])
    right_peri_smooth = smooth_and_interpolate(data['right_periorbital_area'])
    left_peri_smooth = smooth_and_interpolate(data['left_periorbital_area'])

    if patient_side == "right":
        ax5.plot(timestamps, right_peri_smooth, color=affected_color, label=affected_label, linewidth=1.5)
        ax5.plot(timestamps, left_peri_smooth, color=healthy_color, label=healthy_label, linewidth=1.5)
    else:
        ax5.plot(timestamps, left_peri_smooth, color=affected_color, label=affected_label, linewidth=1.5)
        ax5.plot(timestamps, right_peri_smooth, color=healthy_color, label=healthy_label, linewidth=1.5)

    ax5.set_ylabel('Area (px²)', fontsize=10)
    ax5.set_title('5. Periorbital Region (wide convex hull)', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)

    # 6. 口すぼめ検出
    ax6 = fig.add_subplot(gs[5])
    ax6.fill_between(timestamps, 0, data['mouth_pursing'],
                     color='orange', alpha=0.3, label='Mouth Pursing')
    ax6.set_ylabel('Pursing', fontsize=10)
    ax6.set_xlabel('Time (s)', fontsize=10)
    ax6.set_ylim(-0.1, 1.1)
    ax6.set_yticks([0, 1])
    ax6.set_yticklabels(['No', 'Yes'])
    ax6.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"比較グラフを保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="包括的眼計測スクリプト - 5つの手法で比較"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="入力動画パス"
    )
    parser.add_argument(
        "--patient-side",
        type=str,
        choices=["right", "left"],
        default="right",
        help="患側の指定（right または left）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/comprehensive_metrics",
        help="出力ディレクトリ"
    )

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 動画名
    video_name = Path(args.video_path).stem

    # ランドマーク抽出・計測
    logger.info("=" * 60)
    logger.info("包括的眼計測解析開始")
    logger.info("=" * 60)

    data = extract_landmarks_from_video(args.video_path)

    # 比較グラフ作成
    output_plot = output_dir / f"{video_name}_comprehensive_metrics.png"
    create_comparison_plot(data, str(output_plot), patient_side=args.patient_side)

    # データをCSV保存
    output_csv = output_dir / f"{video_name}_comprehensive_metrics.csv"

    import pandas as pd
    df = pd.DataFrame({
        'frame': data['frame_numbers'],
        'time_s': data['timestamps'],
        'right_aperture_px': data['right_aperture'],
        'left_aperture_px': data['left_aperture'],
        'right_eye_area_px2': data['right_eye_area'],
        'left_eye_area_px2': data['left_eye_area'],
        'right_eyelid_area_px2': data['right_eyelid_area'],
        'left_eyelid_area_px2': data['left_eyelid_area'],
        'right_orbicularis_area_px2': data['right_orbicularis_area'],
        'left_orbicularis_area_px2': data['left_orbicularis_area'],
        'right_periorbital_area_px2': data['right_periorbital_area'],
        'left_periorbital_area_px2': data['left_periorbital_area'],
        'mar': data['mar'],
        'mouth_pursing': data['mouth_pursing']
    })

    df.to_csv(output_csv, index=False)
    logger.info(f"データをCSV保存: {output_csv}")

    logger.info("=" * 60)
    logger.info("解析完了")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
