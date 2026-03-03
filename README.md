# HFS Automated Quantification Pipeline

[![DOI](https://zenodo.org/badge/1171376249.svg)](https://doi.org/10.5281/zenodo.18845312)

Automated tonic-clonic decomposition of hemifacial spasm (HFS) severity using smartphone video and MediaPipe Face Mesh.

## Overview

This pipeline provides objective, automated quantification of HFS by:

1. **Landmark detection** - MediaPipe Face Mesh extracts 468 facial landmarks from smartphone video (iPhone, 60 fps)
2. **Eye aperture measurement** - Bilateral palpebral fissure heights are computed frame-by-frame
3. **Relative difference (RD)** - Asymmetry between affected and unaffected sides, normalized to the unaffected side
4. **Tonic-clonic decomposition** - Savitzky-Golay filter separates sustained narrowing (tonic) from intermittent spasms (clonic)
5. **Quantitative metrics** - Tonic mean, tonic elevation ratio, clonic spasm rate, and clonic coverage are computed during standardized mouth-pursing provocation windows

## Key Features

- Fully automated processing from smartphone video to quantitative metrics
- Tonic/clonic decomposition for independent assessment of sustained vs. intermittent spasm components
- Mouth-pursing provocation protocol with automatic detection via mouth aspect ratio (MAR)
- Automated affected-side determination with preoperative anchoring for longitudinal consistency
- Adaptive blink removal algorithm

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

## Core Scripts

| Script | Description |
|--------|-------------|
| `src/comprehensive_eye_metrics.py` | MediaPipe video processing to CSV (landmark extraction, eye aperture, blink detection) |
| `src/compare_case2_longitudinal.py` | Core analysis functions: tonic-clonic decomposition, spasm detection, metric computation |
| `src/compare_all_cases_jns.py` | Multi-case longitudinal analysis with unified visualization |

### Usage

```bash
# Step 1: Extract eye metrics from video
python3 src/comprehensive_eye_metrics.py path/to/video.mp4 --output-dir output/

# Step 2: Run longitudinal analysis
python3 src/compare_all_cases_jns.py
```

## Pipeline Architecture

```
Smartphone Video (iPhone 60fps)
        |
MediaPipe Face Mesh (468 landmarks)
        |
   Left / Right Eye Aperture Heights
        |               \
  Blink Removal     Affected Side (CV + preop anchor)
        |               /
  Relative Difference (RD)
        |
  Savitzky-Golay Filter (3s window, 2nd order)
       / \
  Tonic   Clonic --> Spasm Detection
       \  /
  Quantitative Metrics
  (Tonic Mean | Tonic Elevation | Clonic Rate | Clonic Coverage)
```

## Citation

**If you use this pipeline or any part of its methodology in your research, you must cite the following paper:**

> Nomura K, Sakai H. Automated Tonic-Clonic Decomposition of Hemifacial Spasm Severity from Smartphone Video Using MediaPipe Face Mesh. *Computer Methods and Programs in Biomedicine* (submitted).

This is a condition of the MIT License for academic use of this software.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Data Availability

Patient video data cannot be shared due to privacy restrictions. De-identified quantitative metrics are available from the corresponding author upon reasonable request.
