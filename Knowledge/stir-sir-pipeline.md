# 眼窩STIRのSIR解析パイプライン（TED活動性評価）

## 目的・定義
甲状腺眼症(TED/Graves眼症)の眼窩炎症を定量する **SIR (Signal Intensity Ratio)**。
Higashiyama et al., Jpn J Ophthalmol 2015 の手法。

```
SIR = 外眼筋のSTIR平均信号 / 大脳白質のSTIR平均信号
```
- 白質を内部参照に使う理由: STIR信号は患者・装置で絶対値が変動するが、白質で正規化すると比較可能になる。
- 異常域: **SIR > 2.0**（正常コントロールは約1.1〜1.5）。

## パイプライン全体像
```
DICOM(coronal STIR)
  → 大脳セグメンテーション (SAM2 教師 → LWBNA-UNet に蒸留)
  → ヒストグラムで白質輝度抽出（低信号ピーク = 白質）
  → SIR = 各voxel信号 / 白質参照
  → 眼窩SIRマップ（ヒートマップ + SIR2.0等高線）
```

## 主要ファイル(MRI_TOM)
| ファイル | 役割 |
|---|---|
| `sir_analysis.py` | 白質抽出・SIR・眼窩SIRマップ・DICOM読込。`brain_mask_fn`で脳セグ法を差替 |
| `sam2_brain.py` | SAM2で大脳分離（教師/ラベリング） |
| `lwbna_brain.py` | 蒸留LWBNA-UNetの推論（SAM2と同`brain_mask`I/F） |
| `orbital_sir_map.py` | 一気通貫CLI。`--lwbna`(本番)/`--sam2`(ラベリング) |

## 本番コマンド（軽量・SAM2不要）
```bash
PY=C:/Users/CorneAI/MRI_TOM/venv/Scripts/python.exe
$PY orbital_sir_map.py --dicom HANDAI_STIR_Coronal --series <症例> \
    --brain-slices 8-13 --orbit-slices 18-21 \
    --lwbna brain_lwbna_best.pt --plot out.png --save-nifti out.nii.gz
```
- `--brain-slices`/`--orbit-slices` 省略時は globe検出で自動推定（要QC）。
- 実行は必ず venv（torch/pydicom/transformers/cv2入り）。

## 重要な前提
- **入力STIRは生信号**（min-max/z-score正規化しない）。SIRは生信号の比のため。
- 白質参照は**複数脳スライスの中央値**で安定化（`auto_white_matter_reference`）。
- 関連: [[white-matter-histogram-extraction]] [[brain-segmentation-distillation]] [[stir-domain-shift-and-anatomy]]
