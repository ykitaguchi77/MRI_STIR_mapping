# STIRの解剖配置・信号特性・ドメインシフト

## 冠状STIRのスライス配置
データ例 `HANDAI_STIR_Coronal`（阪大, DICOM, 約344シリーズ, 256×256, 0.7mm, ~19-24スライス）。
ファイル名: `<患者ID>_<日付>_STIR_Coronal_<スライス>.dcm`。

| 位置 | スライス(InstanceNumber順) | 内容 |
|---|---|---|
| 後方 | 小index (例 0-15) | **大脳**（白質参照に使う） |
| 前方 | 大index (例 16-22) | **眼窩**（眼球・外眼筋・脂肪。SIRマップ対象） |

- 眼窩スライスは眼球(globe)が非常に高信号で2つの compact blob → 自動検出に使える(`auto_orbit_and_brain_slices`)。

## STIRの信号順（白質抽出に直結）
**白質 < 灰白質 < CSF/浮腫/炎症**（STIR=脂肪抑制T2系、水・炎症が高信号）。
→ 白質は**低信号**。ヒストグラムでは低信号側のピーク。[[white-matter-histogram-extraction]]

## ⚠️ T2学習モデルはSTIRで動かない（ドメインシフト）
TOM500(T2冠状)で学習した外眼筋セグメンテーション(U-Net/DeepLabV3+/SegFormer)を**STIRに適用すると全background**（何も検出しない）。STIRはT2と信号特性が異なるため。
→ 結論: **外眼筋セグに依存しない眼窩SIRマップ**（voxel毎 SIR=信号/白質）が妥当。筋ROIは不要。

## predict.py の既知の罠
- PyTorch 2.6+ で `torch.load` の `weights_only` 既定がTrueに → チェックポイント読込が失敗。`weights_only=False`を渡す（CLAUDE.md記載）。
- `--model` に `vanilla_unet` 選択肢が無い（best重みは vanilla_unet 系が多い点に注意）。

## 環境
- 実行は必ず `MRI_TOM/venv`（python3.11.5 / torch2.8+cu126 / pydicom / transformers / ultralytics / smp / cv2 / scipy）。素のpythonには pydicom/torch 等が無い。
- SAM2/transformers は `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` でHFキャッシュ利用。
- 関連: [[stir-sir-pipeline]] [[brain-segmentation-distillation]]
