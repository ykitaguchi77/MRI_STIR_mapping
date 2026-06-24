# 大脳セグメンテーション: SAM2教師 → LWBNA-UNetへ蒸留

白質参照の前段で大脳を分離する。形態学では不可、SAM2で解決、軽量モデルへ蒸留。

## ⚠️ 形態学的skull-stripは失敗する
冠状STIRでは**脳と顔面(眼窩・副鼻腔)が頭蓋底で連結**し、組織マスク上で1つの塊になる。
- 円ROI/上部脳ROI/収縮コア+reconstruction いずれも、ROIが眼窩・副鼻腔へ漏れる（むしろ悪化）。
- binary_propagationは連結路を通って顔面へ再流入する。

## ✅ SAM2 が綺麗に分離
`facebook/sam2-hiera-large`(transformers, HFキャッシュ)。脳内部に**自動点プロンプト**(強収縮コアの重心)を置き、3候補マスクから「脳サイズ(面積2-45%)かつ上部重心」の最高スコアを選択。副鼻腔・眼窩を構造的に除外。
- `sam2_brain.py: SAM2BrainSegmenter.brain_mask(stir, z, strict=)`。strict=Trueで脳が無いスライスはNone（ラベリング用）。
- 実行時 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` でキャッシュ利用。

## 蒸留（SAM2は重いので軽量モデルへ）→ LWBNA-UNet
**ピクセルマスク画像 + LWBNA-UNet(セマンティックセグ)** に蒸留。`brain_mask(stir,z)` I/Fは
SAM2と共通で `orbital_sir_map.py --lwbna` から呼ぶ。val Dice≈0.978、輪郭が滑らか。

## SAM2ラベルのQC + 手動キュレーション（必須）
SAM2にも失敗マスクがある（whole-head/片側漏れ/断片化）。
1. `build_brain_seg_dataset.py`: 幾何QCで自動除外（面積0.04-0.40, 境界接触<3%, **solidity≥0.84**, 左右バランス≥0.25）。`--review`でオーバーレイ確認画像出力。
2. レビューフォルダを開いて**人手で不良を削除**。
3. `curate_brain_dataset.py`: 削除分を学習ペアから除去。**train/valは患者単位で再分割**（リーク厳禁）。
- 全344症例: QC採用3162→手動で約半数削除→1457→train1164/val293。

## 学習・推論
- `train_lwbna_brain.py`: Dice+BCE, AdamW, cosine, AMP, val Dice best保存。
- 重み: `brain_lwbna_best.pt`。白質参照はSAM2と一致(症例100で707、未使用症例でも汎化)。
- 関連: [[lwbna-unet-architecture]] [[stir-sir-pipeline]]
