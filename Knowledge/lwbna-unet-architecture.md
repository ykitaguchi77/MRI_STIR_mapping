# LWBNA-UNet アーキテクチャ

Lightweight Bottleneck Narrowing with Attention U-Net。
Sharma et al., Scientific Reports 12:8508 (2022)。
参照実装(Keras): github.com/parmanandsharma/Lightweight_AI。
本リポジトリのPyTorch移植: `lwbna_unet.py`（2.95M params）。

## 設計の要点（通常のU-Netとの違い）
| 項目 | LWBNA-UNet |
|---|---|
| チャネル幅 | **全層 f=128 固定**（深さで増やさない＝軽量化） |
| チャネルアテンション | 各convブロック後に SE型: `x * sigmoid(relu(W·GAP(x)))` |
| skip接続 | **concatでなく add** |
| アップサンプル | 転置畳み込みでなく **UpSampling(interpolate)** |
| ボトルネック | **channel narrowing**: 128→64→32→16 各段でアテンション→128へ戻して最初の特徴をadd |

## 構造（depth=4, 入力256→ボトルネック16）
1. Encoder ×4: ConvBlock(128,attn)→MaxPool→Dropout、skip保存。
2. Mid(narrowing): Conv 128→64→32→16（各Attention）→Conv 16→128→ +xe1(最初の128)。
3. ConvBlock(128,attn)。
4. Decoder ×4: Upsample→ +skip(add)→Dropout→ConvBlock(128,attn)。
5. Head: Conv→logits（num_classes=1）。

## 学習設定（本タスク: 大脳バイナリセグ）
- 損失 Dice+BCE、AdamW(lr1e-3, wd1e-4)、CosineAnnealing、AMP、imgsz256。
- 入力1ch(STIRグレースケール /255)、出力1ch。
- 早期終了(patience25)。本データで **val Dice=0.978**（epoch57）。
- 入力サイズは16の倍数必須（MaxPool×4で/16）。
- 関連: [[brain-segmentation-distillation]] [[stir-sir-pipeline]]
