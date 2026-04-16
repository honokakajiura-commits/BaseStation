# 研究概要
車載画像と電界強度データから基地局候補を推定する研究

# データ
- data/: AOIやgeojson
- runs/: 実験結果、YOLO結果
- tools/: スクリプト群

# 主な処理の流れ
1. AOIからパノラマポイント取得
2. 画像取得
3. YOLOで検出
4. クロップ生成
5. 推定・解析

# 重要ファイル
- tools/agent_detect_only_agent2.py : 検出のメイン
- tools/make_yolo_crops_from_panoramax.py : クロップ生成

# 現在の課題
- 精度が低い
- 処理の流れが分かりづらい
- 推定部分の整理