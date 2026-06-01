# BaseStation

## 研究目的
本研究の最終目的は、電界強度推定AIに必要となる基地局位置情報を自動取得することである。
そのために、道路上視点のパノラマ画像を用いて、基地局候補を探索・検出し、将来的には位置推定してGIS上へ出力する。

## 研究の考え方
人間がストリートビューを見ながら基地局を探すときの行動を模倣する。

- 道路に沿って進む
- 正面を中心に左右を確認する
- 怪しい対象があればズームや再確認を行う

本研究では、この探索行動をエージェントとして実装する。

## 全体フロー
1. ArcGIS Pro で AOI（探索対象領域）をポリゴンとして作成する
2. AOI 内の Panoramax 撮影地点を API から取得する
3. 撮影地点を道路順に並べる
4. パノラマ画像を取得する
5. パノラマ画像から、実カメラ画像に近い透視投影画像を生成する
6. 必要に応じて yaw 補正を行い、画像の傾きや進行方向のずれを補正する
7. YOLO を用いて基地局候補を検出する
8. 正面・右・左を確認し、怪しい候補があればズームなどで再探索する
9. 検出結果を保存する
10. 将来的には検出対象の位置を推定し、ArcGIS に表示する

## 現在できていること
- AOI の作成
- Panoramax API からの撮影地点取得
- 撮影地点の順序決定
- パノラマ画像取得
- パノラマからの透視投影画像生成
- yaw 補正
- YOLO による基地局候補検出
- 検出信頼度に基づく再探索

## 全天球カメラの座標系
`tools/agent/spherical_camera.py` に全天球画像処理を集約している。

- 角度は度で扱う
- 3D 座標は x が右、y が上、z が前
- yaw は右向きが正、pitch は上向きが正、roll は光軸まわり
- 画像座標は OpenCV と同じく u が右、v が下
- 透視投影 crop の bbox は camera ray に逆投影し、現在の yaw/pitch/roll で world ray へ変換する
- 再探索時は bbox 中心の world ray から次の yaw/pitch を直接求め、bbox 四隅の ray との最大角度から安全な FOV を決める

既存 CLI の `--pitch_cli 40` は「上向き 40 度」を表す。内部でも pitch は上向きを正として扱う。

## エージェントパッケージ構成
エージェント中核処理は `tools/agent/` へ集約している。`tools/basestation_agent_complete.py` は
parser と `main()` だけを持つ薄い CLI 入口で、stage 本体は `tools.agent.*` を呼び出す。
`tools/agent_detect_only_agent2.py` などの旧スクリプトは互換用 wrapper として残し、旧本体は
`tools/archive/` に退避している。

- `tools/agent/config.py`: `AgentConfig`
- `tools/agent/io_utils.py`: JSONL/JSON/CSV、path、ログ補助、stage 共通I/O
- `tools/agent/spherical_camera.py`: 全天球画像、3Dレイ、透視投影
- `tools/agent/crop.py`: crop名生成、crop生成、視点設定
- `tools/agent/detector.py`: YOLO推論、検出結果補助
- `tools/agent/refine_policy.py`: 再探索方針、bbox面積比ベースのFOV制御
- `tools/agent/visualize.py`: 検出描画、status、compare画像
- `tools/agent/panoramax_client.py`: Panoramax points/images 取得、pano download stage
- `tools/agent/ordering.py`: sequence/datetime/nearest による pano ordering stage
- `tools/agent/yaw.py`: yaw center 推定と yaw map 生成
- `tools/agent/pipeline.py`: crop 生成、既存 crop 検出、pano からの再探索検出 stage
- `tools/agent/geolocation.py`: bbox中心から検出方向のローカル yaw/pitch と地理方位を計算
- `tools/agent/gis_export.py`: ArcGIS確認用の camera points / detection rays GeoJSON 出力

## ArcGIS確認用出力
検出済みの `detections.jsonl` と `aoi_index.jsonl` から、撮影地点、検出方向線、確認用 observation point、
annotated 画像添付テーブルを出力できる。
`view_azimuth` がある場合は `view_azimuth + local_yaw` を地理方位として使い、ない場合は
`azimuth_source=local_yaw_fallback` として local yaw を仮方位にする。
Google My Maps 由来の基地局点やGoogle写真とは関連付けない。

```bash
python tools/basestation_agent_complete.py \
  --run_dir runs/test_50_agent \
  --ordered_index runs/test_50_prepare/aoi_index_50.jsonl \
  --skip_fetch_points \
  --skip_fetch_images \
  --skip_order_panos \
  --skip_download_panos \
  --skip_make_crops \
  --skip_detect \
  --export_geojson \
  --ray_length_m 100 \
  --arcgis_annotated_dir data/arcgis_detection_annotated \
  --arcgis_windows_annotated_dir "C:\Users\kajiura\Desktop\arcGIS_data\detection_annotated"
```

出力:
- `run_dir/geo/camera_points.csv`
- `run_dir/geo/camera_points.geojson`
- `run_dir/geo/detection_rays.csv`
- `run_dir/geo/detection_rays.geojson`
- `run_dir/geo/detection_observation_points.csv`
- `run_dir/geo/detection_observation_points.geojson`
- `run_dir/geo/detection_annotated_attachments.csv`
- `run_dir/geo/detection_annotated_attachments_windows.csv`

`detection_rays.csv` には `conf`、`conf_class`、`refine_status`、`is_refined`、`ray_id`、
`annotated_path` を出力する。`conf_class` は `high`、`medium`、`low` で、初期しきい値は
`high >= 0.60`、`medium >= 0.30`。`refine_status` は初回cropを `initial`、再探索後cropを
`refined`、判定不能を `unknown` とする。

`--arcgis_annotated_dir` を指定した場合だけ annotated 画像をコピーし、コピー名は
`ray_id_annotated.jpg` にする。未指定の場合は既存の annotated 画像パスを添付CSVに出力する。

ArcGIS Proでの確認手順:
1. `detection_rays.csv` を XY To Line で線にする
2. `conf_class` で線色を分類する
3. `detection_observation_points.csv` を XY Table To Point で点にする
4. `refine_status` または `is_refined` で点記号を分類する
5. `detection_observation_points` に Enable Attachments を実行する
6. Add Attachments で `detection_annotated_attachments_windows.csv` を使う
7. ポップアップで Attachments を Preview 表示する

## 現在の問題
現在使用している YOLO 重みは国外データで学習されたものであり、日本国内の基地局に対してはドメインギャップがある。
そのため、日本の基地局画像を用いた学習データセットを作成し、再学習または微調整する必要がある。

## 最初に取り組む課題
最初の課題は、`tools/make_yolo_crops_from_panoramax.py` を改良して、
パノラマ画像からできるだけ多くの学習用画像を、画素数を落とさず、実カメラ画像に近い形で生成できるようにすることである。

特に以下を重視する。

- 画像枚数を増やす
- 画質をできるだけ維持する
- 斜め画像を補正する
- 日本の基地局学習に適した見え方を作る

## 将来の課題
- 日本国内データで YOLO を再学習する
- 基地局位置を推定する
- ArcGIS 上に結果を可視化する
- 電界強度推定モデルへの入力データとして利用する

## 主要スクリプト
- `tools/panoramax_fetch_points_in_aoi.py`
- `tools/fetch_panos_ordered.py`
- `tools/make_yolo_crops_from_panoramax.py`
- `tools/agent_detect_only_agent2.py`
