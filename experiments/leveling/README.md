# EXPERIMENTAL / horizontal leveling validation

このディレクトリは水平補正の検証用です。crop と元パノラマの点対応、および水平補正あり/なしの見え方を比較します。
本体 pipeline には影響しません。削除してよい実験用ファイルだけを置きます。

## 実行例

```bash
python experiments/leveling/exp_level_auto_points.py \
  --pano runs/full_test_TMU_east_best2/panos/任意の画像.jpg \
  --out_dir outputs/experiments/leveling/sample01 \
  --yaw 0 \
  --pitch 40 \
  --fov 105 \
  --save_level_debug
```

`spherical_camera.py` では pitch は正が上向きなので、この検証では `--pitch 40` を使います。

## 見るべき出力

- `crop_no_level.jpg`: 水平補正なし crop
- `crop_level.jpg`: 水平補正あり crop
- `crop_level_points.jpg`: 自動特徴点を重ねた補正あり crop
- `pano_projected_points.jpg`: crop 上の特徴点を元パノラマへ戻した結果
- `pano_zoom_points.jpg`: 投影点周辺の拡大
- `comparison.jpg`: 補正なし/補正あり/点対応の比較
- `auto_points_mapping.json`: crop 座標、パノラマ座標、roll 推定 metadata
- `level_debug/preview_yaw_*_lines.jpg`: roll 推定に使った Hough 線分の分類

`level_debug` の線分色は、水平候補が赤、垂直候補が青、その他が灰色です。

## ランダム20枚で front / left / right を確認

```bash
mapfile -t panos < <(find runs/full_test_TMU_east_best2/panos -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | shuf -n 20)

for i in "${!panos[@]}"; do
  pano="${panos[$i]}"
  n="$(printf "%02d" "$((i + 1))")"
  for spec in front:0 left:-90 right:90; do
    view="${spec%%:*}"
    yaw="${spec##*:}"
    python experiments/leveling/exp_level_auto_points.py \
      --pano "$pano" \
      --out_dir "outputs/experiments/leveling/random20/${n}_${view}" \
      --yaw "$yaw" \
      --pitch 40 \
      --fov 105 \
      --save_level_debug
  done
done
```

## EXPERIMENTAL: left/right の消失点ベース水平補正

`exp_vanishing_level.py` は本体 pipeline から独立した実験用スクリプトです。
left/right crop の補正なし画像から HoughLinesP の線分を取り、進行方向側の消失点を RANSAC で推定します。
推定した `vp_y` を期待される `horizon_y = H/2 + f * tan(pitch)` に近づける roll 候補を計算し、適用値は `±2.5` 度に制限します。

random20 の left/right だけを確認する例:

```bash
python experiments/leveling/exp_vanishing_level.py \
  --pano_dir runs/full_test_TMU_east_best2/panos \
  --yaw_map runs/full_test_TMU_east_best2/yaw_map.jsonl \
  --out_dir outputs/experiments/leveling/vp_random20_left_right \
  --random 20 \
  --views left,right \
  --pitch 40 \
  --front_fov 105 \
  --side_fov 90 \
  --seed 0
```

単発で確認する例:

```bash
python experiments/leveling/exp_vanishing_level.py \
  --pano runs/full_test_TMU_east_best2/panos/任意の画像.jpg \
  --out_dir outputs/experiments/leveling/vp_sample01 \
  --yaw_center 0 \
  --views left,right
```

主な出力:

- `crop_no_level.jpg`: VP 推定に使う補正なし crop
- `crop_level.jpg`: 既存の Hough angle ベース水平補正 crop
- `crop_vp_level.jpg`: 消失点ベース roll を適用した crop
- `vp_lines.jpg`: Hough 線分、候補、inlier の可視化
- `vp_debug.jpg`: 延長線、推定 vp、期待 horizon、dy、roll/confidence の可視化
- `comparison.jpg`: `crop_no_level` / `crop_level` / `crop_vp_level` / `vp_debug` の横並び
- `vp_level_meta.json`: `view`, `yaw`, `pitch`, `fov`, `vp_x`, `vp_y`, `horizon_y`, `dy`, `estimated_roll_deg`, `applied_roll_deg`, `confidence`, `line_count`, `inlier_count`, `residual` など
- `vp_level_index.json`: batch 全体の index

## EXPERIMENTAL: spherical upright adjustment 簡易再現

`exp_spherical_upright_level.py` は Jung et al. "Upright Adjustment of 360 Spherical Panoramas" に触発された実験用スクリプトです。
完全再現ではなく、equirectangular パノラマから複数 yaw の preview crop を作り、HoughLinesP の線分を球面上の great circle に戻して、great circle 法線群から up vector を RANSAC + SVD で推定する簡易版です。
本体 pipeline には影響しません。

単発 smoke test:

```bash
python experiments/leveling/exp_spherical_upright_level.py \
  --pano runs/full_test_TMU_east_best2/panos/任意の画像.jpg \
  --out_dir outputs/experiments/leveling/spherical_smoke \
  --yaw_center 0
```

selected20 batch:

```bash
python experiments/leveling/run_spherical_upright_selected20.py
```

主な出力:

- `preview_debug/preview_yaw_*.jpg`: up vector 推定用の preview crop
- `preview_debug/preview_yaw_*_lines.jpg`: 検出線分。inlier は橙、その他は灰
- `preview_debug/preview_yaw_*_lines.json`: preview ごとの線分、great circle normal、residual、inlier
- `great_circles.jsonl`: 全 preview の great circle 情報
- `upright_meta.json`: `v_up`, `angle_to_world_up_deg`, `applied`, `reject_reason`, `R_level`, `R_level_inverse`, residual 統計
- `front|left|right/crop_no_level.jpg`: 補正なし crop
- `front|left|right/crop_spherical_level.jpg`: 推定 `R_level` 方向の crop
- `front|left|right/crop_spherical_level_inv.jpg`: 逆向き `R_level_inverse` の crop
- `comparison_front|left|right.jpg`: 補正なし / `R_level` / `R_level_inverse` の横並び

`R_level` と `R_level_inverse` は向きの確認用に両方出力します。
`upright_meta.json` の `applied=false` は、推定角が `--max_apply_deg` を超えた、または inlier 不足で自動適用すべきではないという意味です。
比較画像は convention 確認のため、reject 時でも候補 crop を保存します。

完全再現との差分:

- Atlanta world の水平・垂直方向クラスタ最適化は未実装
- 線分分類は明示的に水平/垂直へ分けず、great circle 法線群の RANSAC で dominant up candidate を探すだけ
- equirectangular 上の線検出ではなく、複数 perspective preview crop 上の HoughLinesP を球面へ戻す
- 最適化は `min ||N v||, ||v||=1` の SVD refinement に限定
- 安全のため、推定角が大きい場合は `applied=false` として扱う

## EXPERIMENTAL: spherical level point mapping validation

`exp_spherical_point_mapping.py` は、`crop_spherical_level` 上の任意点を全天球パノラマへ正しく逆投影できるかを検証するスクリプトです。
`exp_spherical_upright_level.py` で推定した `R_level` を使い、点の `crop -> pano -> crop` の round-trip 誤差を測ります。
`R_level_inverse` は比較用途だけに残し、基本の確認対象は `R_level` です。

単発 smoke test:

```bash
python experiments/leveling/exp_spherical_point_mapping.py \
  --pano runs/full_test_TMU_east_best2/panos/任意の画像.jpg \
  --out_dir outputs/experiments/leveling/spherical_point_mapping_smoke \
  --yaw 0 \
  --yaw_center 0
```

selected20 batch:

```bash
python experiments/leveling/run_spherical_point_mapping_selected20.py
```

主な出力:

- `crop_no_level.jpg`: 補正なし crop
- `crop_spherical_level.jpg`: `R_level` を使った crop
- `crop_points_original.jpg`: 検証点の元位置
- `crop_points_roundtrip.jpg`: round-trip 後の再投影点
- `pano_projected_points.jpg`: パノラマ上の対応点
- `pano_zoom_points.jpg`: 対応点周辺の拡大
- `comparison.jpg`: 原点/round-trip/pano の横並び比較
- `point_mapping_meta.json`: `v_up`, `R_level`, `points`, `roundtrip_error_px`, `PASS/WARN/FAIL` を保存

判定基準:

- `PASS`: `error_mean_px <= 2.0` かつ `error_max_px <= 5.0`
- `WARN`: `error_mean_px <= 5.0` かつ `error_max_px <= 10.0`
- `FAIL`: それ以外

これは agent に `R_level` を組み込む前の安全確認で、点対応が破綻していないかを見るための実験です。

## EXPERIMENTAL: spherical upright outlier ablation

`exp_spherical_outlier_ablation.py` は、great circle からの up vector 推定に対して outlier handling の有効性を比べる実験用スクリプトです。
`no_outlier_handling` / `ransac_inliers` / `robust` の 3 方式を並べ、木・電線・道路斜線などが混ざったときに誤補正しないかを確認します。
本体 pipeline には影響しません。

単発 smoke test:

```bash
python experiments/leveling/exp_spherical_outlier_ablation.py \
  --pano runs/full_test_TMU_east_best2/panos/任意の画像.jpg \
  --out_dir outputs/experiments/leveling/spherical_outlier_smoke \
  --yaw_center 0
```

selected20 batch:

```bash
python experiments/leveling/run_spherical_outlier_ablation_selected20.py
```

主な出力:

- `front|left|right/comparison.jpg`: `no_level` / `no_outlier_handling` / `ransac_inliers` / `robust` の横並び
- `line_debug/preview_yaw_*_all_lines.jpg`: preview 上の全線分
- `line_debug/preview_yaw_*_inlier_lines.jpg`: robust の inlier
- `line_debug/preview_yaw_*_outlier_lines.jpg`: robust の outlier
- `upright_meta_ablation.json`: 3 方式の `applied`, `reject_reason`, `v_up`, `angle_to_world_up_deg`, `inlier_count`, residual 統計
- `summary.json`: method 別の成功/失敗/reject 理由

`robust` は長い線分を強めに扱い、水平に近い線分を弱めます。`angle_to_world_up_deg` が `--max_apply_deg` を超える場合は補正を適用しません。

batch 実行後は `upright_ablation_index.json` を見れば、method 別の `applied_count`, `reject_count`, `mean/median residual`, `angle_to_world_up_deg` 平均を確認できます。

## 削除方法

```bash
rm -rf experiments/leveling
```
