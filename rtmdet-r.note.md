
# Train

```bash
python tools/train.py configs/rotated_rtmdet/rotated_rtmdet_l-3x-dota_ms_custom.py
```

# Infer

```bash
python demo/image_demo.py ../data/oriented_bbox_labels/images2/images/frame10.jpg \
configs/rotated_rtmdet/rotated_rtmdet_l-3x-dota_ms_custom_v2.py \
rotated_rtmdet_l_3x_v2/epoch_36.pth \
--out-file result/result.jpg
```

# Evaluation

```bash
python tools/test.py configs/rotated_rtmdet/rotated_rtmdet_l-3x-dota_ms_custom.py \
rotated_rtmdet_l_3x_v1/epoch_36.pth
```

## 1. Train on images1

### 1.1. Evaluation on images1

2023/08/03 08:46:57
| class   | gts    | dets   | recall | ap     |
| ------- | ------ | ------ | ------ | ------ |
| paper   | 615    | 2114   | 0.951  | 0.908  |
| metal   | 208    | 1316   | 1.000  | 0.991  |
| plastic | 1006   | 2839   | 0.980  | 0.907  |
| nilon   | 127    | 1720   | 0.969  | 0.896  |
| glass   | 98     | 1199   | 0.990  | 0.902  |
| fabric  | 96     | 935    | 0.958  | 0.899  |
| ------  | ------ | ------ | ------ | ------ |
| mAP     |        |        |        | 0.917  |

### 1.2. Evaluation on images2

2023/08/03 08:46:57
| class   | gts    | dets   | recall | ap     |
| ------- | ------ | ------ | ------ | ------ |
| paper   | 758    | 2858   | 0.778  | 0.577  |
| metal   | 275    | 1904   | 0.985  | 0.872  |
| plastic | 1273   | 3983   | 0.939  | 0.852  |
| nilon   | 228    | 2243   | 0.754  | 0.456  |
| glass   | 85     | 2105   | 0.976  | 0.732  |
| fabric  | 97     | 1583   | 0.856  | 0.733  |
| ------  | ------ | ------ | ------ | ------ |
| mAP     |        |        |        | 0.704  |

## 2. Train on images2

### 2.1. Evaluation on images2

2023/08/03 09:33:03
| class   | gts    | dets   | recall | ap     |
| ------- | ------ | ------ | ------ | ------ |
| paper   | 758    | 2986   | 0.937  | 0.900  |
| metal   | 275    | 1386   | 1.000  | 0.984  |
| plastic | 1273   | 3407   | 0.968  | 0.908  |
| nilon   | 228    | 1946   | 0.996  | 0.908  |
| glass   | 85     | 1256   | 1.000  | 0.919  |
| fabric  | 97     | 1050   | 0.948  | 0.907  |
| ------- | ------ | ------ | ------ | ------ |
| mAP     |        |        |        | 0.921  |

### 2.2. Evaluation on images1

2023/08/03 09:35:03
| class   | gts    | dets   | recall | ap     |
| ------- | ------ | ------ | ------ | ------ |
| paper   | 615    | 2553   | 0.925  | 0.858  |
| metal   | 208    | 1193   | 0.981  | 0.842  |
| plastic | 1006   | 3205   | 0.940  | 0.863  |
| nilon   | 127    | 1624   | 0.913  | 0.680  |
| glass   | 98     | 1010   | 0.990  | 0.832  |
| fabric  | 96     | 924    | 0.906  | 0.792  |
| ------- | ------ | ------ | ------ | ------ |
| mAP     |        |        |        | 0.811  |
