
# Setup Environment

## Install require packages 1.x branch

```bash
conda create --name mmrotate1x python=3.8 -y
conda activate mmrotate1x

conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc2"

mim install 'mmdet>=3.0.0rc2'

cd mmrotate
pip install -v -e .
```

## Verify installation

```bash
mim download mmrotate --config oriented-rcnn-le90_r50_fpn_1x_dota --dest .

python demo/image_demo.py demo/demo.jpg oriented-rcnn-le90_r50_fpn_1x_dota.py \
oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --out-file result.jpg
```

## Packages version

```bash
# check version
conda list | grep '^mm\|torch\|cuda' 
```

### 1.x branch

```bash
cudatoolkit               10.2.89              hfd86e86_1  
ffmpeg                    4.3                  hf484d3e_0    pytorch
mmcv                      2.0.1                    pypi_0    pypi
mmdet                     3.1.0                    pypi_0    pypi
mmengine                  0.8.3                    pypi_0    pypi
mmrotate                  1.0.0rc1                  dev_0    <develop>
pytorch                   1.8.0           py3.8_cuda10.2_cudnn7.6.5_0    pytorch
torchvision               0.9.0                py38_cu102    pytorch

```

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

### 1.1. Evaluate on images1

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

### 1.2. Evaluate on images2

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

### 2.1. Evaluate on images2

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

# Analysis Data

## Evaluate on pickle data

Firstly, create pkl (pickle) format file from prediction results
pkl file have format:

```python
List[Dict[
    "img_id": str,
    "pred_instances": Dict[
        "bboxes": Tensor[N, 5],
        "score": Tensor[N, 5],
        "labels": Tensor[N, 5], 
    ],
    "ignored_instances": Dict[
        "bboxes": Tensor[K, 5],
        "labels": Tensor[K, 5],
    ],
    "gt_instances": Dict[
        "bboxes": Tensor[M, 5],
        "labels": Tensor[M, 5]
    ]
]]
```

Example:

```python
    def _create_dummy_data_sample(self):
        bboxes = np.array([[23, 31, 10.0, 20.0, 0.0],
                           [100, 120, 10.0, 20.0, 0.1],
                           [150, 160, 10.0, 20.0, 0.2],
                           [250, 260, 10.0, 20.0, 0.3]])
        labels = np.array([0] * 4)
        bboxes_ignore = np.array([[0] * 5])
        labels_ignore = np.array([0])
        pred_bboxes = np.array([[23, 31, 10.0, 20.0, 0.0],
                                [100, 120, 10.0, 20.0, 0.1],
                                [150, 160, 10.0, 20.0, 0.2],
                                [250, 260, 10.0, 20.0, 0.3]])
        pred_scores = np.array([1.0, 0.98, 0.96, 0.95])
        pred_labels = np.array([0, 0, 0, 0])
        return [
            dict(
                img_id='P2805__1024__0___0',
                gt_instances=dict(
                    bboxes=torch.from_numpy(bboxes),
                    labels=torch.from_numpy(labels)),
                ignored_instances=dict(
                    bboxes=torch.from_numpy(bboxes_ignore),
                    labels=torch.from_numpy(labels_ignore)),
                pred_instances=dict(
                    bboxes=torch.from_numpy(pred_bboxes),
                    scores=torch.from_numpy(pred_scores),
                    labels=torch.from_numpy(pred_labels)))
        ]
```

## Note

===============================================================================================

- Please read this [documentation](https://mmrotate.readthedocs.io/en/latest/intro.html#rotation-direction) if you confuse about angles (it is 5th value in each detection values)

- Make sure `angle` is right format

===============================================================================================

```bash
python tools/analysis_tools/confusion_matrix.py \
result1.pkl \
rotated_rtmdet_l_3x_v4_test
```
