# LaplacianFusion

This repository contains the accompanying code for [LaplacianFusion: Detailed 3D Clothed-Human Body Reconstruction, SIGGRAPH Asia'22]()

## Author Information

- [Hyomin Kim](https://hyomin.me/)
- [Hyeonseo Nam](https://jgkim.info/)
- [Jungeon Kim]()
- [Jaesik Park](http://jaesik.info/) [[Google Scholar]](https://scholar.google.com/citations?user=_3q6KBIAAAAJ&hl=en&oi=ao)
- [Seungyong Lee](http://cg.postech.ac.kr/leesy/) [[Google Scholar]](https://scholar.google.com/citations?user=yGPH-nAAAAAJ&hl=en&oi=ao)

## Run LaplacianFusion
### Prerequisites

- Ubuntu 18.06 or higher
- CUDA 10.2 or higher
- pytorch 1.9 or higher
- python 3.9 or higher

#
### Download data
- Get sample data and pre-trained 'DVM' weight (111 markers) from [here](https://drive.google.com/drive/folders/1A49Oef_UzqLbBm5UsjtQW5BIS8VGrAiW?usp=share_link)
- Get SMPL-X model from [here](https://smpl-x.is.tue.mpg.de/) and make data directory structure as follow:
```
lapfu
├── dvm_weight.pth
├── human_models
│   └── smplx
│       ├── SMPLX_FEMALE.npz
│       └── SMPLX_MALE.npz
├── protocol_info
└── subjects
```
- Get SMPL-X code from [here](https://github.com/vchoutas/smplx/tree/main/smplx) and replace **"./lib/smplx/"** folder  
#
### SMPL-X fitting (using Deep Virtual Markers)
- We recommend using docker
- Replace **DATADIR** in "run_dvm.sh: Line 4" as your path
```
docker pull min00001/dvm_run
./run_dvm.sh
```
- You can also use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) feature if color images are given (we already include keypoints and intrinsic parameters in sample dataset)
#
### LaplacianFusion
- Replace **DataPath.Main** in "config.py: Line 8" as your path
```
conda create -n lapfu python=3.9
conda activate lapfu
pip install -r ./requirements.txt

python ./preprocessing/fit_smplx.py
./script/run_learning.sh
```


## License
This software is being made available under the terms in the [LICENSE](LICENSE) file.

Any exemptions to these terms requires a license from the Pohang University of Science and Technology.

## Citing LaplacianFusion

```
@inproceedings{Kim_LaplacianFusion_SIGGRAPH_Asia_2022,
Title={LaplacianFusion: Detailed 3D Clothed-Human Body Reconstruction},
Author={Hyomin Kim and Hyeonseo Nam and Jungeon Kim and Jaesik Park and Seungyong Lee},
Booktitle={Proceedings of the ACM (SIGGRAPH Asia)},
Year={2022}
}
```