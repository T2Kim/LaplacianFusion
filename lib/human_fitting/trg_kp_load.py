import numpy as np
import torch
import json

def get_keypts(path, key, intrinsic):
    with open(path) as fp:
        data = json.load(fp)
    if len(data['people']) == 0:
        return None
    else:
        keypoints = np.array(data['people'][0][key]).reshape(-1, 3)
        keypoints[:, 0] = (keypoints[:, 0] - intrinsic["cx"]) / intrinsic["fx"]
        keypoints[:, 1] = (keypoints[:, 1] - intrinsic["cy"]) / intrinsic["fy"]
        keypoints[:, 2] = np.where(keypoints[:, 2] > 0.1, 1, 0)
        return keypoints

# 111 sparse markers
smplx_dvm_sparse = torch.tensor(np.array([2868 ,2011 , 195 ,9010 ,2226 ,3059 ,2002 ,1976 ,2039 ,8967 \
                            ,3029 ,2955 ,2132 ,8809 ,1898 ,3354 ,1934 ,5529 ,6602 ,6175 \
                            ,6677 ,7179 ,3340 ,4430 ,3935 ,6699 ,6649 ,6776 ,7040 ,7107 \
                            ,7039 ,6921 ,6958 ,6952 ,7591 ,7455 ,7461 ,7716 ,5399 ,3901 \
                            ,3258 ,4302 ,4383 ,4389 ,4177 ,4190 ,4210 ,4583 ,4900 ,4539 \
                            ,5023 ,3855 ,6054 ,6011 ,8155 ,6158 ,5422 ,3248 ,3288 ,3970 \
                            ,6072 ,6222 ,8221 ,5493 ,5520 ,5504 ,3309 ,5600 ,8721 ,6706 \
                            ,8367 ,5557 ,5674 ,3958 ,4112 ,6349 ,6384 ,6745 ,6400 ,6386 \
                            ,6409 ,6486 ,6482 ,6749 ,6526 ,6907 ,6753 ,8613 ,8566 ,8686 \
                            ,8598 ,8634 ,8547 ,3603 ,3623 ,3998 ,3639 ,3625 ,3648 ,3725 \
                            ,3721 ,3810 ,3768 ,4163 ,3753 ,5919 ,5906 ,8898 ,5903 ,8846 ,5775])).long()

