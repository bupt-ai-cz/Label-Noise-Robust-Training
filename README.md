<h1> <p align="center"> Paper 1 </p> </h1>  

# Noise Robust Learning with Hard Example Aware for Pathological Image classification 
[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Codes%20and%20Data%20for%20Our%20Paper:%20"Noise%20Robust%20Learning%20with%20Hard%20Example%20Aware%20for%20Pathological%20Image%20classification"%20&url=https://github.com/bupt-ai-cz/Label-Noise-Robust-Training)

Implementation detail for our paper ["Noise Robust Learning with Hard Example Aware for Pathological Image classification"](https://ieeexplore.ieee.org/abstract/document/9344937), this code also includes further resaerch beyound this paper.

## Citation

Please cite this paper in your publications if it helps your research:

```
@inproceedings{peng2020noise,
  title={Noise Robust Learning with Hard Example Aware for Pathological Image classification},
  author={Peng, Ting and Zhu, Chuang and Luo, Yihao and Liu, Jun and Wang, Ying and Jin, Mulan},
  booktitle={2020 IEEE 6th International Conference on Computer and Communications (ICCC)},
  pages={1903--1907},
  year={2020},
  organization={IEEE}
}
```

## Dataset
DigestPath 2019:
https://digestpath2019.grand-challenge.org/Dataset/

Colorectal dataset (contributed by this paper):contains 4198 microscopy images, which are distributed as follows: adenoma, polyp, adenocarcinoma, gastrointestinal stromal tumor, and neuroendocrine tumor

## Envs
- Pytorch 1.0
- Python 3+
- cuda 9.0+

## Training
```
$ cd code/
# train label noise dataset and record training history
$ python iter_train.py --cached_data_file='pickle_data/digest_20.p'
# uncomment "detect label noise" code block in iter_train.py and apply label noise detect algorithm
$ python iter_train.py 
# label correction
$ python pre_iter.py
# train neural network on processed label noise dataset (apply different loss functions)
$ python train.py
# co-teaching training
$ python co-teaching.py
```

<h1> <p align="center"> Paper 2 </p> </h1>  

# Pathological Image Classification Based on Hard Example Guided CNN

For the implementation for our paper ["Pathological Image Classification Based on Hard Example Guided CNN"](https://ieeexplore.ieee.org/abstract/document/9119412), please refer to code/code_access/train_history.py 

## Citation
Please cite this paper in your publications if it helps your research:

```
@article{wang2020pathological,
  title={Pathological image classification based on hard example guided CNN},
  author={Wang, Ying and Peng, Ting and Duan, Jiajia and Zhu, Chuang and Liu, Jun and Ye, Jiandong and Jin, Mulan},
  journal={IEEE Access},
  volume={8},
  pages={114249--114258},
  year={2020},
  publisher={IEEE}
}
```

## Contact

* email：wkchen@bupt.edu.cn;czhu@bupt.edu.cn;2391207566@qq.com
* qq: 2391207566
