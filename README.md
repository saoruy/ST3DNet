# ST3DNet

Deep Spatial–Temporal 3D Convolutional Neural Networks for Trafﬁc Data Forecasting

<img src="fig/ST3DNet architecture.png" alt="image-20200103164326338" style="zoom:50%;" />

# Reference

```latex
@article{guo2019deep,
  title={Deep Spatial-Temporal 3D Convolutional Neural Networks for Traffic Data Forecasting},
  author={Guo, Shengnan and Lin, Youfang and Li, Shijie and Chen, Zhaoming and Wan, Huaiyu},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2019},
  publisher={IEEE}
}
```

# Datasets

BikeNYC is one of the datasets we used in the paper, it suffices to reproduce the results what we have reported in the paper.

Step 1: Download **BikeNYC** dataset provided by [DeepST](https://github.com/lucktroy/DeepST/tree/master/data/BikeNYC)  

Step 2: process dataset

```shell
python prepareData.py
```

# Train and Test

```shell
python trainNY.py
```

