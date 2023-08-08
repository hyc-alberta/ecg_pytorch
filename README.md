# ecg_pytorch
This project is a personal practice to enhance coding skills. Please refer to [this page](https://github.com/awni/ecg) for original essay and code.
The original work is based on Keras, and this one is based on Pytorch. The code for data generation and data processing is mostly taken from
[this repository](https://github.com/lxdv/ecg-classification), and you can refer to it for more details. The dataset used here is mit-bih.
Config.json contains most of important information of the model. The version of pytorch here is 1.13.

Below are some rough results of a model trained for 20 epochs.

Train class accuracy:

![Train CLASS accuracy](https://github.com/hyc481/ecg_pytorch/assets/141563901/b7259b31-7dd9-48f8-9a06-95a75fa084eb)


Train loss:

![Train loss (epochs)](https://github.com/hyc481/ecg_pytorch/assets/141563901/8a688d7d-a50a-4b2a-952a-3d9eb06e5afa)


Validation class accuracy:

![Validation CLASS accuracy](https://github.com/hyc481/ecg_pytorch/assets/141563901/ccf38524-f3a3-4cef-9b78-dcfa2c76ab82)


Validation loss:

![Validation loss](https://github.com/hyc481/ecg_pytorch/assets/141563901/e7530869-4ea9-4d21-9e72-b71223dd4082)
