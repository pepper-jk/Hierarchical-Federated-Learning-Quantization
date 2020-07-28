# Hierarchical Federated-Learning (PyTorch)

Implementation of both hierarchical and vanilla federated learning based on the paper : [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629).

Experiments are conducted on MNIST and CIFAR10 datasets. During training, the datasets split are both IID and non-IID. In case of non-IID, the data amongst the users can be split equally or unequally.

Since the purpose of these experiments is to illustrate the effectiveness of the federated learning paradigm, only simple models such as MLP and CNN are used.

## Requirements
Install all the packages from requirements.txt
* Python==3.7.3
* Pytorch==1.2.0
* Torchvision==0.4.0
* Numpy==1.15.4
* Tensorboardx==1.4
* Matplotlib==3.0.1
* Tqdm==4.39.0

## Steps to setting up a Python environment
1. Creating environment:
```
conda create -n myenv python=3.7.3
```
2. Installing Pytorch and torchvision:
```
conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
```
3. Installing other package requirements:
```
pip install -r requirements.txt
```


## Data
* Download train and test datasets manually or they will be automatically downloaded to the [data](/data/) folder from torchvision datasets.
* Experiments are run on MNIST and CIFAR.
* To use your own dataset: Move your dataset to data directory and write a wrapper on pytorch dataset class.

## Running the experiments
All the experiments of reported results are in the [scripts](/src/) below:
* script_bash_FL_diffFP_mnist_mlp.sh
* script_bash_FL_diffFP_mnist_cnn.sh
* script_bash_FL_diffFP_cifar.sh
* script_bash_FL_diffFP.sh
-----
The baseline experiment trains the model in the conventional federated learning.

* To run the baseline federated experiment with MNIST on MLP using CPU:
```
python federated_main.py --local_ep=1 --local_bs=10 --frac=0.1 --model=mlp --dataset=mnist --iid=1 --learning_rate=0.01 --test_acc=95 --mlpdim=200 --epochs=600
```
* Or to run it on GPU (eg: if gpu:0 is available):
```
python federated_main.py --local_ep=1 --local_bs=10 --frac=0.1 --model=mlp --dataset=mnist --iid=1 --gpu --learning_rate=0.01 --test_acc=95 --mlpdim=200 --epochs=600
```
-----

Hierarchical federated experiment involves training a global model using many local models.

* To run the hierarchical federated experiment with 2 clusters on MNIST using CNN (IID):
```
python federated-hierarchical_main.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=cnn --dataset=mnist --iid=1 --num_cluster=2 --gpu --learning_rate=0.01 --epochs=100
```
* To run the same experiment under non-IID condition:
```
python federated-hierarchical_main.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=cnn --dataset=mnist --iid=0 --num_cluster=2 --gpu --learning_rate=0.01 --epochs=100
```
-----
Hierarchical Federated experiments involve training a global model using different clusters with many local models (16-bit).

* To run the hierarchical federated experiment with 2 clusters on CIFAR using CNN (IID):
```
python ./federated-hierarchical_main.py --floating_point_16 --local_ep=5 --local_bs=50 --frac=0.1 --Cepochs=10 --model=cnn --dataset=cifar --iid=1 --num_cluster=2 --gpu --learning_rate=0.01 --epochs=100
```


You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: 'mnist'. Options: 'mnist', 'cifar'
* ```--model:```    Default: 'mlp'. Options: 'mlp', 'cnn'
* ```--gpu:```      Default: False. Set to use cuda.
* ```--gpu_id:```	Default: 'cuda:0' (this specifies which GPU to use)
* ```--epochs:```   Number of rounds of training.
* ```-lr, --learning_rate:``` Learning rate set to 0.01 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.
* ```--seed:```     Random Seed. Default set to 1.

#### Federated Parameters
* ```--iid:```      Distribution of data amongst users. Default set to IID. Set to 0 for non-IID.
* ```--num_users:```Number of users. Default is 100.
* ```--frac:```     Fraction of users to be used for federated updates. Default is 0.1.
* ```--local_ep:``` Number of local training epochs in each user. Default is 1.
* ```--local_bs:``` Batch size of local updates in each user. Default is 10.
* ```--num_clusters:```  Number of clusters in the hierarchy.
* ```--Cepochs:```  Number of rounds of training in each cluster.

## Experimental Results
The results and figures can be found in [evaluation notebooks](/src/)
* Eval.ipynb
* Eval_fp16.ipynb
* Eval_fp16-32-compare.ipynb




