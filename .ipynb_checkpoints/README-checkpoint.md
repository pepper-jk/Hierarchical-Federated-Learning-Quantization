# Federated-Learning (PyTorch)

Implementation of both hierarchical and vanilla federated learning based on the paper : [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629).
Blog Post: https://ai.googleblog.com/2017/04/federated-learning-collaborative.html

Experiments are conducted on MNIST and CIFAR10 datasets. During training, the datasets split are both IID and non-IID. In case of non-IID, the data amongst the users can be split equally or unequally.

Since the purpose of these experiments are to illustrate the effectiveness of the federated learning paradigm, only simple models such as MLP and CNN are used.

## Requirments
Install all the packages from requirments.txt
* Python=3.7.3
* Pytorch=1.2.0
* Torchvision=0.4.0
* Numpy=1.15.4
* Tensorboardx=1.4
* Matplotlib=3.0.1


## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on Mnist and Cifar.
* To use your own dataset: Move your dataset to data directory and write a wrapper on pytorch dataset class.

## Running the experiments
The baseline experiment trains the model in the conventional way.

* To run the baseline experiment with MNIST on MLP using CPU:
```
python baseline_main.py --model=mlp --dataset=mnist --epochs=10
```
* Or to run it on GPU (eg: if gpu:0 is available):
```
python baseline_main.py --model=mlp --dataset=mnist --gpu=1 --epochs=10
```
-----

Federated experiment involves training a global model using many local models.

* To run the federated experiment with CIFAR on CNN (IID):
```
python federated_main.py --local_ep=1 --local_bs=10 --frac=0.1 --model=cnn --dataset=cifar --iid=1 --test_acc=99 --gpu=1
```
* To run the same experiment under non-IID condition:
```
python federated_main.py --local_ep=1 --local_bs=10 --frac=0.1 --model=cnn --dataset=cifar --iid=0 --test_acc=99 --gpu=1
```
-----

Hierarchical Federated experiments involve training a global model using different clusters with many local models.

* To run the hierarchical federated experiment with MNIST on MLP (IID):
```
python federated-hierarchical_main.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=5 --model=mlp --dataset=mnist --iid=1 --num_cluster=2 --test_acc=97  --gpu=1
```
* To run the same experiment under non-IID condition:
```
python federated-hierarchical_main.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=5 --model=mlp --dataset=mnist --iid=0 --num_cluster=2 --test_acc=97  --gpu=1
```

You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: 'mnist'. Options: 'mnist', 'fmnist', 'cifar'
* ```--model:```    Default: 'mlp'. Options: 'mlp', 'cnn'
* ```--gpu:```      Default: None (runs on CPU). Can also be set to the specific gpu id.
* ```--epochs:```   Number of rounds of training.
* ```--lr:```       Learning rate set to 0.01 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.
* ```--seed:```     Random Seed. Default set to 1.

#### Federated Parameters
* ```--iid:```      Distribution of data amongst users. Default set to IID. Set to 0 for non-IID.
* ```--num_users:```Number of users. Default is 100.
* ```--frac:```     Fraction of users to be used for federated updates. Default is 0.1.
* ```--local_ep:``` Number of local training epochs in each user. Default is 10.
* ```--local_bs:``` Batch size of local updates in each user. Default is 10.
* ```--unequal:```  Used in non-iid setting. Option to split the data amongst users equally or unequally. Default set to 0 for equal splits. Set to 1 for unequal splits.
* ```--num_clusters:```  Number of clusters in the hierarchy.
* ```--Cepochs:```  Number of rounds of training in each cluster.

## Results on MNIST
#### Baseline Experiment:
The experiment involves training a single model in the conventional way.

Parameters: <br />
* ```Optimizer:```    : SGD 
* ```Learning Rate:``` 0.01

```Table 1:``` Test accuracy after training for 10 epochs:

| Model | Test Acc |
| ----- | -----    |
|  MLP  |  92.71%  |
|  CNN  |  98.42%  |

----

#### Federated Experiment:
The experiment involves training a global model in the federated setting.

Federated parameters (default values):
* ```Fraction of users (C)```: 0.1 
* ```Local Batch size  (B)```: 10 
* ```Local Epochs      (E)```: 10 
* ```Optimizer            ```: SGD 
* ```Learning Rate        ```: 0.01 <br />

```Table 2:``` Test accuracy after training for 10 global epochs with:

| Model |    IID   | Non-IID (equal)|
| ----- | -----    |----            |
|  MLP  |  88.38%  |     73.49%     |
|  CNN  |  97.28%  |     75.94%     |
