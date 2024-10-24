# Dependencies
python==3.8.16
torch==1.12.0+cu116
torch-geometric==2.3.0
## Introduction
```
Pytorch code for Neural Networks 2024 accepted paper "HiMul-LGG: A hierarchical decision fusion-based local–global graph neural network for multimodal emotion recognition in conversation"
```
# IEMOCAP
## Preparing datasets for training

    python preprocess.py --data './data/iemocap/newdata.pkl' --dataset="iemocap"

## Training networks 

    python train.py --wf -10 --wp -10 --data './data/iemocap/newdata.pkl' --from_begin --device=cuda --epochs=80 --batch_size=20 --n_speakers 2 

## Predictioning networks 

    python prediction.py --data=./data/iemocap/newdata.pkl --device=cuda --epochs=1 --batch_size=20 --n_speakers 2


## Performance Comparision

|Dataset|Weighted F1(w) | Acc |
:-:|:-:|:-:
|IEMOCAP|70.22% | 70.12% |

# MELD

## Preparing datasets for training

    python preprocess_roberta.py --data './data/meld/newdata.pkl' --dataset="meld" 

## Training networks 

    python train.py --wf -10 --wp -10 --data './data/meld/newdata.pkl' --device=cuda --epoch 80 --from_begin --batch_size=20 --n_speakers 9

## Predictioning networks 

    python prediction.py --data=./data/meld/newdata.pkl --device=cuda --epochs=1 --batch_size=20 --n_speakers 9


## Performance Comparision

|Dataset|Weighted F1(w) | Acc |
:-:|:-:|:-:
|MELD|65.18% | 66.21% |


# Acknowledgments

The structure of our code is inspired by [pytorch-DialogueGCN-mianzhang](https://github.com/mianzhang/dialogue_gcn).

# Publication (Please kindly cite our paper)

[HiMul-LGG: A hierarchical decision fusion-based local-global graph neural network for multimodal emotion recognition in conversation](https://www.sciencedirect.com/science/article/pii/S0893608024006889)


@article{fu2024himul,\
  title={HiMul-LGG: A hierarchical decision fusion-based local-global graph neural network for multimodal emotion recognition in conversation},\
  author={Fu, Changzeng and Qian, Fengkui and Su, Kaifeng and Su, Yikai and Wang, Ze and Shi, Jiaqi and Liu, Zhigang and Liu, Chaoran and Ishi, Carlos Toshinori},\
  journal={Neural Networks},\
  pages={106764},\
  year={2024},\
  publisher={Elsevier}\
}


