# Dependencies
python==3.8.16
torch==1.12.0
torch-geometric==2.3.0


# IEMOCAP
## Preparing datasets for training

    python preprocess.py --data './data/iemocap/newdata.pkl' --dataset="iemocap"

## Training networks 

    python train.py --wf -10 --wp -10 --data './data/iemocap/newdata.pkl' --epoch 80 --from_begin --n_speakers 2

## Performance Comparision

-|Dataset|Weighted F1(w) | Acc
:-:|:-:|:-:
Original|IEMOCAP|70.22% | 70.12%

# MELD

## Preparing datasets for training

    python preprocess_roberta.py --data './data/meld/newdata.pkl' --dataset="meld" 

## Training networks 

    python train.py --wf -10 --wp -10 --data './data/meld/newdata.pkl' --epoch 80 --from_begin --n_speakers 9

## Performance Comparision

-|Dataset|Weighted F1(w) | Acc
:-:|:-:|:-:
Original|MELD|65.18% | 66.21%




# Acknowledgments

The structure of our code is inspired by [pytorch-DialogueGCN-mianzhang](https://github.com/mianzhang/dialogue_gcn).
