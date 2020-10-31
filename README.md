# Base DSRE
Baselines of bag-level distantly supervised relation extraction models.

## Requirements
* python==3.8
* pytorch==1.6
* numpy==1.19.1
* tqdm==4.48.2
* scikit_learn==0.23.2

## Data
Download the dataset from [here](https://github.com/thunlp/HNRE/tree/master/raw_data), and unzip it under `./data/`.

## Train and Test
```
python main.py
```

## Experimental Result

| Encoder | P@100 | P@200 | P@300 | Mean | AUC |
| :-----: | :---: | :---: | :---: | :--: | :-: |
| PCNN | 77.0 | 75.0 | 72.7 | 73.4 | 41.4 |
| CNN | 74.0 | 73.0 | 68.7 | 71.9 | 38.1 |
| BiGRU | 60.0 | 58.5 | 59.3 | 59.1 | 35.0 |

## PR Curves
![](https://github.com/tmliang/Base-DSRE/blob/main/pr.jpg)
