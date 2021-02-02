# Base DSRE
Baselines of bag-level distantly supervised relation extraction models.

H<sub>2</sub><sup>2</sup>O

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
| PCNN | 83.0 | 75.5 | 72.7 | 77.1 | 40.8 |
| CNN | 80.0 | 74.5 | 70.3 | 74.9 | 39.2 |
| BiGRU | 60.0 | 58.5 | 59.3 | 59.1 | 35.0 |

## PR Curves
![](https://github.com/tmliang/Base-DSRE/blob/main/pr.jpg)
