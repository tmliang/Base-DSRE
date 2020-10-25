# Base DSRE
Baselines of sentence-level attention-based distantly supervised relation extraction models.

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

## Best Experimental Result

| Encoder | P@100 | P@200 | P@300 | Mean | AUC |
| :-----: | :---: | :---: | :---: | :--: | :-: |
| CNN | 86.0 | 77.0 | 73.3 | 74.6 | 41.3 |
| PCNN | 86.0 | 77.0 | 73.3 | 74.6 | 41.3 |
| BiGRU | 86.0 | 77.0 | 73.3 | 74.6 | 41.3 |
