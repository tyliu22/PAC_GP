# PAC-Bayes NIGP algorithms

The code allows the users to experiment with the proposed NIGP training method.

## Datasets download
  Datasets can be download via [UCI datasets](https://github.com/treforevans/uci_datasets) or [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php)

<!-- 
## Requirements

The PAC-GP core code depends on Tensorflow.
For running the experiments, GPflow and sklearn are also required. -->


## Reproducing results

The experiments reported in the publication can be reproduced by executing

```
python epsilon_study.py  --run --plot
python sparseGP_study.py --run --plot
```


