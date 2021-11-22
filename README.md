# ID3 Decision Tree

An implementation of the [ID3 Algorithm](https://en.wikipedia.org/wiki/ID3_algorithm) for the creation of classification decision trees via maximizing information gain. Intended for continuous data with any number of features with only a single label (which can be multi-class). Binary splitting was employed to account for continuous data. A report summarizing the results and methodology was used as well.

For a less organized view, see branch `20211112-turnedin` for the materials turned in as part of MTSU's CSCI 4350 (Intro to AI) open lab assignment.

For a summary of results, see `report/report.pdf`.

# Installation

This repository relies on the Python scientific computing/data analysis libraries NumPy, Pandas, and Matplotlib.

`pip install numpy pandas matplotlib`

# Testing

To train and subsequently test the decision tree, use the command line interface for `id3.py`

```
python id3.py ./data/iris-data.txt ./data/iris-data.txt
```

The first argument is the training data, the second argument is the testing data. In this case, training and testing occurs on the same data set. For more arguments to any of the scripts, type `python name_of_script.py -h`.

# Analysis

Output for random resamples of testing size `n` were used to obtain the results in `report/report.pdf`. For the iris data set,

`n = [1,5,10,25,50,75,100,125,140,145,149]` while for the cancer data set

`n = [1,5,10,25,50,75,90,100,104]`.

Plotting and computation of descriptive statistics (mean and standard error) can be done using `plot.py` and `stats.py`, respectively.

# Future Work

The program works currently for continuous data only. It could be adapted for categorical data, or different splitting techniques for the continuous data could be implemented.
