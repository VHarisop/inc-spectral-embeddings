# inc-spectral-embeddings
Code for the paper "Incrementally Updated Spectral Embeddings"

# Data
The scripts use 3 different datasets: `college-msg` is available from the SNAP dataset
collection [here](http://snap.stanford.edu/data/CollegeMsg.html), while the
`temporal-reddit-reply` dataset can be found [here](https://www.cs.cornell.edu/~arb/data/temporal-reddit-reply/index.html). Finally, the household power consumption dataset has been downloaded from the UCI ML repository, and is available [here](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption).

All 3 datasets should be downloaded and stored under a folder called
`datasets/` (which the scripts point to by default). Alternatively, the
`--filepath` option should be used to override the location of each data file,
when applicable.
