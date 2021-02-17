# Supervised Learning
mstaszel3

# Code Location

The code is available at: https://github.com/mk6502/supervised-learning/

# Usage

I used Python 3.9 installed via Anaconda. The code should work with Python 3.6-3.8 too but I have not tested those.

Install required packages (standard scientific Python packages) with:

    pip install -r requirements.txt

`main.py` will run everything.

This will run for somewhere on the order of 30-60 minutes depending on the hardware.
All learners, hyperparameter grid search, and plotting all happens by running `main.py`.
Plots are output to the `output/plots` directory. Various metrics are written to the `output/metrics/` directory.

Run the code with:

    python main.py

My output from running `main.py` is included in the `output` directory in this repository.

# Details
The code is broken down into `learners` (which are mainly wrappers are `scikit-learn`) and `experiments` which call
learners and generate metrics and plots.

# Data Sources

* https://archive.ics.uci.edu/ml/datasets/Adult
* https://archive.ics.uci.edu/ml/datasets/Phishing+Websites
