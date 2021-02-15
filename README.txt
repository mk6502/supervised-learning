# Supervised Learning
mstaszel3

# Code Location

The code is available at: https://github.com/mk6502/supervised-learning/

# Usage

I used Python 3.9 installed via Anaconda. The code should work with Python 3.6-3.8 too but I have not tested those.

Install required packages (standard scientific Python packages) with:

    pip install -r requirements.txt

There are two components: `main.py` and `plots.py`.

`main.py` will run everything.

This will run for somewhere on the order of 30-60 minutes depending on the hardware.
All learners, hyperparameter grid search, and plotting all happens by running `main.py`.
Plots are output to the `plots` directory. Various metrics are printed to the console and output to a log file named
`output/main.log`.

Since training takes a long time, some plotting is broken out into a separate `plots.py` file which uses output from a
previous run of `main.py` to generate plots.

This file *is* called by `main.py` so you do NOT need to run it manually after running `main.py`.

In summary: run the code with:

    python main.py

If you want to only rerun plotting (e.g. after modifying `plots.py`, you do not need to run all of `main.py` - you can
run `plots.py` independently. But `main.py` will run it for you.

My output from running `main.py` is included in the `output` directory in this repository.

# Data Sources

* https://archive.ics.uci.edu/ml/datasets/Adult
* https://archive.ics.uci.edu/ml/datasets/Phishing+Websites
