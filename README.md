# spatialTreat-example
a simple example for estimating the causal effects of spatial treatments, following Pollmann (2020).

## Requirements

R packages:

- `dplyr`
- `reticulate`
- `sf`
- `ggplot2`
- `data.table`
- `Rcpp`
- `readr`
- `tidyr`

Python 3, for instance using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for a minimal installation, and packages:

- `numpy`: `conda install numpy`
- `numba`: `conda install numba`
- `pytorch` if running neural networks locally, see [PyTorch website](https://pytorch.org/get-started/locally/) for installation instructions


## Overview

This example consists primarily of 5 scripts (3 R, 2 Python) to run, with an additional 5 code files (1 R, 2 Python, 2 C++) called from within the R scripts.
For ease of viewing, the R code is saved as R Markdown files, 
the resulting .html files, 
and as [Jupyter notebooks](https://jupyter.org/) generated from the R Markdown using [Jupytext](https://github.com/mwouts/jupytext). The Jupyter notebooks `.ipynb` have the advantage of being easily viewable within Github.
These files combine step-by-step explanations and the R code generating the results.
The neural network is best trained on a server.
For interactive use, I also link a Google Colab notebook .

**The data for this example is simulated**; this repository does not contain real data.
Neither the exact locations nor the outcomes are real.
The data are simulated to resemble the spatial distributions of businesses in parts of the San Francisco Bay Area,
specifically parts of the Peninsula between South San Francisco and Sunnyvale,
excluding the Santa Cruz mountains and towns on the Pacific coast.
The `visits` outcomes are distributed similarly to visits in [Safegraph](https://www.safegraph.com/academics) foot-traffic data as used in Pollmann (2020).
The `naics_code` included in the data set gives the 4-digit [NAICS](https://www.census.gov/naics/) industry; two trailing digits are appended for easier use because the neural network data import expects 6-digit codes.

The input data files are

- `data/dat_S.rds` with columns `s_id`,`latitude`,`longitude`,`naics_code` gives the locations of spatial treatments (grocery stores).
- `data/dat_A.rds` with columns `a_id`,`latitude`,`longitude`,`naics_code`,`visits` gives the locations of other businesses. Some of these businesses (those with `naics_code` 722500) are restaurants, the businesses for which we use the outcome. The remaining businesses are used only to find counterfactual locations in neighborhoods similar to the real treatment locations.


The scripts to run are

- `01_place_in_grid.Rmd`
  - creates the input for the neural net, as well as data sets about distances between locations.
  - input: `data/dat_S.rds`, `data/dat_A.rds`
  - output: `data/dat_D.rds`, `data/dat_DS.rds`, `data/dat_S_candidate_random.rds`, `neural-net/grid_S_I.csv`, `neural-net/grid_S_random_I.csv`, `neural-net/grid_S_random_S.csv`, `neural-net/grid_S_S.csv`
- `02_cnn_train.py`
  - trains the neural net that predicts counterfactual (and real) treatment locations. To train resume training the neural network from a given checkpoint, set `use_saved_model = True` and `saved_model_filename` to the file name of the checkpoint.
  - input: `neural-net/grid_S_I.csv`, `neural-net/grid_S_random_I.csv`, `neural-net/grid_S_random_S.csv`, `neural-net/grid_S_S.csv`
  - output: `checkpoint-epoch-<epoch>-YYYY-MM-DD--hh-mm.tar`.
- `03_cnn_pred.py`
    - predicts (likelihood of) counterfactual locations.
    - input: `neural-net/grid_S_I.csv`, `neural-net/grid_S_random_I.csv`, `neural-net/grid_S_random_S.csv`, `neural-net/grid_S_S.csv`, `checkpoint-epoch-<epoch>-YYYY-MM-DD--hh-mm.tar`
    - output: `neural-net/predicted_activation-YYYY-MM-DD--hh-mm.csv` (needs to be zipped manually)
- `04_pick_counterfactuals.R`
    - picks counterfactual locations from the neural network prediction that best resemble true treatment locations and estimates propensity scores (treatment probabilities)
    - input: `data/dat_S.rds`, `data/dat_A.rds`, `data/dat_D.rds`, `data/dat_S_candidate_random.rds`, `neural-net/predicted_activation-YYYY-MM-DD--hh-mm.zip` (note that this file was zipped manually from the output of the previous step to reduce the file size on Github; due to file size, the file is not included in the Github repository, please download it from Google Drive instead: [predicted_activation-2021-11-23--13-37.zip](https://drive.google.com/file/d/15rUm87upjyKz3nRT0r39Mq0Gjqg6EmNS/view?usp=share_link))
    - output: `data/dat_cand.rds`, `data/dat_D_cand.rds`, `data/dat_reg_p_cand.rds`, `data/dat_reg_out_sel.rds`, `data/dat_out_num_treat.rds`
- `05_estimate_effects.R`
    - creates figures and estimates treatment effects based on (conditional on) the counterfactual locations and propensity scores determined in the previous step
    - input: `data/dat_S.rds`, `data/dat_A.rds`, `data/dat_D.rds`, `data/dat_cand.rds`, `data/dat_D_cand.rds`, `data/dat_reg_out_sel.rds`
    - output: see table and figures of estimated effects in `.ipynb` and `.html`


## Notes

Estimating effects at many distances or with very many (realized and unrealized) treatment locations close to one another can be slow.
The difficulty arises primarily in the estimation of the variance where the number of operations needed is exponential in the number of treatment locations an individual (restaurant) is exposed to.
Some optimizations are not yet implemented but in principle feasible:

- an optional argument that allows skipping variance estimation (for faster exploratory analyses)
- computing some objects needed for estimation at multiple distances only once (will increase memory use, however)
- use sparse matrices for `M$pmsa0` and `M$pmsa1` in `allExposures()` in `estimate_variance`
- parallel computing of multiple distances
- optional sampling of exposures (introduces random noise in the estimated variance but can be much faster to compute)

The file `predicted_activation-2021-11-23--13-37.zip` is too large for Github. Please download it from Google Drive instead: [predicted_activation-2021-11-23--13-37.zip](https://drive.google.com/file/d/15rUm87upjyKz3nRT0r39Mq0Gjqg6EmNS/view?usp=share_link) before running the code in `04_pick_counterfactuals.R` (unless you created predictions from the neural network on your local machine and intend to use these).

Please share any bugs, missing features, or desired settings not covered in the current example.
At some point, most data preparation and estimation functions in this repository will be distributed as an R package, with updates to this tutorial to follow.
Contributions are welcome.


## Reference

Michael Pollmann. **Causal Inference for Spatial Treatments**. 2020. [[paper](https://michaelpollmann.github.io/files/pollmann_spatial_treatments.pdf)]