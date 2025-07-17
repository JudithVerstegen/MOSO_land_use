# MOSO_land_use
Multi-objective spatial optimization of land use with the pymoo library.

## Description

This is a tutorial about spatial optimization. The goal of this tutorial is to show you how to apply a multi-objective optimization algorithm to non-spatial problems and how you can extend it to solve spatial problems. This is demonstrated with a case study about optimizing land use in Brazil by maximizing agricultural yield and minimizing carbon emissions from land use change. 

## Installation

Use [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) and the package manager [pip](https://pip.pypa.io/en/stable/) to install MOSO_land_use.

```bash
# create a new environment with python and Numpy installed
conda env create -f environment_moso.yml

# activate the environment
conda activate moso

# now install pymoo, as it can only be installed with pip
pip install -U pymoo
```

## Usage

All steps are explained in the pdf 'MOSO_land_use_tutorial' in the root directory, see [here](https://github.com/JudithVerstegen/MOSO_land_use/blob/main/MOSO_land_use_tutorial.pdf).

The input land use raster is included in the repository; with this raster, the tutorial can be performed up to section 4.3. The potential yield rasters are too large for the tutorial. Please contact me to obtain them, or download your own potential yield rasters (see link in the tutorial). 


