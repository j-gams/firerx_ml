## Overview of FireRX ML

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.21385388.svg)](http://dx.doi.org/10.5281/zenodo.21385388)

### Paper and Citation information
This repository contains code and experiments from our paper: Gammie, J.; Verleye, E.; Illangakoon, N.; LoPresti, A.; Dee, L.; Monteleoni, C.; Stravos, N.; Amaral, C. "Data Pyramids: A Lightweight, Multisource Paradigm for Large-Scale Deep Learning Analysis of Ecosystem Functioning and Services" (in review). 

Here, you will find the data pyramids extraction and preprocessing pipeline, data pyramid-based CNN and ViT models, as well as our experimental pipeline and evaluation methods.

Please cite:
```
@software{firerx_ml,
  author  = {Jerome Gammie et al.},
  title   = {firerx_ml: Framework for Machine Learning with Data Pyramids},
  year    = {2026},
  url     = {https://github.com/j-gams/firerx_ml},
  version = {v1.0.0},
  doi = {https://doi.org/10.5281/zenodo.21385388}
}
```

#### manage_data
This directory contains tools for managing data, particularly for cleaning raw raster data, aligning raster data for use in causal inference, and extracting analysis-ready datasets suitable for machine learning from raw raster data (both data cubes and data pyramids). 

#### models
This directory contains a framework for running machine learning models and collecting performance data, plus baseline models and novel pyramid-ready CNN and ViT models. It also includes the ML dataset management framework.

#### utils
This directory contains primarily tools for generating config files (used to set parameters when aligning data, building datasets, or training ML models

### Funding information
National Aeronautics and Space Administration (NASA) grant #80NSSC23K0397 and the National Science Foundation (NSF) grant #2153040. 
