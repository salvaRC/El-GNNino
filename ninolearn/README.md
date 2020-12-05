## This code has been build upon & adapted from the Ninolearn library, see below.
Note that the longitude coordinates are in degrees East, which is why ONI region corresponds to 190-240.


# NinoLearn

<img src="https://github.com/pjpetersik/ninolearn/blob/master/logo/logo.png" width="250" align="right">

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://pjpetersik.github.io/ninolearn/

[![latest][docs-latest-img]][docs-latest-url]

NinoLearn is a research framework for the application of machine learning (ML)
methods for the prediction of the El Nino-Southern Oscillation (ENSO).

It contains methods for downloading relevant data from their sources, reading
raw data, postprocessing it and then access the postprocessed data in an easy way. 


Moreover, it contains models for the ENSO forecasting:

· Deep Ensemble Model (DEM)

· Encoder-Decoder Ensemble Model

## Installation for own development

1. Fork the repository.
2. Clone the repository to your local machine.
```
git clone https://github.com/Your_Username/ninolearn
```
3. Make a conda environment from the .yml file.
```
conda env create -f ninolearn.yml
```
4. Activate the environment.
```
conda activate ninolearn
```
5.  Add ninolearn to the conda environment in 'development mode'.
```
conda develop /path/to/ninolearn
```
6. Fill out the `ninolearn/private_template.py` file with the required pathes and save a copy as `private.py`. The `private.py` will not be pushed to your remote repository because it contain sensitive information as well as pathes that are specific to your machine.

Now you should be ready to use ninolearn. For the beginning you can try to run the Jupyter Notebook tutorials which are currently located in `docs-sphinx/source/jupyter_notebook_tutorials/`. 

## Folder structure
In the folder `ninolearn` the actual ninolearn code is located. 
The `research` folder contains pervious research that was done with ninolearn. Hence, if you want to do your own research with ninolearn, make a new directory in the research folder in which you can start to do your own stuff.

The folders `docs` and `docs-sphinx` contain the documentation of ninolearn. Currently the documentation is somewhat outdated.
