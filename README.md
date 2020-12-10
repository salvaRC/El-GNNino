# Graph Neural Networks for Improved El Nino Forecasting (NeurIPS 2020 CCAI Workshop proposal)
*Deep learning-based models have recently outperformed state-of-the-art seasonal forecasting models, such as for predicting El Nino-Southern Oscillation (ENSO).
However, current deep learning models are based on convolutional neural networks which are difficult to interpret and can fail to model large-scale atmospheric patterns called teleconnections. We propose the first application of spatiotemporal graph neural networks (GNNs), that can model teleconnections for seasonal forecasting. Our GNN outperforms other state-of-the-art machine learning-based (ML) models for forecasts up to 3 month ahead. The explicit modeling of information flow via edges makes our model more interpretable, and our model indeed is shown to learn sensible edge weights that correlate with the ENSO anomaly pattern.*

## Environment setup
- Git clone the repo 

- Implemented using Python3 (3.7) with dependencies specified in requirements.txt, install them in a clean conda environment: <br>
    - ``conda create --name enso python=3.7`` <br>
    - ``conda activate enso`` <br>
    - ``pip install -r requirements.txt``
    - the [correct PyTorch version for your platform](https://pytorch.org/get-started/locally/]), e.g. ``conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch``
    - If you want to use any plotting code: ``conda install -c conda-forge cartopy``

This should be enough for Pycharm, for command line stuff you'll probably need to then also run

- ``python setup.py install``.


## Models
All reported models are saved [here](models).

## Running the experiments
- To run the models from scratch just rerun the [corresponding notebook](experiment1.ipynb).

## Citation

Please consider citing the following paper if you find it, or the code, helpful. Thank you!

    @article{cachay2020graph,
      title={Graph Neural Networks for Improved El Ni$\backslash$\~{} no Forecasting},
      author={Cachay, Salva R{\"u}hling and Erickson, Emma and Bucker, Arthur Fender C and Pokropek, Ernest and Potosnak, Willa and Osei, Salomey and L{\"u}tjens, Bj{\"o}rn},
      journal={NeurIPS Tackling Climate Change with Machine Learning Workshop. arXiv preprint arXiv:2012.01598},
      year={2020}
    }