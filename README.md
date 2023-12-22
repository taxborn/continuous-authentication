# Continuous Authentication Experimentation
I wanted to extend my research done on [taxborn/mauth-research-project](https://github.com/taxborn/mauth-research-project) to
clean up the code, and implement some ideas I didn't get to investigate.

The dataset I am using is the same dataset from that repository for now, but hope to extend to other datasets in the future.

# Running the models
All you need to run the models is to ensure you have the requirements installed in [requirements.txt](./requirements.txt), and execute `train.py`.
```
pip install -r requirements.txt && python train.py
```

There is also a Jupyter Notebook at [ca-visualization.ipynb](./ca-visualization.ipynb), which has an additional dependency of `matplotlib`.
This contains some visualizations of validations of the models.

## Implemented
- Preprocessing for the [mauth dataset](https://github.com/taxborn/mauth-research-project).
- Training KNearestNeighbors Classifier & Regressor
    - The regressor is used in computing *trust score*, metric of trust the system has in the current user
- Type-checked with [mypy](https://www.mypy-lang.org/)!

## TODO
- Create a script for data collection
    - Compute features ***continuously***.
    - Run against the model ***continuously***.
- Experiment with different trust score equations
- Configurable if this is deterministic or not (e.g. `random_state`)