# Continuous Authentication Experimentation
I wanted to extend my research done on [taxborn/mauth-research-project](https://github.com/taxborn/mauth-research-project) to
clean up the code, and implement some ideas I didn't get to investigate.

The dataset I am using is the same dataset from that repository for now, but hope to extend to other datasets in the future.

# Running the models
All you need to run the models is to ensure you have the requirements installed in [requirements.txt](./requirements.txt), and execute `python train.py`.

## Implemented
- Preprocessing for the [mauth dataset](https://github.com/taxborn/mauth-research-project).
- Training KNearestNeighbors Classifier & Regressor
    - The regressor is used in computing *trust score*, metric of trust the system has in the current user

## TODO
- Create a script for data collection
    - Compute features ***continuously***.
    - Run against the model ***continuously***.
- Experiment with different trust scores