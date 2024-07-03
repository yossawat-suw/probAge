# %% ########################
### LOADING

# %load_ext autoreload 
# %autoreload 2

import os
import sys
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.stats as stats
import anndata as ad
import glob

# Set the working directory to the script's location
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
sys.path.append("..")   # fix to import modules from root

from src.general_imports import *
from src import preprocess_func
from src import modelling_bio_beta as model

N_CORES = 15
N_SITES = 1024  # number of sites to take in for the final person inferring
MULTIPROCESSING = True
split_age = True  # Control flag for splitting the analysis by age

""" normalization_list = ["Raw_Beta_Values_QCed", 
                      "Normalised_Beta_Values_noob",
                      "Normalised_Beta_Values_bmiq", 
                      "Normalised_Beta_Values_noob_bmiq"] """

normalization_list = ["Normalised_Beta_Values_noob_bmiq"] 
for normalization in normalization_list:

    DATASET_NAME = normalization

    participants = pd.read_csv('../data/sample_data_Control_876.csv', index_col=0)

    if split_age:
        # Split participants by age
        young_participants = participants[participants['age'] <= 20]
        old_participants = participants[participants['age'] > 20]

        groups = {'young': young_participants, 'old': old_participants}
    else:
        # Use all participants if not splitting by age
        groups = {'all': participants}

    for group, group_participants in groups.items():
        
        # Find the correct file path dynamically
        search_pattern = f'../exports/{DATASET_NAME}_{group}_Control_*_sites_fitted.h5ad'
        files = glob.glob(search_pattern)
        
        if not files:
            print(f"No files found for pattern: {search_pattern}")
            continue
        
        # Assume the first matching file is the correct one (adjust logic if needed)
        amdata_file = files[0]
        amdata = amdata_src.AnnMethylData(amdata_file)
        
        # select only not saturating sites
        print(f'There are {(~amdata.obs.saturating == 1).sum()} not saturating sites in {DATASET_NAME}')
        amdata = amdata[~amdata.obs.saturating]

        # further select sites
        site_indexes = amdata.obs.sort_values('spr2').tail(N_SITES).index
        amdata = amdata[site_indexes].to_memory()

        ### MODELLING PEOPLE

        amdata_chunks = model.make_chunks(amdata.T, chunk_size=15)
        amdata_chunks = [chunk.T for chunk in amdata_chunks]

        if MULTIPROCESSING:
            with Pool(N_CORES) as p:
                map_chunks = list(tqdm(p.imap(model.person_model,
                                                amdata_chunks),
                                        total=len(amdata_chunks)))

        if not MULTIPROCESSING:
            map_chunks = map(model.person_model, amdata_chunks)

        for param in ['acc', 'bias']:
            param_data = np.concatenate([map[param] for map in map_chunks])
            amdata.var[param] = param_data

        # compute log likelihood for inferred parameters to perform quality control
        ab_ll = model.person_model_ll(amdata)
        amdata.var['ll'] = ab_ll

        amdata.var['qc'] = model.get_person_fit_quality(amdata.var['ll'])

        # Save the .h5ad file with amdata.shape[1]
        amdata.write_h5ad(f'../exports/{DATASET_NAME}_{group}_Control_{amdata.shape[1]}_person_fitted.h5ad')

        # Save the .csv file with the number of participants
        group_participants = participants.loc[amdata.var.index]
        group_participants['acc'] = amdata.var['acc']
        group_participants['bias'] = amdata.var['bias']
        group_participants['ll'] = amdata.var['ll']
        group_participants['qc'] = amdata.var['qc']

        group_participants.to_csv(f'../exports/{DATASET_NAME}_{group}_Control_{group_participants.shape[0]}_participants.csv')


