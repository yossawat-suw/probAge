# %% ########################
### LOADING

# %load_ext autoreload 
# %autoreload 2

import scipy.stats as stats

import sys
import os
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
import anndata as ad


from src.general_imports import *
from src import preprocess_func
from src import modelling_bio_beta as model

sys.path.append("..")   # fix to import modules from root
# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Set the working directory to the script directory
os.chdir(script_dir)

# Add the script directory to the system path (optional, if you need to import modules from the script's directory)
sys.path.append(script_dir)


N_CORES = 12
N_SITES = 3_000  # number of sites to take in for the final person inferring
N_PART = 876
MULTIPROCESSING = True

""" normalization_list = ["Raw_Beta_Values_QCed", 
                      "Normalised_Beta_Values_noob",
                      "Normalised_Beta_Values_bmiq", 
                      "Normalised_Beta_Values_noob_bmiq"] """

normalization_list = ["Normalised_Beta_Values_bmiq", 
                      "Normalised_Beta_Values_noob_bmiq"]

split_age = True

# %%
for normalization in normalization_list:

    DSET_NAME = normalization  # external dataset name

    meta_data = pd.read_csv('../data/sample_data_Control_876.csv', index_col=0)
    data = ad.read_csv('../data/' + normalization + '_Control_876.csv')
    data.var = meta_data

    # Load into methylation specific anndata
    amdata = amdata_src.AnnMethylData(data)

    if split_age:
        amdata_young = amdata[:, amdata.var.age <= 20]
        amdata_old = amdata[:, amdata.var.age > 20]

        datasets = {'young': amdata_young, 'old': amdata_old}
    else:
        datasets = {'all': amdata}

    for group, amdata in datasets.items():
        ### PREPROCESS

        print(f'Preprocessing {DSET_NAME} dataset ({group}) with {amdata.shape[0]} sites...')

        amdata_chunks = amdata.chunkify(chunksize=1_000)

        print(f'Dropping NaNs...')
        for chunk in tqdm(amdata_chunks):
            chunk = preprocess_func.drop_nans(chunk)
            chunk.X = np.where(chunk.X == 0, 0.00001, chunk.X)
            chunk.X = np.where(chunk.X == 1, 0.99999, chunk.X)

        print(f'Calculating spearman correlation...')
        def spearman_r_loop(site_idx):
            spr = stats.spearmanr(amdata[site_idx].X.flatten(), amdata.var.age)
            return spr.statistic

        with Pool(N_CORES) as pool:
            result = list(tqdm(pool.imap(spearman_r_loop, amdata.obs.index), total=amdata.shape[0]))

        amdata.obs['spr'] = result
        amdata.obs['spr2'] = amdata.obs.spr**2

        ### SAVE DATA
        print(f'Saving the results...')
        amdata = amdata[amdata.obs.sort_values('spr2').index]
        amdata.write_h5ad(f'{paths.DATA_PROCESSED_DIR}/{DSET_NAME}_{group}_Control_' + str(amdata.shape[1]) + '_meta.h5ad')

        amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/{DSET_NAME}_{group}_Control_' + str(amdata.shape[1]) + '_meta.h5ad', backed='r')

        # select sites
        site_indexes = amdata.obs.sort_values('spr2').tail(N_SITES).index
        amdata = amdata[site_indexes, 0:N_PART].to_memory()

        print(f'Modelling sites for {DSET_NAME} dataset ({group}) with {amdata.shape[0]} sites and {amdata.shape[1]} participants after selection...')
        ########################
        ### MODELLING SITES

        amdata_chunks = model.make_chunks(amdata, chunk_size=15)

        # multiprocessing
        if MULTIPROCESSING:
            with Pool(N_CORES) as p:
                map_chunks = list(tqdm(p.imap(model.site_MAP, amdata_chunks), total=len(amdata_chunks)))

        ### SAVING

        # reload the dataset to keep all the participants
        amdata = amdata_src.AnnMethylData(f'{paths.DATA_PROCESSED_DIR}/{DSET_NAME}_{group}_Control_' + str(amdata.shape[1]) + '_meta.h5ad', backed='r')
        amdata = amdata[site_indexes, 0:N_PART].to_memory()

        # store results
        for param in model.SITE_PARAMETERS.values():
            param_data = np.concatenate([map[param] for map in map_chunks])
            amdata.obs[param] = param_data
        amdata = model.get_saturation_inplace(amdata)

        # save
        amdata.write_h5ad(f'{paths.DATA_PROCESSED_DIR}/{DSET_NAME}_{group}_Control_' + str(amdata.shape[1]) + '_sites_fitted.h5ad')
