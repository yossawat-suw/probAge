# %%
# IMPORTS
# %load_ext autoreload
# %autoreload 2
import sys
import os
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import anndata as ad
import numpy as np
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src import paths
from src import modelling_bio_beta as modelling

# %% #################
# Determine the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change to the script directory
os.chdir(script_dir)

# Change working directory to 'probAge' using relative path
os.chdir('..')

# Print the current working directory
print("Current Working Directory:", os.getcwd())
# %% #################

N_CORES = 10

REF_DSET_NAME = 'probage_bc'
EXPORT_DIR_PATH = './exports'
DATA_PATH = './data'

normalization_list = ["Raw_Beta_Values_QCed", 
                      "Normalised_Beta_Values_noob",
                      "Normalised_Beta_Values_bmiq", 
                      "Normalised_Beta_Values_noob_bmiq"]
disease_list = ["IFN_40","Sotos_71","Control_876"]

normalization_chosen = normalization_list
disease_chosen = disease_list[0:2]


#%%
# Load reference sites
#sites_ref = pd.read_csv('resources/wave3_sites.csv', index_col=0)


# %%
for normalization in normalization_chosen:

    data_ref = ad.read_h5ad('./exports/'+normalization+'_Control_876_sites_fitted.h5ad')
    sites_ref = data_ref.obs

    for disease in disease_chosen:
        EXPORT_FILE_NAME = 'probage' + '_' + normalization + "_" + disease + ".csv"
        EXPORT_FILE_PATH = EXPORT_DIR_PATH + "/" + EXPORT_FILE_NAME

        DATA_FILE_NAME = normalization + "_" + disease + ".csv"
        path_to_data = DATA_PATH + "/" + DATA_FILE_NAME
        path_to_meta = DATA_PATH + "/" + 'sample_data' + '_' + disease + ".csv"

        # Load external datasets to pandas
        data_df = pd.read_csv(path_to_data, index_col=0)
        meta_df = pd.read_csv(path_to_meta, index_col=0)

        # intersection of indexes between data and meta
        part_index_intersection = data_df.columns.intersection(meta_df.index)

        # create anndata object
        amdata = ad.AnnData(data_df[part_index_intersection],
                            var=meta_df.loc[part_index_intersection])

        # Load intersection of sites in new dataset
        params = list(modelling.SITE_PARAMETERS.values())

        intersection = sites_ref.index.intersection(amdata.obs.index)
        amdata.obs[params] = sites_ref[params]

        amdata = amdata[intersection].copy()


        # BATCH (MODEL) CORRECTION

        # Create amdata chunks using only control
        amdata_chunks = modelling.make_chunks(amdata[:, amdata.var.status=='control'],
                                                chunk_size=15)

        with Pool(N_CORES) as p:
            offsets_chunks = list(tqdm(p.imap(modelling.site_offsets, amdata_chunks), total=len(amdata_chunks)))

        offsets = np.concatenate([chunk['offset'] for chunk in offsets_chunks])

        # Infer the offsets
        amdata.obs['offset'] = offsets
        amdata.obs.eta_0 = amdata.obs.eta_0 + amdata.obs.offset
        amdata.obs.meth_init  = amdata.obs.meth_init + amdata.obs.offset


        # PERSON MODELLING  
        print(f'Calculating person parameters (acceleration and bias) for {normalization} and {disease}...')

        amdata_chunks = modelling.make_chunks(amdata.T, chunk_size=15)
        amdata_chunks = [chunk.T for chunk in amdata_chunks]
        with Pool(N_CORES) as p:
            map_chunks = list(tqdm(p.imap(modelling.person_model, amdata_chunks), total=len(amdata_chunks)))

        for param in ['acc', 'bias']:
            param_data = np.concatenate([map[param] for map in map_chunks])
            amdata.var[f'{param}_{REF_DSET_NAME}'] = param_data

        # Export
        amdata.var.to_csv(EXPORT_FILE_PATH)
        print(f'Exported results to {EXPORT_FILE_PATH}')


#%%
#sns.scatterplot(amdata.var,x="age",y="acc_probage_bc")
sns.scatterplot(amdata.var,x="age",y="bias_probage_bc")
# %%
amdata.var
# %%
param_data
# %%
