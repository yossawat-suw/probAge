'''
Apply the model on a external dataset
'''

# %%
# IMPORTS
# %load_ext autoreload
# %autoreload 2
import sys
import os
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


normalization_list = ["Raw_Beta_Values_QCed", 
                      "Normalised_Beta_Values_noob",
                      "Normalised_Beta_Values_bmiq", 
                      "Normalised_Beta_Values_noob_bmiq"]
disease_list = ["IFN_40","Sotos_71","Control_876"]

normalization = normalization_list[0]
disease = disease_list[2]

EXPORT_DIR_PATH = 'exports'
EXPORT_FILE_NAME = 'probage'+'_' + normalization + "_"  + disease + ".csv"
EXPORT_FILE_PATH = EXPORT_DIR_PATH + "/" + EXPORT_FILE_NAME

# Set paths to external data
#path_to_data = 'data/Raw_Beta_Values_QCed_Sotos_71.csv'
DATA_PATH = 'data'
DATA_FILE_NAME = normalization + "_"  + disease + ".csv"
path_to_data = DATA_PATH + "/" + DATA_FILE_NAME

path_to_meta = DATA_PATH + "/" + 'sample_data' + '_'  + disease + ".csv"



# %% #################
# Load external datasets to pandas
data_df = pd.read_csv(path_to_data, index_col=0)
meta_df = pd.read_csv(path_to_meta, index_col=0)

# intersection of indexes between data and meta
part_index_intersection = data_df.columns.intersection(meta_df.index)

# create anndata object
amdata = ad.AnnData(data_df[part_index_intersection],
                    var=meta_df.loc[part_index_intersection])


# Load reference sites
sites_ref = pd.read_csv('resources/wave3_sites.csv', index_col=0)
# amdata = ad.read_h5ad('resources/downsyndrome.h5ad')

# Load intersection of sites in new dataset
params = list(modelling.SITE_PARAMETERS.values())

intersection = sites_ref.index.intersection(amdata.obs.index)
amdata.obs[params] = sites_ref[params]


amdata = amdata[intersection].copy()

# %% #################
# BATCH (MODEL) CORRECTION

# Create amdata chunks using only control
amdata_chunks = modelling.make_chunks(amdata[:, amdata.var.status=='control'],
                                      chunk_size=15)

# print('Calculating the offsets...')
# if 'status' in amdata.var.columns:
#     offsets_chunks = [model.site_offsets(chunk[:,amdata.var.status=='healthy']) for chunk in tqdm(amdata_chunks)]
# else:
#     offsets_chunks = [model.site_offsets(chunk) for chunk in tqdm(amdata_chunks)]

with Pool(N_CORES) as p:
    offsets_chunks = list(tqdm(p.imap(modelling.site_offsets, amdata_chunks), total=len(amdata_chunks)))


offsets = np.concatenate([chunk['offset'] for chunk in offsets_chunks])

# # Infer the offsets
amdata.obs['offset'] = offsets
amdata.obs.eta_0 = amdata.obs.eta_0 + amdata.obs.offset
amdata.obs.meth_init  = amdata.obs.meth_init + amdata.obs.offset

# %% ##################
# PERSON MODELLING  
print('Calculating person parameters (acceleration and bias)...')

# ab_maps = model.person_model(amdata, method='map', progressbar=True, map_method=None)

amdata_chunks = modelling.make_chunks(amdata.T, chunk_size=15)
amdata_chunks = [chunk.T for chunk in amdata_chunks]
with Pool(N_CORES) as p:
    map_chunks = list(tqdm(p.imap(modelling.person_model, amdata_chunks)
                            ,total=len(amdata_chunks)))

for param in ['acc', 'bias']:
    param_data = np.concatenate([map[param] for map in map_chunks])
    amdata.var[f'{param}_{REF_DSET_NAME}'] = param_data

# Export
amdata.var.to_csv(EXPORT_FILE_PATH)

