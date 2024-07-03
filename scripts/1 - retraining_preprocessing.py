# %% ########################
### LOADING

# %load_ext autoreload 
# %autoreload 2

import scipy.stats as stats

import sys

sys.path.append("..")   # fix to import modules from root

from src.general_imports import *
from src import preprocess_func

N_CORES = 12
MULTIPROCESSING = True

DSET_NAME = 'Control' # external dataset name

# %%
#meta_data = pd.read_csv('../data/sample_data_Control_1004.csv', index_col=0)
#data = ad.read_csv('../data/Normalised_Beta_Values_noob_bmiq_Control_1004.csv')
meta_data = pd.read_csv('../data/sample_data_Control_876.csv', index_col=0)
data = ad.read_csv('../data/Raw_Beta_Values_QCed_Control_876.csv')
data.var = meta_data

# Load into methylation specific anndata
amdata = amdata_src.AnnMethylData(data)

## Reduce CpG dimensionality for testing
#amdata = amdata[:3000]

# %%

### PREPROCESS

print(f'Preprocessing {DSET_NAME} dataset with {amdata.shape[0]} sites...')

amdata_chunks = amdata.chunkify(chunksize=1_000)

print(f'Dropping NaNs...')
for chunk in tqdm(amdata_chunks):
    chunk = preprocess_func.drop_nans(chunk)
    chunk.X = np.where(chunk.X == 0, 0.00001, chunk.X)
    chunk.X = np.where(chunk.X == 1, 0.99999, chunk.X)

# %%
print(f'Calculating spearman correlation...')
def spearman_r_loop(site_idx):
    spr = stats.spearmanr(amdata[site_idx].X.flatten(), amdata.var.age)
    return spr.statistic

with Pool(N_CORES) as pool:
    result = list(tqdm(pool.imap(spearman_r_loop, amdata.obs.index), total=amdata.shape[0]))

amdata.obs['spr'] = result
amdata.obs['spr2'] = amdata.obs.spr**2

# %% ########################
### SAVE DATA
print(f'Saving the results...')
amdata = amdata[amdata.obs.sort_values('spr2').index]
amdata.write_h5ad(f'{paths.DATA_PROCESSED_DIR}/{DSET_NAME}_'+ str(amdata.shape[1]) +'_meta.h5ad')
# %%


# %%
