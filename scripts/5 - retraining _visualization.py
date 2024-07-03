#%%import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
#%%
# Set the working directory
os.chdir("/home/point/Documents/work/research/pmayp/Project/nelly/type_1_interferonopathy/other_project/probAge/")


# Import person_modelling result result from different normalization methods
normalizations = [
    "Raw_Beta_Values_QCed", 
    "Normalised_Beta_Values_noob",
    "Normalised_Beta_Values_bmiq", 
    "Normalised_Beta_Values_noob_bmiq"
]

norm_list = []
for normalization in normalizations:
    csv_name = f"./exports/{normalization}_participants.csv"
    csv_each = pd.read_csv(csv_name)
    csv_each['normalization'] = normalization
    norm_list.append(csv_each)

norm_all = pd.concat(norm_list, ignore_index=True)
#%%
# Reshape data for Sankey
norm_all_qc = norm_all.drop(columns=["bias", "ll", "acc"]).pivot(index="Unnamed: 0", columns='normalization', values='qc').reset_index()


norm_all_qc = norm_all_qc.drop(columns=["Unnamed: 0"])
link = norm_all_qc.to_dict(orient="list")


#%%
import plotly.graph_objects as go

data = go.Sankey(link = link)
#%%
fig = go.Figure(data)

fig.show()
#%%
# Prepare data for Sankey diagram
sankey_data = norm_all_qc.melt(id_vars='X', value_name='qc', var_name='normalization')
sankey_data['source'] = sankey_data['normalization'].shift(1)
sankey_data.dropna(inplace=True)
#%%
# Create Sankey diagram
fig_sankey = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=sankey_data['normalization'].unique(),
        color="blue"
    ),
    link=dict(
        source=sankey_data['source'].apply(lambda x: sankey_data['normalization'].unique().tolist().index(x)),
        target=sankey_data['normalization'].apply(lambda x: sankey_data['normalization'].unique().tolist().index(x)),
        value=sankey_data['qc']
    )
))
fig_sankey.update_layout(title_text="Sankey Diagram of Normalizations", font_size=10)
fig_sankey.show()
#%%
# Scatter plot of accuracy vs bias
fig_scatter = px.scatter(norm_all, x='acc', y='bias', color='normalization', facet_col='normalization')
fig_scatter.update_layout(title_text="Accuracy vs Bias by Normalization")
fig_scatter.show()

# Statistical summary and selection of top and bottom results
norm_all_stat = norm_all.groupby('X').agg(
    sd_acc=('acc', 'std'),
    mean_acc=('acc', 'mean'),
    cov_acc=('acc', lambda x: x.std() / abs(x.mean())),
    
    sd_bias=('bias', 'std'),
    mean_bias=('bias', 'mean'),
    cov_bias=('bias', lambda x: x.std() / abs(x.mean())),
    
    cov_acc_and_bias=('acc', lambda x: x.std() / abs(x.mean()) + norm_all.loc[x.index, 'bias'].std() / abs(norm_all.loc[x.index, 'bias'].mean())),
    
    age=('age', 'mean')
).reset_index()

norm_all_stat = norm_all_stat.sort_values(by='cov_acc_and_bias', ascending=False)

# Select top and bottom 20
n = 20
norm_all_stat_top = norm_all_stat.head(n)['X']
norm_all_stat_tail = norm_all_stat.tail(n)['X']

# Plot top results
fig_top = px.scatter(norm_all[norm_all['X'].isin(norm_all_stat_top)], x='acc', y='bias', color='normalization', facet_col='X')
fig_top.update_layout(title_text="Top 20 Results by Covariance of Accuracy and Bias")
fig_top.show()

# Plot bottom results
fig_tail = px.scatter(norm_all[norm_all['X'].isin(norm_all_stat_tail)], x='acc', y='bias', color='normalization', facet_col='X')
fig_tail.update_layout(title_text="Bottom 20 Results by Covariance of Accuracy and Bias")
fig_tail.show()

# %%
