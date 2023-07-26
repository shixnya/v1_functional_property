#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.signal import welch
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

#%% load data
manifest_path = os.path.join("/local1/data/ecephys_cache_dir/", "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
print(cache.get_all_session_types())

sessions = cache.get_session_table()
bo_sessions = sessions[sessions.session_type == "brain_observatory_1.1"]
len(sessions)
print("# Brain Observatory 1.1 sessions: " + str(len(bo_sessions)))
cell_metrics = pd.read_csv('/allen/programs/mindscope/workgroups/realistic-model/shinya.ito/ResourceTable/NeuroPixels/cortical_metrics_1.4.csv')
layer_info = pd.read_csv('layer_info.csv')
waveform_info = pd.read_csv('waveform_type.csv')

#%% functions
def calc_power(row,binsize=5e-3):
    # first count spikes using bin size of 5 ms
    spike_time_since = np.array(row['spike_time_since'])
    n_units = row['n_units']
    (counts,bins) = np.histogram(spike_time_since, bins=np.arange(.2,np.max([spike_time_since.max(),1.2])+binsize,binsize))
    counts = counts/n_units
    sr = 1/binsize
    # calculate power
    f, Pxx = welch(counts,fs=sr,nperseg=sr,scaling='spectrum')
    return f,Pxx

def power_analysis(table,unit_id):
    spike_df_all = session.presentationwise_spike_times( 
            stimulus_presentation_ids=table.index.values,  
            unit_ids=unit_id,
            )
    # drop spikes within .2 sec of stimulus start
    spike_df = spike_df_all.drop(spike_df_all[spike_df_all['time_since_stimulus_presentation_onset']<.2].index).reset_index()
    spike_df.rename(columns = {'time_since_stimulus_presentation_onset':'spike_time_since'},inplace=True)

    # group spikes by stimulus
    spike_df_group = spike_df.groupby(['stimulus_presentation_id']).agg({
        'spike_time_since': list,
    }).reset_index()

    # add n_units
    count_df = spike_df.groupby('stimulus_presentation_id')['unit_id'].nunique().to_frame().reset_index()
    count_df.rename(columns={'unit_id':'n_units'},inplace=True)
    spike_df_group = pd.merge(spike_df_group,count_df,on='stimulus_presentation_id')

    # calculate power for each stimulus presentation
    spike_df_group[['f','power']] = spike_df_group.apply(calc_power,axis=1,result_type='expand')

    return spike_df_group


# %%%%%%%%%%%%%%%%%%%%%%% power analysis 

df_cell_type = pd.DataFrame({})
df_wf_layer = pd.DataFrame({})
df_all = pd.DataFrame({})
t = time.time()

for i in bo_sessions.index.values:
# i = sessions.index.values[0]
    print(i)
    print(bo_sessions.loc[i,'session_type'])
    session=cache.get_session_data(i)
    units = session.units
    units.reset_index(inplace=True)
    # add cell_type from cell_metrics to units based on unit_id
    units = pd.merge(units,cell_metrics[['ecephys_unit_id','cell_type']],left_on='unit_id',right_on='ecephys_unit_id',how='left')
    # add layer info
    units = pd.merge(units,layer_info[['ecephys_unit_id','cortical_layer']],left_on='unit_id',right_on='ecephys_unit_id',how='left')
    # add waveform type
    units = pd.merge(units,waveform_info[['ecephys_unit_id','waveform_type']],left_on='unit_id',right_on='ecephys_unit_id',how='left')

    dg_table = session.get_stimulus_table(['drifting_gratings'])
    dg_table = dg_table[dg_table.temporal_frequency==2]
    sp_table = session.get_stimulus_table(['spontaneous'])

    # drop short spontaneous recording
    sp_table.drop(sp_table[sp_table['duration']<5].index,inplace=True)
    # only keep units in VIS
    units_v1 = units[units.ecephys_structure_acronym.str.contains("VISp")]

    ############################### for each cell type #########################################
    print('group by cell_type')
    # get cell types
    cell_types = units_v1['cell_type'].unique()
    cell_types = cell_types[~pd.isnull(cell_types)]

    for cell_type in cell_types:
        select_unit_id = units_v1[units_v1['cell_type']==cell_type]['unit_id'].values
        n_units = len(select_unit_id)
        if n_units<5:
            continue
        # power analysis (cell type)
        for stimulus,table in zip(['drifting_gratings','spontaneous'],[dg_table,sp_table]):
            spike_df_group = power_analysis(table,select_unit_id)
            # calculate mean and se power
            mean_power = np.mean(np.stack(spike_df_group['power'].array),axis=0)
            se_power = np.std(np.stack(spike_df_group['power'].array),axis=0)/np.sqrt(len(spike_df_group))
            f = spike_df_group['f'].iloc[0]
            # add to df
            df_cell_type = pd.concat([df_cell_type,pd.DataFrame({
                'session':i,
                'stimulus':stimulus,
                'cell_type':cell_type,
                'n_units':n_units,
                'mean_power':[mean_power],
                'se_power':[se_power],
                'f':[f],
                })],ignore_index=True)
            

    ############################### for each waveform type and layer #########################################
    print('group by waveform and layer')

    # get unique combination of waveform type and layer
    wf_layer = units_v1[['waveform_type','cortical_layer']].drop_duplicates()

    for row in wf_layer.itertuples():
        wf_type,layer = row.waveform_type,row.cortical_layer
        select_unit_id = units_v1[(units_v1['waveform_type']==wf_type) & (units_v1['cortical_layer']==layer)]['unit_id'].values
        n_units = len(select_unit_id)
        if n_units<5:
            continue
        # power analysis (wf and layer)
        for stimulus,table in zip(['drifting_gratings','spontaneous'],[dg_table,sp_table]):
            spike_df_group = power_analysis(table,select_unit_id)
            # calculate mean and se power
            mean_power = np.mean(np.stack(spike_df_group['power'].array),axis=0)
            se_power = np.std(np.stack(spike_df_group['power'].array),axis=0)/np.sqrt(len(spike_df_group))
            f = spike_df_group['f'].iloc[0]
            # add to to df
            df_wf_layer = pd.concat([df_wf_layer,pd.DataFrame({
                'session':i,
                'stimulus':stimulus,
                'waveform_type':wf_type,
                'cortical_layer':layer,
                'n_units':n_units,
                'mean_power':[mean_power],
                'se_power':[se_power],
                'f':[f],
                })],ignore_index=True)
            
    
    ############################### all units #########################################
    print('all')
    v1unit_id = units_v1.unit_id.values
    n_units = len(v1unit_id)
    # power analysis (all units)
    for stimulus,table in zip(['drifting_gratings','spontaneous'],[dg_table,sp_table]):
        spike_df_group = power_analysis(table,v1unit_id)
        # calculate mean and se power
        mean_power = np.mean(np.stack(spike_df_group['power'].array),axis=0)
        se_power = np.std(np.stack(spike_df_group['power'].array),axis=0)/np.sqrt(len(spike_df_group))
        f = spike_df_group['f'].iloc[0]
        # add to session,stimulus,mean,se,f to df
        df_all = pd.concat([df_all,pd.DataFrame({
            'session':i,
            'stimulus':stimulus,
            'n_units':n_units,
            'mean_power':[mean_power],
            'se_power':[se_power],
            'f':[f],
            })],ignore_index=True)
        
    print(time.time()-t)

# df_all.to_pickle('power_analysis_all.pkl')
# df_cell_type.to_pickle('power_analysis_cell_type.pkl')
# df_wf_layer.to_pickle('power_analysis_wf_layer.pkl')

# %% plot mean power across all sessions
fig = plt.figure(figsize=(4,4),dpi=300)
for c,stimulus in zip(['black','grey'],['drifting_gratings','spontaneous']):
    df_stim = df_all[df_all.stimulus==stimulus]
    mean_power = np.mean(np.stack(df_stim['mean_power'].array),axis=0)
    se_power = np.std(np.stack(df_stim['mean_power'].array),axis=0)/np.sqrt(len(df_stim))
    f = df_stim['f'].iloc[0]
    f_=f[(f>=1)&(f<=99)]
    p = mean_power[(f>=1)&(f<=99)]
    s = se_power[(f>=1)&(f<=99)]
    plt.plot(f_,p,label = stimulus,color=c)
    plt.fill_between(f_,p-s,p+s,alpha=.5,color=c,edgecolor='none')
    plt.axvline(25)
    # plt.plot(f[f>=1],mean_power[f>=1],color=c,label = stimulus)
    # plt.fill_between(f[f>=1],mean_power[f>=1]-se_power[f>=1],mean_power[f>=1]+se_power[f>=1],color=c,alpha=.5)
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (a.u.)')
plt.yscale('log')
plt.title('all units')
plt.xlim([0,100])
#%% plot mean power across all session for each cell type

# spilt cell type into 2 columns by '_'
df_cell_type[['cell','layer']] = df_cell_type['cell_type'].str.split('_',expand=True)
fig,axes = plt.subplots(nrows = 4,ncols=4,figsize=(12,8),dpi=300)
for k,(alpha,stimulus) in enumerate(zip([1,.5],['drifting_gratings','spontaneous'])):
    df_stim = df_cell_type[df_cell_type.stimulus==stimulus]
    for i,(c,cell) in enumerate(zip(['tab:red','tab:blue'],['EXC','PV'])):
        for j,layer in enumerate(['L23','L4','L5','L6']):
            df_cell = df_stim[(df_stim.cell==cell) & (df_stim.layer==layer)]
            mean_power = np.mean(np.stack(df_cell['mean_power'].array),axis=0)
            se_power = np.std(np.stack(df_cell['mean_power'].array),axis=0)/np.sqrt(len(df_cell))
            f = df_cell['f'].iloc[0]
            ax = axes[j,k+2*i]
            f_=f[(f>=1)&(f<=99)]
            p = mean_power[(f>=1)&(f<=99)]
            s = se_power[(f>=1)&(f<=99)]
            ax.plot(f_,p,label = cell+' '+layer,color=c,alpha=alpha)
            ax.fill_between(f_,p-s,p+s,alpha=.5*alpha,color=c,edgecolor='none')
            ax.set_title(cell+' '+layer)
            ax.set_yscale('log')
            ax.set_xlim([0,100])
            ax.axvline(2)


        fig.supxlabel('Frequency (Hz)')
        fig.supylabel('Power (a.u.)')
        fig.tight_layout()
    
# %% plot mean power across all session for each waveform and layer
fig,axes = plt.subplots(nrows = 4,ncols=4,figsize=(12,8),dpi=300)
for k,(alpha,stimulus) in enumerate(zip([1,.5],['drifting_gratings','spontaneous'])):
    df_stim = df_wf_layer[df_wf_layer.stimulus==stimulus]
    for i,(c,wf_type) in enumerate(zip(['tab:orange','tab:green'],['RS','FS'])):
        for j,layer in enumerate([2,4,5,6]):
            df_cell = df_stim[(df_stim.waveform_type==wf_type) & (df_stim.cortical_layer==layer)]
            if len(df_cell)==0:
                continue
            mean_power = np.mean(np.stack(df_cell['mean_power'].array),axis=0)
            se_power = np.std(np.stack(df_cell['mean_power'].array),axis=0)/np.sqrt(len(df_cell))
            f = df_cell['f'].iloc[0]
            ax = axes[j,i+2*k]
            # ax = axes[j,k+2*i]
            f_=f[(f>=1)&(f<=99)]
            p = mean_power[(f>=1)&(f<=99)]
            s = se_power[(f>=1)&(f<=99)]
            ax.plot(f_,p,color=c,alpha=alpha)
            ax.fill_between(f_,p-s,p+s,alpha=.5*alpha,color=c,edgecolor='none')
            ax.set_title(wf_type+' layer='+str(layer))
            ax.set_yscale('log')
            ax.set_xlim([0,100])
    fig.supxlabel('Frequency (Hz)')
    fig.supylabel('Power (a.u.)')
    fig.tight_layout()
