import pandas as pd
import numpy as np

def calculate_ms2_summary(df):
    """
    Calculate summary properties for features from MS2 data
    """
    
    spectra = {'label':[], 
               'spectrum':[], 
               'rt':[], 
               'precursor_mz':[],
               'precursor_peak_height':[]}
    
    for label_group, label_data in df[df['in_feature']==True].groupby('label'):
        
        label = label_group
        
        for rt_group, rt_data in pd.DataFrame(label_data).groupby('rt'):
            
            mz = np.array(rt_data.mz.tolist())
            i = np.array(rt_data.i.tolist())
        
            mzi = np.array([mz, i])
        
            spectra['label'].append(label)
            spectra['spectrum'].append(mzi)
            spectra['rt'].append(rt_group)
            
            spectra['precursor_mz'].append(rt_data.precursor_MZ.median())
            spectra['precursor_peak_height'].append(rt_data.precursor_intensity.median())
        
    return pd.DataFrame(spectra)


def calculate_ms1_summary(row):
    """
    Calculate summary properties for features from data
    """
    d = {}
    #Before doing this make sure "in_feature"==True has already occured
    d['num_datapoints'] = row['i'].count()
    d['peak_area'] = row['i'].sum()
    idx = row['i'].idxmax()
    d['peak_height'] = row.loc[idx,'i']
    d['mz_centroid'] = sum(row['i']*row['mz'])/d['peak_area']
    d['rt_peak'] = row.loc[idx,'rt']
    return pd.Series(d)