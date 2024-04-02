import pandas as pd
import numpy as np
import glob
import os
from typing import TypeAlias
from tqdm import tqdm

import matchms as mms
from matchms.similarity import CosineHungarian

from metatlas.metatlas.io import feature_tools as ft
from feature_tools_addition import calculate_ms2_summary

# Typing:
MS2Spectrum: TypeAlias = np.ndarray[np.ndarray, np.ndarray]


## Set atlas pre-filter parameters. This will eventually be derived from the SLURM input and config file paramters
## In a future version, both the template C18 atlas and the RT adjustment atlases need to be retrieved from the MySQL database on NERSC

raw_data_dir = '/global/cfs/cdirs/metatlas/raw_data/jgi'
experiment = '20231103_JGI_MW_507961_Char_final-dil_EXP120B_C18_USDAY81385'

msms_refs_path = '/global/cfs/cdirs/metatlas/projects/spectral_libraries/20240222_labeled-addition_msms_refs.tab'
c18_base_atlas_dir = './metatlas-data/C18/'

# either 'positive' or 'negative'
polarity = 'positive'

# filtering hit generation
ppm_tolerance = 5
extra_time = 0.5

# filtering hits
rt_window = 0.5
peak_height = 5e5
num_points = 4.0

msms_filter = True
msms_score = 0.65
msms_matches = 4
frag_tolerance = 0.02

# use regression for rt alignment, otherwise use median offset
rt_regression = False

# rt alignment model degree if using regression
model_degree = 1


def order_ms2_spectrum(spectrum: MS2Spectrum) -> MS2Spectrum:
    """Order spectrum by m/z from lowest to highest.

    Ordering spectrum by m/z prevents MatchMS errors during MS/MS scoring.
    """
    order_idx = np.argsort(spectrum[0])
    ordered_spec = np.array([spectrum[0][order_idx], spectrum[1][order_idx]])
    
    return ordered_spec


def extract_file_polarity(file_path: str) -> str:
    """Extract file polarity from file path.
    
    Per file naming conventions, field 9 of each (underscore delimited) filename contains the polarity information for that particular file.
    """
    return os.path.basename(file_path).split('_')[9]


def subset_file_paths(all_files: list, polarity: str) -> tuple[list[str, ...], list[str, ...]]:
    """Return lists of QC file paths and sample file paths filtered by polarity.

    Filter polarity is used in this case to retrieve both fast polarity switching (FPS) files and files matching the defined polarity
    """

    if polarity == 'positive':
        file_polarity = 'POS'
        filter_polarity = 'NEG'
    else:
        file_polarity = 'NEG'
        filter_polarity = 'POS'

    all_files = glob.glob(os.path.join(raw_data_dir, experiment, '*.h5'))

    sample_files = [file for file in all_files if extract_file_polarity(file) == file_polarity and 'QC' not in file]
    qc_files = [file for file in all_files if extract_file_polarity(file) != filter_polarity and 'QC_' in file]

    return (sample_files, qc_files)


def get_rt_adjustment_ms1_data(rt_adjustment_atlas: pd.DataFrame, qc_files: list[str, ...], ppm_tolerance: int,
                               extra_time: float, polarity: str) -> pd.DataFrame:
    """Collect all MS1 feature data for each entry in the retention time adjustment atlas."""

    experiment_input = ft.setup_file_slicing_parameters(rt_adjustment_atlas, qc_files, base_dir=os.getcwd(), ppm_tolerance=ppm_tolerance, extra_time=extra_time, polarity=polarity)

    ms1_data = []
    for file_input in experiment_input:
        
        data = ft.get_data(file_input, save_file=False, return_data=True)
        data['ms1_summary']['lcmsrun_observed'] = file_input['lcmsrun']
        
        ms1_data.append(data['ms1_summary'])
        
    return pd.concat(ms1_data)


def align_rt_adjustment_peaks(ms1_data: pd.DataFrame, rt_adjustment_atlas: pd.DataFrame) -> tuple[list[float, ...], list[float, ...]]:
    """align median experimental retention time peaks with rt adjustment atlas peaks."""
    
    median_experimental_rt_peaks = ms1_data[ms1_data['peak_height'] >= 1e4].groupby('label')['rt_peak'].median()
    rt_peaks_merged = pd.merge(rt_adjustment_atlas[['label', 'rt_peak']], median_experimental_rt_peaks, on='label')

    original_rt_peaks = rt_peaks_merged['rt_peak_x'].tolist()
    experimental_rt_peaks = rt_peaks_merged['rt_peak_y'].tolist()

    return (original_rt_peaks, experimental_rt_peaks)


def adjust_template_atlas_rt_peaks(template_atlas: pd.DataFrame, original_rt_peaks: list[float, ...], experimental_rt_peaks:list[float, ...], 
                     rt_regression: bool, model_degree: int) -> pd.DataFrame:
    """Build and use model to adjust template atlas retention time peaks to match experimental retention time space."""

    aligned_template_atlas = template_atlas.copy()

    if rt_regression:
        rt_alignment_model = np.polyfit(original_rt_peaks, experimental_rt_peaks, model_degree)
        aligned_template_atlas['rt_peak'] = aligned_template_atlas['rt_peak'].apply(lambda x: np.polyval(rt_alignment_model, x))
    
    else:
        median_offset = (original_rt_peaks - experimental_rt_peaks).median()
        aligned_template_atlas['rt_peak'] = aligned_template_atlas['rt_peak'] + median_offset

    aligned_template_atlas['rt_min'] = aligned_template_atlas['rt_peak'] - rt_window
    aligned_template_atlas['rt_max'] = aligned_template_atlas['rt_peak'] + rt_window

    return aligned_template_atlas


# Note: refactor this function after adding the calculate ms2 summary function to feature tools
def get_experimental_ms_data(aligned_template_atlas: pd.DataFrame, sample_files: list[str, ...], 
                             ppm_tolerance: int, extra_time: float, polarity: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect MS1 and MS2 data from experimental sample data using aligned atlas."""

    experiment_input = ft.setup_file_slicing_parameters(aligned_template_atlas, sample_files, base_dir=os.getcwd(), ppm_tolerance=ppm_tolerance, extra_time=extra_time, polarity=polarity)

    ms1_data = []
    ms2_data = []

    # Note: disable tqdm in papermill for future version
    for file_input in tqdm(experiment_input):
        
        data = ft.get_data(file_input, save_file=False, return_data=True)
        
        data['ms1_summary']['lcmsrun_observed'] = file_input['lcmsrun']
        
        ms2_summary = calculate_ms2_summary(data['ms2_data'])
        ms2_summary['lcmsrun_observed'] = file_input['lcmsrun']
        
        ms1_data.append(data['ms1_summary'])
        ms2_data.append(ms2_summary)
        
    ms1_data = pd.concat(ms1_data)
    ms2_data = pd.concat(ms2_data)
    ms2_data['spectrum'] = ms2_data['spectrum'].apply(order_ms2_spectrum)

    return (ms1_data, ms2_data)


#note, replace with new load function from metatlas once msms hits refactor is in place
def load_msms_refs_file(msms_refs_path: str, polarity: str) -> pd.DataFrame:
    """Load and filter MSMS refs file.
    
    In addition to loading and filtering MSMS refs, spectral data is converted to Numpy array format.
    """

    ref_dtypes = {'database': str, 'id': str, 'name': str,
                          'spectrum': object, 'decimal': int, 'precursor_mz': float,
                          'polarity': str, 'adduct': str, 'fragmentation_method': str,
                          'collision_energy': str, 'instrument': str, 'instrument_type': str,
                          'formula': str, 'exact_mass': float,
                          'inchi_key': str, 'inchi': str, 'smiles': str}
    
    msms_refs_df = pd.read_csv(msms_refs_path, sep='\t', dtype=ref_dtypes)
    msms_refs_filtered = msms_refs_df[(msms_refs_df['database'] == 'metatlas') & (msms_refs_df['polarity'] == polarity)].copy()
    
    msms_refs_filtered['spectrum'] = msms_refs_filtered['spectrum'].apply(lambda x: np.asarray(eval(x)))
    msms_refs_filtered['spectrum'] = msms_refs_filtered.apply(order_ms2_spectrum, axis=1)

    return msms_refs_filtered


def calculate_ms2_scores(ms2_data_enriched: pd.DataFrame, frag_tolerance: float) -> list[float, ...]:
    """Calculate cosine similarity scores for each feature in the sample MS2 data.
    
    To maintain parity with the MSMS hits collection, scoring is performed using the Hungarian alignment algorithm.
    """

    cosine_hungarian = CosineHungarian(tolerance=frag_tolerance)
    scores = ms2_data_enriched.apply(lambda x: cosine_hungarian.pair(x.mms_spectrum_x, x.mms_spectrum_y), axis=1)
    
    return scores


def enrich_ms2_data(msms_refs_path: str, polarity: str, 
                    aligned_template_atlas: pd.DataFrame, ms2_data: pd.DataFrame) -> pd.DataFrame:
    """Enrich collected MS2 data with InChi Keys and MSMS refs information.
    
    This function merges the MSMS refs with the MS2 data collected from the samples for scoring.
    Additionally, spectral data are converted to MatchMS Spectrum objects.
    """

    msms_refs = load_msms_refs_file(msms_refs_path, polarity)

    ms2_data_enriched = pd.merge(ms2_data, aligned_template_atlas[['label', 'inchi_key']], on='label')
    ms2_data_enriched = pd.merge(ms2_data, msms_refs[['id', 'inchi_key', 'spectrum']], on='inchi_key')

    ms2_data_enriched['mms_spectrum_x'] = ms2_data_enriched.apply(lambda x: mms.Spectrum(x.spectrum_x[0], x.spectrum_x[1], metadata={'precursor_mz':x.precursor_mz}), axis=1)
    ms2_data_enriched['mms_spectrum_y'] = ms2_data_enriched.apply(lambda x: mms.Spectrum(x.spectrum_y[0], x.spectrum_y[1], metadata={'precursor_mz':x.precursor_mz}), axis=1)

    return ms2_data_enriched
    
