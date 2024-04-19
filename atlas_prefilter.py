import pandas as pd
import numpy as np
import glob
import os
from typing import TypeAlias
from tqdm.notebook import tqdm

import matchms as mms
from matchms.similarity import CosineHungarian

from metatlas.io import feature_tools as ft
from feature_tools_addition import calculate_ms2_summary

# Typing:
MS2Spectrum: TypeAlias = np.ndarray[np.ndarray, np.ndarray]


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


def subset_file_paths(raw_data_dir: str, experiment:str, polarity: str) -> tuple[list[str, ...], list[str, ...]]:
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


def get_rt_alignment_ms1_data(rt_alignment_atlas: pd.DataFrame, qc_files: list[str, ...], ppm_tolerance: int,
                               extra_time: float, polarity: str) -> pd.DataFrame:
    """Collect all MS1 feature data for each entry in the retention time adjustment atlas."""

    experiment_input = ft.setup_file_slicing_parameters(rt_alignment_atlas, qc_files, base_dir=os.getcwd(), ppm_tolerance=ppm_tolerance, extra_time=extra_time, polarity=polarity)

    ms1_data = []
    for file_input in experiment_input:
        
        data = ft.get_data(file_input, save_file=False, return_data=True)
        data['ms1_summary']['lcmsrun_observed'] = file_input['lcmsrun']
        
        ms1_data.append(data['ms1_summary'])
        
    return pd.concat(ms1_data)


def align_rt_adjustment_peaks(ms1_data: pd.DataFrame, rt_alignment_atlas: pd.DataFrame) -> tuple[list[float, ...], list[float, ...]]:
    """align median experimental retention time peaks with rt adjustment atlas peaks."""
    
    median_experimental_rt_peaks = ms1_data[ms1_data['peak_height'] >= 1e4].groupby('label')['rt_peak'].median()
    rt_peaks_merged = pd.merge(rt_alignment_atlas[['label', 'rt_peak']], median_experimental_rt_peaks, on='label')

    original_rt_peaks = rt_peaks_merged['rt_peak_x'].tolist()
    experimental_rt_peaks = rt_peaks_merged['rt_peak_y'].tolist()

    return (original_rt_peaks, experimental_rt_peaks)


def adjust_template_atlas_rt_peaks(template_atlas: pd.DataFrame, original_rt_peaks: list[float, ...], experimental_rt_peaks:list[float, ...], 
                     rt_regression: bool, model_degree: int, rt_window: float) -> pd.DataFrame:
    """Build and use model to adjust template atlas retention time peaks to match experimental retention time space."""

    aligned_template_atlas = template_atlas.copy()

    if rt_regression:
        rt_alignment_model = np.polyfit(original_rt_peaks, experimental_rt_peaks, model_degree)
        aligned_template_atlas['rt_peak'] = aligned_template_atlas['rt_peak'].apply(lambda x: np.polyval(rt_alignment_model, x))
    
    else:
        median_offset = np.median(np.array(original_rt_peaks) - np.array(experimental_rt_peaks))
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
    for file_input in tqdm(experiment_input, unit="file"):
        
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


#Note, replace with new load function from metatlas once msms hits refactor is in place
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
    msms_refs_filtered['spectrum'] = msms_refs_filtered['spectrum'].apply(order_ms2_spectrum)

    return msms_refs_filtered


#Note: This will need to come from the Metatlas DB in the final version.
def load_atlas_files(template_atlas_path: str | os.PathLike, rt_alignment_atlas_path: str | os.PathLike) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load template and retention time alignment atlese from tsv files.
    
    This function will need to be evetually replaced with a database retreival function.
    """
    
    template_atlas = pd.read_csv(template_atlas_path, sep='\t')
    rt_alignment_atlas = pd.read_csv(rt_alignment_atlas_path, sep='\t')
    
    return (template_atlas, rt_alignment_atlas)


def calculate_ms2_scores(ms2_data_enriched: pd.DataFrame, frag_tolerance: float) -> list[float, ...]:
    """Calculate cosine similarity scores for each feature in the sample MS2 data.
    
    To maintain parity with the MSMS hits collection, scoring is performed using the Hungarian alignment algorithm.
    """

    cosine_hungarian = CosineHungarian(tolerance=frag_tolerance)
    scores = ms2_data_enriched.apply(lambda x: cosine_hungarian.pair(x.mms_spectrum_x, x.mms_spectrum_y), axis=1)
    
    return scores


def enrich_ms2_data(msms_refs_path: str, polarity: str, frag_tolerance: float,
                    aligned_template_atlas: pd.DataFrame, ms2_data: pd.DataFrame) -> pd.DataFrame:
    """Enrich collected MS2 data with InChi Keys and MSMS refs information.
    
    This function merges the MSMS refs with the MS2 data collected from the samples for scoring.
    Additionally, spectral data are converted to MatchMS Spectrum objects.
    """

    msms_refs = load_msms_refs_file(msms_refs_path, polarity)

    ms2_data_enriched = pd.merge(ms2_data, aligned_template_atlas[['label', 'inchi_key']], on='label')
    ms2_data_enriched = pd.merge(ms2_data_enriched, msms_refs[['id', 'inchi_key', 'spectrum']], on='inchi_key')

    ms2_data_enriched['mms_spectrum_x'] = ms2_data_enriched.apply(lambda x: mms.Spectrum(x.spectrum_x[0], x.spectrum_x[1], metadata={'precursor_mz':x.precursor_mz}), axis=1)
    ms2_data_enriched['mms_spectrum_y'] = ms2_data_enriched.apply(lambda x: mms.Spectrum(x.spectrum_y[0], x.spectrum_y[1], metadata={'precursor_mz':x.precursor_mz}), axis=1)
    
    ms2_data_enriched['mms_out'] = calculate_ms2_scores(ms2_data_enriched, frag_tolerance)
    
    ms2_data_enriched['score'] = ms2_data_enriched['mms_out'].apply(lambda x: x['score'])
    ms2_data_enriched['matches'] = ms2_data_enriched['mms_out'].apply(lambda x: x['matches'])

    return ms2_data_enriched


def filter_atlas_labels(ms1_data: pd.DataFrame, ms2_data_enriched: pd.DataFrame, 
                       peak_height: float, num_points: int, msms_filter: bool, msms_score: float, msms_matches: int) -> set[str, ...]:
    """Filter atlas labels to include only those that pass the MS1 and MS2 thresholds."""
    
    ms1_data_filtered = ms1_data[(ms1_data['peak_height'] >= peak_height) & (ms1_data['num_datapoints'] >= num_points)]
    ms1_reduced_labels = set(ms1_data_filtered.label.tolist())
    
    if msms_filter:
        ms2_data_filtered = ms2_data_enriched[(ms2_data_enriched['score'] >= msms_score) & (ms2_data_enriched['matches'] >= msms_matches)]
        ms2_reduced_labels = set(ms2_data_filtered.label.tolist())
    else:
        ms2_reduced_labels = ms1_reduced_labels
        
    reduced_labels = ms1_reduced_labels.intersection(ms2_reduced_labels)
    
    return reduced_labels


# Note: this will need to be replaced with a function to add atlas to the database rather than save as csv
def save_reduced_atlas(aligned_template_atlas: pd.DataFrame, reduced_labels: set[str, ...], experiment: str, polarity: str) -> None:
    """Save retention time aligned and filtered template atlas.
    
    In the future, this will be replace with adding the new atlas to the metatlas database
    """
    
    aligned_template_atlas = aligned_template_atlas[aligned_template_atlas['label'].isin(reduced_labels)]
    
    atlas_cols = ['label', 'adduct', 'polarity', 'mz', 'rt_peak', 'rt_min', 'rt_max', 'inchi_key']

    if not os.path.exists(experiment):
        os.mkdir(experiment)

    aligned_template_atlas[atlas_cols].to_csv(os.path.join(experiment, '{}_reduced_atlas.csv'.format(polarity)), index=False)
    
def generate_outputs(raw_data_dir: str,
                     experiment: str,
                     msms_refs_path: str | os.PathLike,
                     template_atlas_path: str | os.PathLike,
                     rt_alignment_atlas_path: str | os.PathLike,
                     polarity: str,
                     ppm_tolerance: int,
                     extra_time: float,
                     rt_window: float,
                     peak_height: float,
                     num_points: int,
                     msms_filter: bool,
                     msms_score: float,
                     msms_matches: int,
                     frag_tolerance: float,
                     rt_regression: bool,
                     model_degree: int) -> None:
    """Generate reduced retention time aligned reduced template atlas."""
    
    
    template_atlas, rt_alignment_atlas = load_atlas_files(template_atlas_path, rt_alignment_atlas_path)
    sample_files, qc_files = subset_file_paths(raw_data_dir, experiment, polarity)
    
    rt_alignment_ms1_data = get_rt_alignment_ms1_data(rt_alignment_atlas, qc_files, ppm_tolerance, extra_time, polarity)
    original_rt_peaks, experimental_rt_peaks = align_rt_adjustment_peaks(rt_alignment_ms1_data, rt_alignment_atlas)
    
    aligned_template_atlas = adjust_template_atlas_rt_peaks(template_atlas, original_rt_peaks, experimental_rt_peaks, rt_regression, model_degree, rt_window)
    
    ms1_data, ms2_data = get_experimental_ms_data(aligned_template_atlas, sample_files, ppm_tolerance, extra_time, polarity)
    ms2_data_enriched = enrich_ms2_data(msms_refs_path, polarity, frag_tolerance, aligned_template_atlas, ms2_data)
    
    reduced_labels = filter_atlas_labels(ms1_data, ms2_data_enriched, peak_height, num_points, msms_filter, msms_score, msms_matches)
    
    save_reduced_atlas(aligned_template_atlas, reduced_labels, experiment, polarity)
    