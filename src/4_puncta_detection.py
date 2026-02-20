"""
Detect and analyze features of puncta per cell
"""

import os
import importlib.util
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import skimage.io
from skimage import measure, segmentation, morphology
from skimage.morphology import remove_small_objects
from scipy import stats
from scipy.stats import skewtest
from loguru import logger
import functools
# special import, path to script
napari_utils_path = 'punctalyze-SRT/src/3_napari.py' # adjust as needed

# load the module dynamically due to annoying file name
spec = importlib.util.spec_from_file_location("napari", napari_utils_path)
napari_utils = importlib.util.module_from_spec(spec)
sys.modules["napari_utils"] = napari_utils
spec.loader.exec_module(napari_utils)
remove_saturated_cells = napari_utils.remove_saturated_cells

logger.info('import ok')

# plotting setup
plt.rcParams.update({'font.size': 14})
sns.set_palette('Paired')

# --- configuration ---
STD_THRESHOLD = 2.7
SAT_FRAC_CUTOFF = 0.01  # for consistency with remove_saturated_cells
COI_1 = 0  # channel of interest for saturation check (e.g., 1 for channel 2) (puncta)
COI_2 = 1  # secondary channel of interest for comparisons (other)
COI_1_name = 'coilin'  # name of the first channel of interest, for plotting
COI_2_name  = 'sumo1'  # name of the second channel of interest, for plotting
SCALE_PX = (294.67/2720) # size of one pixel in units specified by the next constant
#--- puncta shape filtering ---
MIN_PUNCTA_SIZE = 3  # minimum size of puncta
MAX_PUNCTA_SIZE = 300 #px, tune
MIN_CIRCULARITY = 0.55 # 1.0 = perfect circle
MAX_ECCENTRICITY = 0.9 # 0 = circle; 1 = line
MIN_SOLIDITY = 0.7 #1 = solid; lower = ragged/fragmented 
MIN_ASPECT_RATIO = 0.45   #minor/major; 1 is circle
SHOW_SURVIVORS_ONLY = True #proofs show puncta masks after filtering

# --- quantification region ---
# options: "cell", "nucleus"
QUANT_REGION = "nucleus"

SCALE_UNIT = 'um'  # units for the scale bar
image_folder = 'results/initial_cleanup/'
mask_folder = 'results/napari_masking/'
output_folder = 'results/summary_calculations/'
proofs_folder = 'results/proofs/'

for folder in [output_folder, proofs_folder]:
    if not os.path.exists(folder):
        os.mkdir(folder)


def feature_extractor(mask, properties=None):
    if properties is None:
        properties = [
            'area', 'eccentricity', 'solidity', 'label',
            'major_axis_length', 'minor_axis_length',
            'perimeter', 'coords'
        ]
    props = measure.regionprops_table(mask, properties=properties)
    return pd.DataFrame(props)


def load_images(image_folder):
    images = {}
    for fn in os.listdir(image_folder):
        if fn.endswith('.npy'):
            name = fn.removesuffix('.npy')
            images[name] = np.load(f'{image_folder}/{fn}')
    return images


def load_masks(mask_folder):
    masks = {}
    for fn in os.listdir(mask_folder):
        if fn.endswith('_mask.npy'):
            name = fn.removesuffix('_mask.npy')
            masks[name] = np.load(f'{mask_folder}/{fn}', allow_pickle=True)
    return masks

def build_quant_masks(masks, region="cell"):
    """
    Returns label masks defining the region in which puncta detection happens.
    """
    quant_masks = {}

    for name, m in masks.items():
        cell_mask, nuc_mask = m[0], m[1]

        if region == "cell":
            quant_masks[name] = cell_mask

        elif region == "nucleus":
            # ensure labeled nuclei (not just binary)
            quant_masks[name] = morphology.label(nuc_mask > 0)

        else:
            raise ValueError(f"Unknown QUANT_REGION: {region}")

    return quant_masks

def generate_cytoplasm_masks(masks):
    logger.info('removing nuclei from cell masks...')
    cyto_masks = {}
    for name, img in masks.items():
        cell_mask, nuc_mask = img[0], img[1]
        cell_bin = (cell_mask > 0).astype(int) # make binary masks
        nuc_bin = (nuc_mask > 0).astype(int)

        single_cyto = []
        labels = np.unique(cell_mask)
        if labels.size > 1:
            for lbl in labels[labels != 0]:
                cyto = np.where(cell_mask == lbl, cell_bin, 0)
                cyto_minus_nuc = cyto & ~nuc_bin
                if np.any(cyto_minus_nuc):
                    single_cyto.append(np.where(cyto_minus_nuc, lbl, 0))
                else:
                    single_cyto.append(np.zeros_like(cell_mask, dtype=int))
        else:
            single_cyto.append(np.zeros_like(cell_mask, dtype=int))

        cyto_masks[name] = sum(single_cyto)
    logger.info('cytoplasm masks created.')
    return cyto_masks


def filter_saturated_images(images, cytoplasm_masks, masks):
    logger.info('filtering saturated cells...')
    filtered = {}
    for name, img in images.items():
        # Build a stack: [stain, coi, cytoplasm mask]
        stack = np.stack([
            img[COI_2], img[COI_1], cytoplasm_masks[name]
        ])
        # apply imported saturation check function
        cells = remove_saturated_cells(
            image_stack=stack,
            mask_stack=masks[name],
            COI=COI_1
        )
        filtered[name] = np.stack([img[COI_2], img[COI_1], cells])
    logger.info('saturated cells filtered.')
    return filtered

def collect_features(image_dict, STD_THRESHOLD=STD_THRESHOLD):
    logger.info('collecting cell & puncta features...')
    results = []
    for name, img in image_dict.items():
        coi2, coi1, mask = img
        unique_cells = np.unique(mask)[1:]
        contours = measure.find_contours((mask > 0).astype(int), 0.8)
        contour = [c for c in contours if len(c) >= 100]

        for lbl in unique_cells:
            cell_mask = mask == lbl
            coi1_vals = coi1[cell_mask]
            mean_coi1 = coi1_vals.mean()
            std_coi1 = coi1_vals.std()

            threshold = (std_coi1 * STD_THRESHOLD) + mean_coi1
            binary = (coi1 > threshold) & cell_mask

            puncta_labels = morphology.label(binary)
            puncta_labels = remove_small_objects(puncta_labels, min_size=MIN_PUNCTA_SIZE)

            df_p = feature_extractor(puncta_labels).add_prefix('puncta_')

            # --- size filter ---
            if not df_p.empty:
                size_keep = (
                    (df_p['puncta_area'] >= MIN_PUNCTA_SIZE) &
                    (df_p['puncta_area'] <= MAX_PUNCTA_SIZE)
                )
                kept_labels = df_p.loc[size_keep, 'puncta_label'].astype(int).to_numpy()

                puncta_labels = np.where(np.isin(puncta_labels, kept_labels), puncta_labels, 0)
                puncta_labels = morphology.label(puncta_labels > 0)

                df_p = feature_extractor(puncta_labels).add_prefix('puncta_')

            # --- shape filter ---
            if not df_p.empty:
                df_p['puncta_circularity'] = (4 * np.pi * df_p['puncta_area']) / (df_p['puncta_perimeter']**2 + 1e-9)
                df_p['puncta_aspect_ratio'] = df_p['puncta_minor_axis_length'] / (df_p['puncta_major_axis_length'] + 1e-9)

                shape_keep = (
                    (df_p['puncta_circularity'] >= MIN_CIRCULARITY) &
                    (df_p['puncta_eccentricity'] <= MAX_ECCENTRICITY) &
                    (df_p['puncta_solidity'] >= MIN_SOLIDITY) &
                    (df_p['puncta_aspect_ratio'] >= MIN_ASPECT_RATIO)
                )
                kept_labels = df_p.loc[shape_keep, 'puncta_label'].astype(int).to_numpy()

                puncta_labels = np.where(np.isin(puncta_labels, kept_labels), puncta_labels, 0)
                puncta_labels = morphology.label(puncta_labels > 0)

                df_p = feature_extractor(puncta_labels).add_prefix('puncta_')

            # if nothing left after filtering, skip this cell
            if df_p.empty:
                continue

            # --- intensity stats per punctum ---
            stats_columns = [
                'puncta_cv',
                'puncta_skew',
                'puncta_intensity_mean',
                'puncta_intensity_mean_in_coi2'
            ]

            stats_list = []
            for _, row in df_p.iterrows():
                p_mask = puncta_labels == row['puncta_label']
                puncta_vals = coi1[p_mask]
                cv = puncta_vals.std() / puncta_vals.mean() if puncta_vals.mean() != 0 else np.nan
                skew_stat = skewtest(puncta_vals).statistic if len(puncta_vals) >= 8 else np.nan
                mean_p = puncta_vals.mean()
                mean_coi2 = coi2[p_mask].mean()
                stats_list.append((cv, skew_stat, mean_p, mean_coi2))

            df_stats = pd.DataFrame(stats_list, columns=stats_columns)

            df = pd.concat([df_p.reset_index(drop=True), df_stats], axis=1)
            df['image_name'], df['cell_number'] = name, lbl
            df['cell_size'] = cell_mask.sum()
            df['cell_std'] = std_coi1
            df['cell_cv'] = std_coi1 / mean_coi1
            df['cell_skew'] = skewtest(coi1_vals).statistic if len(coi1_vals) >= 8 else np.nan
            df['cell_coi1_intensity_mean'] = mean_coi1
            df['cell_coi2_intensity_mean'] = (coi2[cell_mask]).mean()
            df['cell_coords'] = [contour] * len(df)

            results.append(df)

    logger.info('feature extraction done.')
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def extra_puncta_features(df):
    df = df.copy()  # avoid modifying in place
    df['puncta_aspect_ratio'] = df['puncta_minor_axis_length'] / df['puncta_major_axis_length']
    df['puncta_circularity'] = (4 * np.pi * df['puncta_area']) / (df['puncta_perimeter']**2 + 1e-9)
    df['coi2_partition_coeff'] = df['puncta_intensity_mean_in_coi2'] / df['cell_coi2_intensity_mean']
    df['coi1_partition_coeff'] = df['puncta_intensity_mean'] / df['cell_coi1_intensity_mean']

    return df


def aggregate_features_by_group(df, group_cols, agg_cols, agg_func='mean'):
    """
    Aggregate multiple columns by group and merge results into a single DataFrame.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        group_cols (list): Columns to group by.
        agg_cols (list): Columns to aggregate.
        agg_func (str or callable): Aggregation function, default is 'mean'.

    Returns:
        pd.DataFrame: Aggregated dataframe with group_cols and agg_cols.
    """
    grouped_dfs = []
    for col in agg_cols:
        agg_df = df.groupby(group_cols)[col].agg(agg_func).reset_index()
        grouped_dfs.append(agg_df)

    merged_df = functools.reduce(
        lambda left, right: left.merge(right, on=group_cols),
        grouped_dfs
    )
    return merged_df.reset_index(drop=True)


# --- Proof Plotting ---

def get_surviving_puncta_labels_for_image(puncta_img, mask_labels):
    """
    Returns a labeled puncta mask for the whole image AFTER size+shape filtering,
    computed per object in mask_labels (each nucleus/cell).
    """
    out = np.zeros_like(puncta_img, dtype=int)
    next_id = 1

    for lbl in np.unique(mask_labels):
        if lbl == 0:
            continue

        region = (mask_labels == lbl)
        vals = puncta_img[region]
        if vals.size == 0:
            continue

        mean_v = vals.mean()
        std_v = vals.std()
        thr = mean_v + STD_THRESHOLD * std_v

        binary = (puncta_img > thr) & region
        plab = morphology.label(binary)
        plab = remove_small_objects(plab, min_size=MIN_PUNCTA_SIZE)

        df_p = feature_extractor(plab).add_prefix("puncta_")
        if df_p.empty:
            continue

        # size filter
        size_keep = (
            (df_p["puncta_area"] >= MIN_PUNCTA_SIZE) &
            (df_p["puncta_area"] <= MAX_PUNCTA_SIZE)
        )
        kept = df_p.loc[size_keep, "puncta_label"].astype(int).to_numpy()
        plab = np.where(np.isin(plab, kept), plab, 0)
        plab = morphology.label(plab > 0)

        df_p = feature_extractor(plab).add_prefix("puncta_")
        if df_p.empty:
            continue

        # shape filter
        df_p["puncta_circularity"] = (4 * np.pi * df_p["puncta_area"]) / (df_p["puncta_perimeter"]**2 + 1e-9)
        df_p["puncta_aspect_ratio"] = df_p["puncta_minor_axis_length"] / (df_p["puncta_major_axis_length"] + 1e-9)

        shape_keep = (
            (df_p["puncta_circularity"] >= MIN_CIRCULARITY) &
            (df_p["puncta_eccentricity"] <= MAX_ECCENTRICITY) &
            (df_p["puncta_solidity"] >= MIN_SOLIDITY) &
            (df_p["puncta_aspect_ratio"] >= MIN_ASPECT_RATIO)
        )
        kept = df_p.loc[shape_keep, "puncta_label"].astype(int).to_numpy()
        plab = np.where(np.isin(plab, kept), plab, 0)
        plab = morphology.label(plab > 0)

        # merge into image-level labels (ensure unique IDs)
        for k in np.unique(plab):
            if k == 0:
                continue
            out[plab == k] = next_id
            next_id += 1

    return out
def generate_proofs(df, image_dict):
    logger.info('Generating proof plots...')
    for name, img in image_dict.items():

        # unpack from your filtered dict/stack format
        # If image_dict stores np.stack([coi2, coi1, mask]) like your code:
        coi2, coi1, mask = img

        # Pull ONE set of region contours from df (you saved per-row, so take first)
        contour_series = df.loc[df['image_name'] == name, 'cell_coords']
        if contour_series.empty:
            continue
        contour = contour_series.iloc[0]

        # image for right panel: puncta channel only inside region
        region_img = coi1 * (mask > 0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

        # LEFT: overlay (puncta channel in gray, other channel in blue)
        ax1.imshow(coi1, cmap='gray_r')
        ax1.imshow(coi2, cmap='Blues', alpha=0.6)

        # RIGHT: region + outlines
        ax2.imshow(region_img, cmap='gray_r')
        for line in contour:
            ax2.plot(line[:, 1], line[:, 0], c='k', lw=0.5)

        # --- SURVIVORS overlay (post-filter) ---
        if SHOW_SURVIVORS_ONLY:
            survivors = get_surviving_puncta_labels_for_image(coi1, mask)

            # draw red outlines for surviving puncta
            for c in measure.find_contours((survivors > 0).astype(float), 0.5):
                ax2.plot(c[:, 1], c[:, 0], color='red', lw=1.0)

            n_survive = (np.unique(survivors).size - 1)
        else:
            n_survive = None

        # scalebar + labels on LEFT panel
        scalebar = ScaleBar(
            SCALE_PX, SCALE_UNIT, location='lower right',
            pad=0.3, sep=2, box_alpha=0, color='gray',
            length_fraction=0.3
        )
        ax1.add_artist(scalebar)
        ax1.text(50, 2000, COI_1_name, color='gray')
        ax1.text(50, 1800, COI_2_name, color='steelblue')

        # title
        if n_survive is not None:
            fig.suptitle(f'{name} | survivors: {n_survive}', y=0.88)
        else:
            fig.suptitle(name, y=0.88)

        fig.tight_layout()
        fig.savefig(f'{proofs_folder}{name}_proof.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    logger.info('proofs saved.')


if __name__ == '__main__':
    logger.info('loading images and masks...')
    images = load_images(image_folder)
    masks = load_masks(mask_folder)

    # # -- generate and filter cytoplasm masks ---
    # cyto_masks = generate_cytoplasm_masks(masks)
    # filtered = filter_saturated_images(images, cyto_masks, masks)

    # -- OR collect and filter cell masks ---
    quant_masks = build_quant_masks(masks, QUANT_REGION)
    filtered = filter_saturated_images(images, quant_masks, masks)

    # --- feature extraction ---
    features = collect_features(filtered)
    if features.empty:
        logger.warning("No puncta detected after filtering; nothing to save/plot.")
        sys.exit(0)
    features = extra_puncta_features(features)

    # --- generate proofs ---
    generate_proofs(features, filtered, coi1=COI_1, coi2=COI_2)
    logger.info('proofs complete.')

    # --- data wrangling ---
    logger.info('starting data wrangling and saving...')
   
    # features['condition'] = features['image_name'].str.split('-').str[0]
    # features['rep'] = features['image_name'].str.split('-').str[-2]

    cols = features.columns.tolist()
    cols = [item for item in cols if '_coords' not in item]
    cols = ['puncta_area', 'puncta_eccentricity', 'puncta_aspect_ratio',
            'puncta_circularity', 'puncta_cv', 'puncta_skew',
            'coi2_partition_coeff', 'coi1_partition_coeff', 'cell_std',
            'cell_cv', 'cell_skew']
    
    # # remove outliers based on z-score
    # features = features[(np.abs(stats.zscore(features[cols[:-1]])) < 3).all(axis=1)]

    # --- data trimming and saving ---
    # trim off coordinates used for proofs, save the main features dataframe
    cols_to_drop = [col for col in features.columns if '_coords' in col]
    features = features.drop(columns=cols_to_drop)
    features.to_csv(f'{output_folder}puncta_features.csv', index=False)

    # # save averages per biological replicate
    # rep_df = aggregate_features_by_group(features, ['condition', 'rep'], cols)
    # rep_df.to_csv(f'{output_folder}puncta_features_reps.csv', index=False)

    # save features normalized to cell intensity of channel of interest
    df_norm = features.copy()
    for col in cols:
        df_norm[col] /= df_norm['cell_coi1_intensity_mean']
    df_norm.to_csv(f'{output_folder}puncta_features_normalized.csv', index=False)

    # # save normalized averages per biological replicate
    # rep_norm_df = aggregate_features_by_group(df_norm, ['condition', 'tag', 'rep'], cols)
    # rep_norm_df.to_csv(f'{output_folder}puncta_features_normalized_reps.csv', index=False)

    logger.info('data wrangling and saving complete.')
    logger.info('pipeline complete.')