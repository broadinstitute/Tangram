# install CytoSPACE via https://github.com/PMBio/SpatialDE
import sys
sys.path.insert(1, '../../cytospace')

import cytospace
import numpy as np
import pandas as pd

def cell_sampling(adata_sc, adata_st):
    scRNA_data = pd.DataFrame(adata_sc.raw.X.toarray(),index=adata_sc.obs_names,columns=[g.lower() for g in adata_sc.var_names]).T
    cell_type_data = pd.DataFrame({"CellType" : adata_sc.obs["cell_subclass"].values}, index=adata_sc.obs_names)
    st_data = pd.DataFrame(adata_st.raw.X.toarray(),index=adata_st.obs_names,columns=[g.lower() for g in adata_st.var_names]).T
    
    scRNA_path = "../../cytospace_input/sc.csv"
    cell_type_path = "../../cytospace_input/ct.csv"
    st_path = "../../cytospace_input/st.csv"
    output_path = "../../cytospace_output/"
    output_prefix = ""
    
    st_data.to_csv(st_path)
    scRNA_data.to_csv(scRNA_path+".gz", compression="gzip")
    adata_sc.obs["cell_subclass"].to_csv(cell_type_path)

    mean_cell_numbers = 5
    scRNA_max_transcripts_per_cell = 1500
    sampling_method = "duplicates"
    seed = 1234

    cell_type_fraction_estimation_path = cytospace.estimate_cell_type_fractions(scRNA_path, cell_type_path, st_path, output_path, output_prefix)
    cell_type_fractions_data = pd.read_csv(cell_type_fraction_estimation_path, sep="\t")

    cell_number_to_node_assignment = cytospace.estimate_cell_number_RNA_reads(st_data, mean_cell_numbers)
    number_of_cells = np.sum(cell_number_to_node_assignment)
    cell_type_numbers_int = cytospace.get_cell_type_fraction(number_of_cells, cell_type_fractions_data)
    scRNA_data_sampled = scRNA_data
    scRNA_data_sampled = cytospace.downsample(scRNA_data_sampled, scRNA_max_transcripts_per_cell)
    scRNA_data_sampled = cytospace.sample_single_cells(scRNA_data_sampled, cell_type_data, cell_type_numbers_int, sampling_method, seed)
    
    adata_sc_cytospace_preprocessed = adata_sc[scRNA_data_sampled.columns,scRNA_data_sampled.index].copy()
    adata_sc_cytospace_preprocessed.X = scRNA_data_sampled.values.T
    return adata_sc_cytospace_preprocessed