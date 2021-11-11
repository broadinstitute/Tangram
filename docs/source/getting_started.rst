Getting Started 
=====================

Installing Tangram
----------------------

To install Tangram, make sure you have `PyTorch <https://pytorch.org/>`_ and `scanpy <https://scanpy.readthedocs.io/en/stable/>`_ installed. If you need more details on the dependences, look at the `environment.yml <https://github.com/broadinstitute/Tangram/blob/master/environment.yml>`_ file.

Install Tangram from shell::

    pip install tangram-sc
    
Running Tangram 
--------------------------

Cell Level
**************************
To install Tangram, make sure you have `PyTorch <https://pytorch.org/>`_ and `scanpy <https://scanpy.readthedocs.io/en/stable/>`_ installed. If you need more details on the dependences, look at the `environment.yml <https://github.com/broadinstitute/Tangram/blob/master/environment.yml>`_ file. 

Create a conda environment for Tangram::

    conda env create --file environment.yml

Install tangram-sc from shell::

    conda activate tangram-env
    pip install tangram-sc
    
Import tangram::

    import tangram as tg
    
Then load your spatial data and your single cell data (which should be in `AnnData <https://anndata.readthedocs.io/en/latest/anndata.AnnData.html>`_ format), and pre-process them using **tg.pp_adatas**::

    ad_sp = sc.read_h5ad(path)
    ad_sc = sc.read_h5ad(path)
    tg.pp_adatas(ad_sc, ad_sp, genes=None)
    
The function **pp_adatas** finds the common genes between adata_sc, adata_sp, and saves them in two **adatas.uns** for mapping and analysis later. Also, it subsets the intersected genes to a set of training genes passed by **genes**. If **genes=None**, Tangram maps using all genes shared by the two datasets. Once the datasets are pre-processed we can map::

    ad_map = tg.map_cells_to_space(ad_sc, ad_sp)
    
The returned AnnData, **ad_map** , is a cell-by-voxel structure where **ad_map.X[i, j]** gives the probability for cell *i* to be in voxel *j*. This structure can be used to project gene expression from the single cell data to space, which is achieved via **tg.project_genes**::
    
    ad_ge = tg.project_genes(ad_map, ad_sc)
    
The returned **ad_ge** is a voxel-by-gene AnnData, similar to spatial data **ad_sp**, but where gene expression has been projected from the single cells. This allows to extend gene throughput, or correct for dropouts, if the single cells have higher quality (or more genes) than single cell data. It can also be used to transfer cell types onto space. 

For more details on how to use Tangram check out `our tutorial <https://github.com/broadinstitute/Tangram/blob/master/tangram_tutorial.ipynb>`_. 

.. raw:: html

    <a href="https://colab.research.google.com/drive/1SVLUIZR6Da6VUyvX_2RkgVxbPn8f62ge?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Tutorial"></a>

Cluster Level
***************************
To enable faster training and consume less memory, Tangram mapping can be done at cell cluster level.

Prepare the input data as the same you would do for cell level Tangram mapping. Then map using following code::
    
    ad_map = tg.map_cells_to_space(
                   ad_sc, 
                   ad_sp,         
                   mode='clusters',
                   cluster_label='subclass_label')
                   
Provided cluster_label must belong to ad_sc.obs. Above example code is to map at **subclass_label** level, and the **subclass_label** is in ad_sc.obs.

To project gene expression to space, use **tg.project_genes** and be sure to set the **cluster_label** argument to the same cluster label in mapping::

    ad_ge = tg.project_genes(
                  ad_map, 
                  ad_sc,
                  cluster_label='subclass_label')

