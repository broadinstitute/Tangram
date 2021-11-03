Tangram Under the Hood
===========================

Tangram instantiates a `Mapper` object passing the following arguments:
| _S_: single cell matrix with shape cell-by-gene. Note that genes is the number of training genes.
| _G_: spatial data matrix with shape voxels-by-genes. Voxel can contain multiple cells.

Then, Tangram searches for a mapping matrix *M*, with shape voxels-by-cells, where the element *M\_ij* signifies the probability of cell *i* of being in spot *j*. Tangram computes the matrix *M* by minimizing the following loss:

.. image:: _static/images/tangram_loss.gif
    :align: center
    :width: 600px
    :height: 100px
    
where cos_sim is the cosine similarity. The meaning of the loss function is that gene expression of the mapped single cells should be as similar as possible to the spatial data *G*, under the cosine similarity sense.

The above accounts for basic Tangram usage. In our manuscript, we modified the loss function in several ways so as to add various kinds of prior knowledge, such as number of cell contained in each voxels.