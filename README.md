# Tangram

## Overview


## System requirements

Dependencies for running the algorithm and the example notebooks are listed in `environment.yml`.

## Installation guide

Tangram mapper can be instantiated as a class. Two different classes are used to do mapping without and with constraint (i.e., learned filter), respectively:
- `mapping.mapping_optimizer.Mapper` 
- `mapping.mapping_optimizer.MapperConstrained`

### Initialization

A `Mapper` instance is initialized with the following arguments:
- S (`ndarray`): Single nuclei matrix, shape = (number_cell, number_genes).
- G (`ndarray`): Spatial transcriptomics matrix, shape = (number_spots, number_genes). Spots can be single cells or they can contain multiple cells.
- d (`ndarray`): Spatial density of cells, shape = (number_spots,). This array should satisfy the constraints d.sum() == 1.
- Optional hyperparameters to weight the different terms in the loss function and to enable the weight regularizer.
- The device (`str` or `torch.device`).

In addition to the arguments passed to `Mapper`, a `MapperConstrained` is initialized with:
- The number of cells to be filtered.
- Optional hyperparameters to weight the different terms related to the learned filter in the loss function.

Please refer to the documentation in `mapping_optimizer.py` for details about initialization for the two mapping classes.

### Training

A `Mapper` or  `MapperConstrained` is optimized with the `train` method. The method takes as arguments:
- num_epochs (`int`): Number of epochs.
- learning_rate (`float`): Optional. Learning rate for the optimizer. Default is 0.1.
- print_each (`int`): Optional. Prints the loss each print_each epochs. If None, the loss is never printed. Default is 100.
And returns the optimized data structures.

For an unconstrained `Mapper`, the `train` method returns the optimized mapping matrix with shape = (number_cells, number_spots). For a `MapperConstrained`, also the optimized filter with shape = (number_cells,) is returned. 

### Transfer annotations

Giving the output of the mapping, we can transfer any annotation onto space. 

For uncontrained mapping, the `mapping.utils.transfer_annotations_prob(mapping_matrix, to_transfer)` function can be used. `mapping_matrix` with shape = (number_cells, number_spots) is the optimized mapping, `to_transfer` with shape = (number_cells, number_annotations) is the annotation matrix. 

For contrained mapping, the `mapping.utils.transfer_annotations_prob_filter(mapping_matrix, filter, to_transfer)` function can be used. In this case, the learned filter with shape = (number_cells,) is also passed. 

## Examples

Example Jupyter notebooks are available in the `example` subfolder. Currently, two examples are available:
- smFISH: contrained mapping, prediction cell type probabilities in space, deterministic assigmment of cell types.
- Visium: constrained mapping, prediction cell type probabilities in space, deconvolution.

The notebooks include detailed instructions and expected outputs. Optimization takes few minutes using a single P100 GPU.
