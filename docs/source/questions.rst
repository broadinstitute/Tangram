Frequently Asked Questions
==============================

**Do I need a GPU for running Tangram?**

A GPU is not required but is recommended. We run most of our mappings on a single P100 which maps ~50k cells in a few minutes. 

**How do I choose a list of training genes?**

A good way to start is to use the top 1k unique marker genes, stratified across cell types, as training genes. Alternatively, you can map using the whole transcriptome. Ideally, training genes should contain high quality signals: if most training genes are rich in dropouts or obtained with bad RNA probes your mapping will not be accurate.

**Do I need cell segmentation for mapping on Visium data?**

You do not need to segment cells in your histology for mapping on spatial transcriptomics data (including Visium and Slide-seq). You need, however, cell segmentation if you wish to deconvolve the data (_ie_ deterministically assign a single cell profile to each cell within a spatial voxel).

**I run out of memory when I map: what should I do?**

Reduce your spatial data in various parts and map each single part. If that is not sufficient, you will need to downsample your single cell data as well.