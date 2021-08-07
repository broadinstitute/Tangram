.. Tangram documentation master file, created by
   sphinx-quickstart on Thu Jun 24 20:01:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TANGRAM 
==========================
Tangram is a Python package, written in `PyTorch <https://pytorch.org/>`_ and based on `scanpy <https://scanpy.readthedocs.io/en/stable/>`_ , for mapping single-cell (or single-nucleus) gene expression data onto spatial gene expression data. The single-cell dataset and the spatial dataset should be collected from the same anatomical region/tissue type, ideally from a biological replicate, and need to share a set of genes. Tangram aligns the single-cell data in space by fitting gene expression on the shared genes. The best way to familiarize yourself with Tangram is to check out `our tutorial <https://github.com/broadinstitute/Tangram/blob/master/example/1_tutorial_tangram.ipynb>`_. 

.. image:: _static/images/tangram_overview.png
    :align: center

Manuscript
--------------




.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: General
   
   installation
   running_tangram 
   working 
   classes
   questions
   release_notes 
   contributors 
   cite
   news
   
.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials 

   tutorial_link
   tutorial_sq_link
