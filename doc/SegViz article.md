---
title: 'SegViz : A generic tool for 3D visualization and spatial clustering of mixed systems of spherical and rod-like objects'
tags:
  - Python
  - ParaView
  - mathematical modeling
  - biology
  - three-dimensional structure
date: 11 December 2025
authors:
  - name: Pauline Chassonnery
    orcid: 0009-0002-5751-1533
    affiliation: 1 # (Multiple affiliations must be quoted)
    corresponding: true # (This is how to denote the corresponding author)
  - name: Diane Peurichard
    orcid: 0000-0002-0807-2266
    affiliation: 1
  - name: Sinan Haliyo
    orcid: 0000-0003-4587-381X
    affiliation: 2
affiliations:
  - index: 1
    name: INRIA Paris, team Mamba, Sorbonne Université, CNRS, Université de Paris, Laboratoire Jacques-Louis Lions UMR7598, 75005 Paris, France
  - index: 2
    name: Institute for Intelligent Systems and Robotics, Sorbonne Université, CNRS UMR7222, 75005 Paris, France
bibliography: biblio.bib
---

# Statement of need

The objective of the SegViz tool is to provide non-specialists in image processing with an integrated, user-friendly tool for the 3D visualisation of systems composed of a mixture of spherical (e.g. cells, particles) and rod-like (e.g. fibers, bacteria) elements and for the automatic detection of spatially connected clusters of spherical objects separated by rod-like elements in such systems. It is aimed in particular at modellers, who may generate such an *in silico* dataset through numerical simulations, and at biologists who may retrieve it from *in vivo* or *in vitro* experiments (for example through 3D microscopy imaging of tissue samples).

Despite recent improvements in the field of high resolution tri-dimensional imaging techniques, the effective 3D visualization of the large datasets remains a challenge. This is particularly the case if a common plateform is to be used for both biological images and mathematical data. Robust scientific visualization tools like the Visualization Toolkit (VTK) and its front-end appli- cation, Paraview [@ParaView], can bridge the gap between these two types of data. The VTK software platform is well-maintained, contains an expansive set of native functionalities and provides a robust foundation for scientific visualization. Yet, few efforts have been put in the development of plugins adapted to biological data and models.

On the other hand, objects segmentation and clustering is a wide-spread problem in the domain of image analysis and many methods have been developed to solve it, of which one of the most widely used is the watershed transformation [@Meyer1994]. But these are generally not well-known to researchers outside this field and the procedures provided by image analysis software usually require a number of preprocessing steps. Moreover, data produced by a mathematical simulation will usually not even be in the form of images and will thus require an extra processing step that can be quite time-consuming if not optimized. Hence, both biologists and modelers may be interest by an integrated, automatic spatial-clustering algorithm.


As researchers in the field of mathematical biology, we created a 3D agent-based model of interacting rod-like fibers and spherical cells aiming to reproduce the emergence of the 3D architecture of connective tissues [@Cha2024; @ChaAT]. To visualize the structures produced by this model, we developed two ParaView macros (`SphereViz` and `RodViz`) enabling a dynamic, easy-to-handle 3D visualization of large sets of spherical and/or rod-like objects. Comparison with 3D images of biological samples is achieved using ParaView tiff-file reader.

To analyze the spatial structuring of the tissue produced by our model, we wanted to separate the cells of the system into different "clusters" based on (i) spatial proximity inside one cluster and (ii) the presence of fibers in-between two clusters. To achieve this, we represented our system as a black-and-white image where cells appear in white and fibers in black (possible intersections between a cell and a fiber being also in black) and applied a watershed segmentation algorithm to divide the image into different regions. We then considered that all the cells located in the same region formed a cluster. We used the same watershed segmentation algorithm to identify cells clusters in tri-dimensional images of biological tissue samples based on the same two criteria. This enables a comparison between *in vivo* and *in silico* structures.

The whole process is packaged in a Python function called `WatershedSegmentation`, which takes as input either a 3D binary image or a list of spherical objects, and return a segmented version of the image (each identified region being colored with a unique hue) or list (each cell being attributed a number identifying the cluster it pertains to). It also includes an option for periodic boundary conditions, a very common hypothesis for mathematical models in finite spatial domain that has not equivalent in the field of image analysis and is thus never included in the related softwares.


# Overview

The SegViz tool can be split in two parts.

The visualization part consists of two ParaView macros, `Sphereviz` and `Rodviz`. They take as input a .csv file describing the position, size and other optional properties of a set of spherical (resp. rod-like) objects and display them as 3D glyphs that can be colored according to any property referenced in the dataset. By default, the spherical objects are colored according to their `ClusterIndex` if that information is available (that is, if the csv file does contain a column headed `ClusterIndex`) and white otherwise. Considering that ParaView's preset categorical color maps are either limited to 12 colors or contain colors that are not easy to distinguish, we provide a custom categorical color map based on the work of Sasha Trubetskoy [@Trubetskoy2017]. It contains 20 colors listed in descending order of compatibility with color blindness. This color map can be loaded in any ParaView setup and will be used by the macro `Sphereviz` if it is present. If not, the default ParaView color-map `KAAMS` will be used.

The segmentation part consists in a Python function called `WatershedSegmentation`. It takes as input either a 3D binary image (in the form of a tiff file or a Python numpy.ndarray) or a list of spherical objects with their properties (in the form of a csv file or a Python pandas.DataFrame), as well as a number of fine-tuning parameters. In the first case, it returns a labeled image where each region has been attributed a unique label. In the second case, it groups the objects into clusters based on spatial proximity and returns a csv file with the original data plus a column containing the index of the cluster each object pertains to. In both cases, the segmentation step uses the watershed transformation [@Meyer1994].

The SegViz tool is free of use, upon proper citation of this article. The source code and a complete user guide have been archived to Zenodo with the linked DOI [@XXX].


# Fundings

This study has been partially supported through the grant EUR CARe no. ANR-18-EURE-0003 in the framework of the Programme des Investissements d’Avenir, by Sorbonne Alliance University with the Emergence project MATHREGEN under grant no. S29-05Z101 and by Agence Nationale de la Recherche (ANR) under the project grant no. ANR-22-CE45-0024-01.


# References