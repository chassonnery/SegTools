---
title: 'SegViz : A generic tool for 3D visualization and spatial clustering of mixed systems of spherical and rod-like objects'
tags:
  - Python
  - Paraview
  - mathematical modeling
  - biology
  - three-dimensional structure
authors:
  - name: Pauline Chassonnery
    orcid: 0009-0002-5751-1533
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Diane Peurichard
    orcid: 0000-0002-0807-2266
    affiliation: 1
  - name: Sinan Haliyo
    orcid: 0000-0003-4587-381X
    affiliation: 2
affiliations:
 - name: INRIA Paris, team Mamba, Sorbonne Université, CNRS, Université de Paris, Laboratoire Jacques-Louis Lions UMR7598, 75005 Paris, France
   index: 1
 - name: Institute for Intelligent Systems and Robotics, Sorbonne Université, CNRS UMR7222, 75005 Paris, France
   index: 2
date: 17 April 2024
bibliography: biblio.bib
---

# Statement of need

For a long time, the lack of high resolution tri-dimensional imaging techniques for biological tissues has hindered the development of 3D mathematical models for these tissues, as there was no possibility to compare the model results to biological data. The recent improvements in this field led to a renewed interest for 3D bio-mathematical models, whose validation against biological observations always begins with a visual comparison. However, the effective visualization of the data produced, e.g. through the imaging of biological samples or through numerical simulations of 3D agent-based models, remains a challenge. Visualization tools developed for biological images are generally inappropriate for mathematical data, requiring cumbersome conversion procedures and often resulting in lagging visualization. The reverse is also true. On the other hand, powerful visualization tools like the Visualization Toolkit (VTK) and its front-end application, Paraview [@ParaView], can bridge the gap between these two types of data. The VTK software platform is well-maintained, contains an expansive set of native functionalities and provides a robust foundation for scientific visualization. Yet, few efforts have been put in the development of plugins adapted to biological data and models.

As researchers in the field of mathematical biology, we created a 3D agent-based model of interacting rod-like fibers and spherical cells aiming to reproduce the emergence of the 3D architecture of connective tissues [@Cha2024; @ChaAT]. To visualize the structures produced by this model, we developed two Paraview macros (`SphereViz` and `RodViz`) enabling a dynamic, easy-to-handle 3D visualization of large sets of spherical and/or rod-like objects. Comparison with 3D images of biological samples is achieved using Paraview tiff-file reader.


To analyze the spatial structuring of the tissue produced by our model, we wanted to separate the cells of the system into different "clusters" based on (i) spatial proximity inside one cluster and (ii) the presence of fibers in-between two clusters.

To achieve this, we represented our system as a black-and-white image where cells appear in white and fibers in black (possible intersections between a cell and a fiber being also in black) and applied a watershed segmentation algorithm to divide the image into different regions. We then considered that all the cells located in the same region formed a cluster. We used the same watershed segmentation algorithm to identify cells clusters in tri-dimensional images of biological tissue samples based on the same two criteria. This enables a comparison between *in vivo* and *in silico* structures.

Watershed segmentation in 2 or 3D is a well-known method in the domain of image analysis, but is generally unknown to researchers outside this field. Moreover, the watershed procedure provided by image analysis softwares usually requires a number of preprocessing steps, including distance transform and filtering. Besides, data produced by a mathematical simulation will usually not even be in the form of images (as is the case for us) and will thus require an extra processing step that can be quite time-consuming if not optimized.

Hence, as both biologists and modelers may be interest by an integrated, automatic spatial-clustering algorithm, we propose our procedure in the format of a Python function named `WatershedSegmentation`. This function take as input either a 3D binary image or a list of spherical objects, and return a segmented version of the image (each identified region being colored with a unique hue) or list (each cell being attributed a number identifying the cluster it pertains to). It also includes an option for periodic boundary conditions, a very common hypothesis for mathematical models in finite spatial domain that has not equivalent in the field of image analysis and is thus never included in the related softwares.


# Overview


The SegViz tool can be split in two parts.

The Python function `WatershedSegmentation` takes as input either a 3D binary image displaying white objects over a black background (in the form of a tiff file or a Python numpy.array) or a dataset of spherical and possibly rod-like objects (in the form of a csv file or a Python pandas.DataFrame listing these objects and their properties). If the input is a binary image, the function divides it into different connected regions using the watershed segmentation algorithm, and save the result in a tiff file where each region is colored with a unique hue. If the input is a dataset, the function uses the same watershed segmentation algorithm to classify the spherical objects into spatially connected clusters, possibly separated by the rod-like objects, and save the result in a csv file where each spherical object is attributed a `ClusterIndex` property containing the unique index of the cluster it pertains to.

The Paraview macro `Rodviz` takes as input a csv file containing a list of rod-like objects and displays them as gray double-headed arrows. The Paraview macro `Sphereviz` takes as input a csv file containing a list of spherical objects and displays them as 3D spheres. Each glyph is colored according to the `ClusterIndex` property of the parent object if that information is available (that is, if the csv file does contain a column headed `ClusterIndex`) and white otherwise. Considering that Paraview's preset categorical color maps are either limited to 12 colors or contain colors that are not easy to distinguish, we provide a custom categorical color map based on the work of Sasha Trubetskoy [@Trubetskoy2017]. It contains $20$ colors listed in descending order of compatibility with color blindness. This color map can be loaded in any Paraview setup and will be used by the macro `Sphereviz` if it is present. If not, the default Paraview color-map `KAAMS` will be used.

The source code for SegViz and a complete user guide have been archived to Zenodo with the linked DOI [@SegViz].

# References