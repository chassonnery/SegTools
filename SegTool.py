"""
// -------------------------------------------------- //
//                                                    //
//             **WatershedSegmentation :**            //
//        Automation Script for the clustering        //
//        of spherical objects using watershed        //
//                                                    //
// -------------------------------------------------- //
// **Original algorithm :**                           //
//                                                    //
// **Script developers :**                            //
//   Pauline CHASSONNERY                              //
// -------------------------------------------------- //


## In case you use the results of this script in your article, please don't forget to cite us:
****************

## Purpose: *****************

## Copyrights (C) ***********

## License:
WatershedSegmentation is a free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. 
https://www.gnu.org/licenses/gpl-3.0.en.html 

## Commercial use:
The GPLv3 license cited above does not permit any commercial (profit-making or proprietary) use or re-licensing or re-distributions. Persons 
interested in for-profit use should contact the author. Note that the commercial use of this script is also protected by patent number: *******

"""
# Version: 2025-11-11

import numpy as np
import pandas as pd
import math as math
import scipy.ndimage as scim
from PIL import Image, ImageSequence
from scipy.sparse import issparse
from skimage.morphology import local_maxima, remove_small_objects
from skimage.segmentation import watershed
from warnings import warn



def WatershedSegmentation(datatype, InputSpheres, InputRods=None, xmax=None, ymax=None, zmax=None, PeriodicBoundaries=False, dil_coeff=1,
                          resolution=5, pixelsize=None, smooth_coeff=1, MinSize=1, savefile="data_seg", return_map=False,
                          header="ClusterIndex", header_per="_cluster"):
    """ Summary
        -------
        ***************** < Copy purpose here > ****************
        
        
        Parameters
        ----------
        datatype :
        
        InputSpheres : pandas.DataFrame or str
            Either a DataFrame containing data relative to/ describing a set of spherical objects or a string path leading to a csv file from which to
            retrieve the data.
            The DataFrame or file must have at least 2 rows and 4 columns and is assumed to be formatted as follow : 
                - one row per object,
                - three columns with label/header "X", "Y" and "Z" which contain the positional vector of the sphere's center,
                - one column with label/header "R" which contains the sphere's radius. 
            Extra/additional columns are accepted but will not be used.
        
        InputRods : pandas.DataFrame or str or None, optional
            Either a DataFrame containing data relative to/ describing a set of spherocylindrical (rod-like) objects or a string path leading to a csv
            file from which to retrieve the data or None if there is no rod-like object in the system.
            The DataFrame or file must have at least 1 rows and 4 columns and is assumed to be formatted as follow :
                - one row per object,
                - three columns with label/header "X", "Y" and "Z" which contain the positional vector of the spherocylinder's center,
                - three columns with label/header "wX", "wY" and "wZ" which contain its orientation vector,
                - one column with label/header "L" which contains its length (i.e. length of the central cylindrical part),
                - one column with label/header "R" which contains its radius.
            Extra/additional columns are accepted but will not be used.
            Default is None.
        
        PeriodicBoundaries : bool, optional 
            Specify if the domain boundaries are periodic or not. Default is False.
        
        xmax : float, optional
            Half-length of the computation domain in the x direction. User-provided value is only needed if ``PeriodicBoundaries`` is True, 
            otherwise default value is alright. Default is computed as the smallest value allowing to enclose all the spheres. See corresponding section 
            in the manual for more details.
            If ``PeriodicBoundaries`` is True but no value is provided for ``xmax``, computation will run with default value but a warning will be 
            issued to the user.
        
        ymax : float, optional
            Half-length of the computation domain in the y direction (identical to xmax).
        
        zmax : float, optional
            Half-length of the computation domain in the z direction (identical to xmax).
        
        dil_coeff : float, optional
            Dilatation coefficient to be applied to the spheres' radius to make clustering easier. See corresponding section in the manual for more
            details. Default value is 1 (i.e. no dilatation).
        
        resolution : int, optional
            Resolution of the image created for clustering, expressed as the number of pixels in the diameter of the smallest sphere. Default value is 10.
        
        smooth_coeff : float, optional
            Smoothing coefficient applied to the image (via a h-minima transformation) before watershed. See corresponding section in the manual for more 
            details. Default value is 3.
        
        MinSize : int, optional
            Minimal number of pixels (if datatype is image) or objects (if datatype is dataset) a cluster must contain to be considered as valid.
            Default value is 1 (i.e. all clusters are valid).
        
        header : str, optional
            Header to be given to the column containing the result of the clusterization process. Default value is "Cluster".
        
        
        References
        ----------
            The manual *****************
        
        
        Examples
        --------
        example 1
            import pandas as pd
            from WatershedSegmentation import ClusterizeSphereObjects
            
            Results = ClusterizeSphereObjects("demo_data_spheres.csv", InputRods="demo_data_rods.csv", dil_coeff=1.2)
            
            disp(Results)
            
            Results.to_csv("my_file.csv")
        
        example 2
            import pandas as pd
            from WatershedSegmentation import ClusterizeSphereObjects
            import matplotlib.pyplot as plt
            
            Results = ClusterizeSphereObjects("demo_data_spheres.csv", header="cluster (test 1)")
            Results = ClusterizeSphereObjects(Results, dil_coeff=1.2, header="cluster (test 2)")
            Results = ClusterizeSphereObjects(Results, InputRods="demo_data_rods.csv", dil_coeff=1.2, header="cluster (test 3)")
            Results = ClusterizeSphereObjects(Results, InputRods="demo_data_rods.csv", PeriodicBoundaries=True, xmax=20, ymax=10, zmax=10, \
                                              dil_coeff=1.2, header="cluster (test 4)")
            
            plt.figure(1)
            plt.subplot(1,4,1)
            plt.plot3()
            plt.title("Clusterization with default parameters")
        
            plt.subplot(1,4,2)
            plt.plot3()
            plt.title("Clusterization with dilation by coefficient 1.2")
        
            plt.subplot(1,4,3)
            plt.plot3()
            plt.title("Clusterization with dilation by coefficient 1.2 \n and using rod objects as separators")
        
            plt.subplot(1,4,4)
            plt.plot3()
            plt.title("Clusterization with periodic boundary conditions enabled")
            
            
            # ## Save watershed output as tiff file
            # norm = matplotlib.colors.Normalize(vmin=np.min(labels3), vmax=np.max(labels3), clip=True)
            # mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.nipy_spectral)
            # 
            # imstack = []
            # for z in range(labels3.shape[2]):
            #     RGBA_data = mapper.to_rgba(labels3[:,:,z])*255
            #     RGBA_data = RGBA_data.astype(np.uint8)
            #     imstack.append(Image.fromarray( RGBA_data[:,:,:3],mode='RGB' )) # drop last coordinate, that is transparency
            # 
            # imstack[0].save('Python_watershed_cv.tiff',save_all=True,append_images=imstack[1:])
    """

    # Retrieve the input data and check their validity
    if datatype == "image":
        SpheresSet = RetrieveImage(InputSpheres, "InputSpheres")
        if InputRods is not None:
            RodsSet = RetrieveImage(InputRods, "InputRods")
            if np.shape(SpheresSet) != np.shape(RodsSet):
                raise ValueError(f"Mismatched shape {np.shape(SpheresSet)} for InputSpheres and {np.shape(RodsSet)} for InputRods.")
        else:
            RodsSet = None
            
    elif datatype == "dataset":
        SpheresSet = RetrieveDataset(InputSpheres, "InputSpheres", mandatory_cols=["X", "Y", "Z", "R"], min_size=2)
        if InputRods is not None:
            RodsSet = RetrieveDataset(InputRods, "InputRods", mandatory_cols=["X", "Y", "Z", "wX", "wY", "wZ", "L", "R"], min_size=0)
        else:
            RodsSet = None
    
    else:
        raise TypeError("Argument datatype must be a string equal either to 'image' or 'dataset'")
        
    # Check validity of the output parameters
    assert isinstance(savefile, str)
    assert isinstance(return_map, bool)
    assert isinstance(header, str)
    assert isinstance(header_per, str)
    
    # Instanciate the class ParametersForClustering (check validity of user-defined parameters and compute internal parameters)
    params = ParametersForClustering(SpheresSet, RodsSet, PeriodicBoundaries, xmax, ymax, zmax, dil_coeff, resolution, smooth_coeff,\
                                     MinSize)
    
    # Clusterize data
    if datatype == "image":
        ClusterMap = ClusterizeBinaryImage(SpheresSet, RodsSet, pixelsize, smooth_coeff, MinSize)
        imstack = []
        for z in range(ClusterMap.shape[2]):
            imstack.append( Image.fromarray( ClusterMap[:,:,z] ))
        imstack[0].save(savefile+".tiff", save_all=True, append_images=imstack[1:])
        
        return ClusterMap
        
    elif datatype == "dataset":
        SpheresSet, ClusterMap = ClusterizeSphereObjects(SpheresSet, RodsSet, params, header, header_per)
        SpheresSet.to_csv(savefile+".csv", index=False)
        
        if return_map:
            imstack = []
            for z in range(ClusterMap.shape[2]):
                imstack.append( Image.fromarray( np.rot90( ClusterMap[:,:,z], axes=(0,1) )))
            imstack[0].save(savefile+".tiff", save_all=True, append_images=imstack[1:])
            return SpheresSet, ClusterMap
        else:
            return SpheresSet


    
def RetrieveImage(Input, Input_name):
    """ Retrieve the input data and check that it is formatted as a 3D binary image.
    """
    if isinstance(Input, np.ndarray):
        # If the user provided the data as a numpy.ndarray of dtype bool, make a copy of it
        image = (np.copy(Input)).astype("bool")
    elif isinstance(Input, str) and Input.endswith(".tiff"):
        # If the user provided a file name, read the data from this file
        image = np.array([ np.array(frame).astype("bool") for frame in ImageSequence.Iterator(Image.open(Input)) ])
    else:
        # Otherwise protest
        raise TypeError("For datatype='image', argument "+Input_name+" must be either a *numpy.ndarray* or a *string* path to a tiff file.")
#        raise TypeError("For datatype='image', argument "+Input_name+" must be either a 3D binary array (i.e. a 3D *numpy.ndarray* with dtype *bool*)"\
#                        +" or a *string* path to a tiff file containing a 3D binary image.")
    
    # Check the image dimension
    if image.ndim != 3:
        raise TypeError(f"For datatype='image', argument "+Input_name+" must be a 3D binary image. The number of dimensions of the data provided is {image.ndim}.")
        
    # Check the image type
    if image.dtype != 'b':
        raise TypeError(f"For datatype='image', argument "+Input_name+" must be a 3D binary image. The type of the data provided is {image.dtype}.")
    
    return image



def RetrieveDataset(Input, Input_name, mandatory_cols, min_size):
    """ Retrieve the input dataset and check that it is contains the mandatory columns and minimal number of objects.
    """
    assert isinstance(Input_name, str)
    assert all(isinstance(col, str) for col in mandatory_cols)
    assert isinstance(min_size, int)
    
    if isinstance(Input, pd.DataFrame):
        # If the user provided the data as a pandas.DataFrame, make a copy of it
        dataset = Input.copy(deep=True)
    elif isinstance(Input, str) and Input.endswith(".csv"):
        # If the user provided a file name, read the data from this file
        dataset = pd.read_csv(Input)
    else:
        # Otherwise protest
        raise TypeError("For datatype='dataset', argument "+Input_name+" must be either a *pandas.DataFrame* or a *string* path to a csv file.")

    # Check that the dataset contains the mandatory columns and that their values are scalar
    for col in mandatory_cols:
        if col not in dataset.columns:
            raise ValueError(f"Missing column '"+col+"' in dataset "+Input_name)
        elif ((dataset[col].dtypes.kind not in np.typecodes["AllFloat"]) and (dataset[col].dtypes.kind != 'i')):
            raise TypeError(f"Column '"+col+"' of dataset "+Input_name+" must contain scalar numbers")

    # Check that the dataset size (i.e. number of rows) is greater than the minimum size
    if (dataset.shape[0] < min_size):
        raise ValueError("For datatype='dataset', argument "+Input_name+" must be a dataset with at least "+str(min_size)+\
                         " rows (that is, it must describe at least "+str(min_size)+" objects).")
    
    return dataset



class ParametersForClustering:
    """ This class implement and check the validity/compatibility of all the parameters needed for the WatershedSegmentation algorithm. 
        
        
        Parameters
        ----------
        SpheresSet : pandas.DataFrame
            DataFrame containing data relative to/ describing a set of spherical objects. It must have at least 2 rows and 4 columns and is assumed to be 
            formatted as follow : 
                - one row per object,
                - three columns with label/header "X", "Y" and "Z" which contain the positional vector of the sphere's center,
                - one column with label/header "R" which contains the sphere's radius. 
            Extra/additional columns are accepted but will not be used.
        
        RodsSet : pandas.DataFrame or None, optional
            Either a DataFrame containing data relative to/ describing a set of spherocylindrical (rod-like) objects or None if there is no rod-like
            object in the system. The DataFrame must have at least 1 rows and 4 columns and is assumed to be formatted as follow :
                - one row per object,
                - three columns with label/header "X", "Y" and "Z" which contain the positional vector of the spherocylinder's center,
                - three columns with label/header "wX", "wY" and "wZ" which contain its orientation vector,
                - one column with label/header "L" which contains its length (i.e. length of the central cylindrical part),
                - one column with label/header "R" which contains its radius.
            Extra/additional columns are accepted but will not be used.
            Default is None.
        
        PeriodicBoundaries : bool, optional 
            Specify if the domain boundary conditions periodic or not. Default is False.
        
        xmax : float, optional
            Half-length of the computation domain in the x direction. User-provided value is only needed if ``PeriodicBoundaries`` is True,
            otherwise default value is alright. Default is computed as the smallest value allowing to enclose all the spheres. See corresponding section 
            in the manual for more details.
            If ``PeriodicBoundaries`` is True but no value is provided for ``xmax``, computation will run with default value but a warning will be 
            issued to the user.
        
        ymax : float, optional
            Half-length of the computation domain in the y direction (identical to xmax).
        
        zmax : float, optional
            Half-length of the computation domain in the z direction (identical to xmax).
        
        dil_coeff : float, optional
            Dilatation coefficient to be applied to the spheres' radius to make clustering easier. See corresponding section in the manual for more 
            details. Default value is 1 (i.e. no dilatation).
        
        resolution : int, optional
            Resolution of the image created for clustering, expressed as the number of pixels in the diameter of the smallest sphere. Default value is 10.
        
        smooth_coeff : float, optional
            Smoothing coefficient applied to the image (via a h-minima transformation) before watershed. See corresponding section in the manual for more 
            details. Default value is 3.
        
        MinSize : int, optional
            Minimal number of objects a cluster must contain to be considered as valid. Default value is 5.
        
        header : str, optional
            Header to be given to the column containing the result of the clusterization process. Default value is "ClusterIndex".
        
        header_per : str, optional
            Header suffix to be given to the columns containing the coordinates of each spherical object with respect to the filtered cluster list (see
            __MergePeriodicCluster). Only used if ``PeriodicBoundaries'' is True. Default value is "_cluster" (leading to "X_Cluster", "Y_cluster",
            "Z_cluster").
        
        
        Attributes
        ----------
        Nspheres : int
            Number of spherical objects in the system.
        Nrods : int
            Number of rod-like objects in the system.
        PeriodicBoundaries : bool
            Specify if the domain boundary conditions periodic or not.
        xmax, ymax, zmax : float
            Half-length of the computation domain in the x, y and z direction respectively.
        dil_coeff : float
            Dilatation coefficient to be applied to the spheres' radius to make clustering easier.
        resolution : int
            Resolution of the image created for clustering, expressed as the number of pixels in the diameter of the smallest sphere.    
        pixelsize : float
            Size of a cubic pixel, equal to the diameter of the smallest sphere divided by ``resolution``.
        xgrid : numpy.ndarray (ndim = 3)
            3D array containing a the x-grid part of a meshgrid.
            If ``PeriodicBoundaries`` is False, the grid span over domain [-``xmax``,``xmax``] with a uniform step size equal to ``pixelsize``
            If ``PeriodicBoundaries`` is True, the grid span over domain [-2``xmax``,2``xmax``] with a uniform step size equal to ``pixelsize``
            In case the value of ``pixelsize`` does not allow for a whole number of points in the domain described above, this domain will be slightly 
            extended on the right-hand side. 
        ygrid : numpy.ndarray (ndim = 3)
            3D array containing a the y-grid part of a meshgrid (same as xgrid).
        zgrid : numpy.ndarray (ndim = 3)
            3D array containing a the z-grid part of a meshgrid (same as xgrid).
        Nx : int
            Number of pixels of the image created for clustering in the x direction.
            If ``PeriodicBoundaries`` is False then (``Nx`` - 1) x ``pixelsize`` ≥ ``xmax``.
            If ``PeriodicBoundaries`` is True then (``Nx`` - 1) x ``pixelsize`` ≥ 2 x ``xmax``.
        Ny : int
            Number of pixels of the image created for clustering in the y direction (same as Nx).
        Nz : int
            Number of pixels of the image created for clustering in the z direction (same as Nx).
        smooth_coeff : float
            Smoothing coefficient applied to the image (via a h-minima transformation) before watershed.
        MinSize : int
            Minimal number of objects a cluster must contain to be considered as valid.   
        header : str
            Header to be given to the column containing the result of the clusterization process.
        header_per : str
            Header suffix to be given to the columns containing the coordinates of each spherical object with respect to the filtered cluster list (see
            __MergePeriodicCluster). Only used if ``PeriodicBoundaries'' is True.
    """
    
    def __init__(self, SpheresSet, RodsSet=None, PeriodicBoundaries=False, xmax=None, ymax=None, zmax=None, dil_coeff=1, resolution=5, \
                 smooth_coeff=1, MinSize=1):
        """ Constructor.
        """
        # Retrieve number of spherical objects
        self.Nspheres = len(SpheresSet["X"])
        
        # Retrieve number of rod-like objects
        if RodsSet is None:
            self.Nrods = 0
        else:
            self.Nrods = len(RodsSet["X"])
        
        # Copy value of parameter 'PeriodicBoundaries' into the equivalent attribute
        if isinstance(PeriodicBoundaries,bool):
            self.PeriodicBoundaries = PeriodicBoundaries
        else:
            raise TypeError("PeriodicBoundaries must be of type logical (i.e. either True or False).")
        
        
        # If the user provided no value for xmax and/or ymax and/or zmax, compute them as the half-length of the smallest cuboid box centered on the
        # origin and enclosing all the spheres of the set.
        # If the user asked for periodic boundary condition but did not provide value for xmax and/or ymax and/or zmax, keep running but issue a 
        # warning.
        warning_message = "For periodic boundary conditions, auto-estimation of the domain size is not a reliable option and may lead to invalid"\
                          +" results. Please provide value for the half-length of the computation domain in the x, y and z directions."
        warning_trigger = False
        
        if xmax is None:
            self.xmax = np.max(np.abs(SpheresSet["X"]) + SpheresSet["R"]) 
            if self.PeriodicBoundaries == True:
                warning_trigger = True
        elif (isinstance(xmax, (int,float)) and (xmax > 0)):
            self.xmax = xmax
        else:
            raise TypeError("xmax must be a positive scalar number")
        
        if ymax is None:
            self.ymax = np.max(np.abs(SpheresSet["Y"]) + SpheresSet["R"]) 
            if self.PeriodicBoundaries == True:
                warning_trigger = True
        elif (isinstance(ymax, (int,float)) and (ymax > 0)):
            self.ymax = ymax
        else:
            raise TypeError("ymax must be a positive scalar number")
        
        if zmax is None:
            self.zmax = np.max(np.abs(SpheresSet["Z"]) + SpheresSet["R"]) 
            if self.PeriodicBoundaries == True:
                warning_trigger = True
        elif (isinstance(zmax, (int,float)) and (zmax > 0)):
            self.zmax = zmax
        else:
            raise TypeError("zmax must be a positive scalar number")
            
        if warning_trigger:
            warn(warning_message) 
        
        
        # Copy value of parameter 'dil_coeff' into the equivalent attribute
        if (isinstance(dil_coeff, (int,float)) and (dil_coeff > 0)):
            self.dil_coeff = dil_coeff
        else:
            raise TypeError("dif_coeff must be a positive scalar number")
        
        # Copy value of parameter 'resolution' into the equivalent attribute
        if (isinstance(resolution, int) and (resolution > 0)):
            self.resolution = resolution
        else:
            raise TypeError("resolution must be a positive integer")
        
        # Copy value of parameter 'smooth_coeff' into the equivalent attribute
        if (isinstance(smooth_coeff, (int,float)) and (smooth_coeff >= 0)):
            self.smooth_coeff = smooth_coeff
        else:
            raise TypeError("smooth_coeff must be a non-negative scalar number")
        
        # Copy value of parameter 'MinSize' into the equivalent attribute
        if (isinstance(MinSize, int) and (MinSize >= 0)):
            self.MinSize = MinSize
        else:
            raise TypeError("MinSize must be a non-negative integer")
        
        # Copy value of parameter 'header' into the equivalent attribute
        if isinstance(header,str):
            self.header = header
        else:
            raise TypeError("header must be a string.")
            
        if isinstance(header_per,str):
            self.header_per = header_per
        else:
            raise TypeError("header_per must be a string.")
        
        
        # Compute value of 'pixelsize' as the diameter of the smallest sphere divided by 'resolution'.
        self.pixelsize = min(SpheresSet["R"])/self.resolution
        
        
        # Create 3D grid spaning over domain [-2'xmax',2'xmax']x[-2'ymax',2'ymax']x[-2'zmax',2'zmax'] with a uniform step size equal to 'pixelsize'
        if self.PeriodicBoundaries:
            [self.xgrid, self.ygrid, self.zgrid] = np.meshgrid(np.arange(-2*self.xmax, 2*self.xmax+self.pixelsize, self.pixelsize), \
                                                               np.arange(-2*self.ymax, 2*self.ymax+self.pixelsize, self.pixelsize), \
                                                               np.arange(-2*self.zmax, 2*self.zmax+self.pixelsize, self.pixelsize), indexing='ij')
        # Create 3D grid spaning over domain [-'xmax','xmax']x[-'ymax','ymax']x[-'zmax','zmax'] with a uniform step size equal to 'pixelsize'
        else:
            [self.xgrid, self.ygrid, self.zgrid] = np.meshgrid(np.arange(-self.xmax, self.xmax+self.pixelsize, self.pixelsize), \
                                                               np.arange(-self.ymax, self.ymax+self.pixelsize, self.pixelsize), \
                                                               np.arange(-self.zmax, self.zmax+self.pixelsize, self.pixelsize), indexing='ij')
        
        # Compute number of pixels in the grid
        [self.Nx, self.Ny, self.Nz] = self.xgrid.shape



def ClusterizeBinaryImage(SpheresSet, RodsSet, params):
    """
    """
    if RodsSet is None:
        bw = SpheresSet
    else:
        # Substract the image RodsSet from SpheresSet
        bw = SpheresSet & (~RodsSet)
    
    # Remove isolated spots/objects too small to constitute a cluster
    bw = remove_small_objects(bw, min_size=params.MinSize, connectivity=3)
    
    # Apply watershed algorithm to this image (with standard preprocess)
    ClusterMap = MapRegionsUsingWatershed(bw, params.pixelsize, params.smooth_coeff)
    
    return ClusterMap



def ClusterizeSphereObjects(SpheresSet, RodsSet, params, header, header_per):
    """
    """
    # Create a 3D binary image with 1's for spheres and 0's for background
    bw = Create3Dbinaryimage(SpheresSet, RodsSet, params)
    
    # Apply watershed algorithm to this image (with standard preprocess)
    ClusterMap = MapRegionsUsingWatershed(bw, params.pixelsize, params.smooth_coeff)
    
    # Identify the cluster index attributed to each element of 'SpheresSet' according to the cluster-map obtained by watershed
    # and, if parameter 'PeriodicBoundaries'=True, the coordinates of each element with respect to the filtered cluster
    # list (see __MergePeriodicCluster)
    ClusterIndex, SpheresSetTranslatedPosition = ClusterizeFromMap(SpheresSet, ClusterMap, params)
    
    # Add to 'SpheresSet' a column containing the cluster index of each element
    SpheresSet[header] = pd.Series(ClusterIndex)
    
    # Add to 'SpheresSet' three columns containing the coordinates of each element with respect to the filtered cluster list
    # (see __MergePeriodicCluster)
    if params.PeriodicBoundaries==True:
        SpheresSet["X"+header_per] = pd.Series(SpheresSetTranslatedPosition[:,0])
        SpheresSet["Y"+header_per] = pd.Series(SpheresSetTranslatedPosition[:,1])
        SpheresSet["Z"+header_per] = pd.Series(SpheresSetTranslatedPosition[:,2])
    
    return SpheresSet, ClusterMap



def Create3Dbinaryimage(SpheresSet, RodsSet, params):
    """ Return a binary image with 1's for spheres and 0's for background.
        
        The radius of the spherical objects is multiplied by parameter ``dil_coeff'' to help with the upcoming segmentation. If rod-like objects are provided,
        pixels located within a rod are set to 0 (even if overlapping with a sphere).
    """
    # Create void binary image with the right size
    bw_tot = params.xgrid*0.0 <= -1
    
    # Insert into this image all the elements of SpheresSet
    if params.PeriodicBoundaries:
        for index in range(params.Nspheres):
            # Extract the coordinates of the 'index'-th object
            x, y, z = SpheresSet["X"][index], SpheresSet["Y"][index], SpheresSet["Z"][index]
            # Multiply its radius by dil_coeff to help with the upcoming segmentation
            r = SpheresSet["R"][index]*params.dil_coeff
            # Insert the object and its virtual duplicates into the image
            for per in range(8):
                # Compute position of the 'per'-th duplicate
                [xper, yper, zper] = __FindVirtualDuplicate(x, y, z, per, params)
                # Retrieve binary mask corresponding to this duplicate
                bw_object, [Nx_min,Nx_max,Ny_min,Ny_max,Nz_min,Nz_max] = __SphereMask(xper, yper, zper, r, params)
                # Insert this mask into the image 'bw'
                bw_tot[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] = bw_tot[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] | bw_object
    else:
        for index in range(params.Nspheres):
            # Extract the object coordinates
            x, y, z = SpheresSet["X"][index], SpheresSet["Y"][index], SpheresSet["Z"][index]
            # Multiply its radius by dil_coeff to help with the upcoming segmentation
            r = SpheresSet["R"][index]*params.dil_coeff
            # Retrieve the binary mask corresponding this object
            bw_object, [Nx_min,Nx_max,Ny_min,Ny_max,Nz_min,Nz_max] = __SphereMask(x, y, z, r, params)
            # Insert this mask into the image 'bw'
            bw_tot[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] = bw_tot[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] | bw_object
            
            
    # Fill closed holes between objects
    bw_tot = scim.binary_fill_holes(bw_tot)
    
    
    # Substract all the elements of RodsSet from the image
    if RodsSet is not None:
        if params.PeriodicBoundaries:
            for index in range(params.Nrods):
                # Extract the coordinates of the 'index'-th object
                x, y, z = RodsSet["X"][index], RodsSet["Y"][index], RodsSet["Z"][index]
                wx, wy, wz = RodsSet["wX"][index], RodsSet["wY"][index], RodsSet["wZ"][index]
                l, r = RodsSet["L"][index], RodsSet["R"][index]
                # Substract the object and its duplicates from the image
                for per in range(8):
                    # Compute position of the 'per'-th duplicate
                    [xper,yper,zper] = __FindVirtualDuplicate(x, y, z, per, params)
                    # Retrieve the binary mask corresponding to this duplicate
                    bw_object, [Nx_min,Nx_max,Ny_min,Ny_max,Nz_min,Nz_max] = __RodMask(xper, yper, zper, wx, wy, wz, l, r, params)
                    # Substract this mask from the image 'bw'
                    bw[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] = bw[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] & (~bw_object)
        else:
            # Substract each object from the image
            for index in range(params.Nrods):
                # Extract the coordinates of the 'index'-th object
                x, y, z = RodsSet["X"][index], RodsSet["Y"][index], RodsSet["Z"][index]
                wx, wy, wz = RodsSet["wX"][index], RodsSet["wY"][index], RodsSet["wZ"][index]
                l, r = RodsSet["L"][index], RodsSet["R"][index]
                # Retrieve the binary mask corresponding this object
                bw_object, [Nx_min,Nx_max,Ny_min,Ny_max,Nz_min,Nz_max] = __RodMask(x, y, z, wx, wy, wz, l, r, params)
                # Substract this mask from the image 'bw'
                bw_tot[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] = bw_tot[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max] & (~bw_object)
        
        bw_tot = scim.binary_fill_holes(bw_tot)
    
    
    return bw_tot



def __SphereMask(x, y, z, r, params):
    """ Return a small binary image of a sphere of center (``x``,``y``,``z``) and radius ``r``, as well as the coordinates
        [Nx_min, Nx_max, Ny_min, Ny_max, Nz_min, Nz_max] of this image in the whole picture.
    """
    # Compute the smallest box [Nx_min,Nx_max[ x [Ny_min,Ny_max[ x [Nz_min,Nz_max[ entirely containing this sphere. To help with the segmentation to
    # come, the radius of the sphere is dilated by dil_coeff. Note the left-hand inclusion and right-hand exclusion to accommodate Python indexing.
    # Grid's index are between 0 and Nx-1 (both included), so the range is cut-off to [0,Nx[.
    Nx_min = max(math.floor( (x - r + (int(params.PeriodicBoundaries) + 1)*params.xmax)/params.pixelsize )    , 0)
    Nx_max = min(math.floor( (x + r + (int(params.PeriodicBoundaries) + 1)*params.xmax)/params.pixelsize ) + 2, params.Nx)
    
    Ny_min = max(math.floor( (y - r + (int(params.PeriodicBoundaries) + 1)*params.ymax)/params.pixelsize )    , 0)
    Ny_max = min(math.floor( (y + r + (int(params.PeriodicBoundaries) + 1)*params.ymax)/params.pixelsize ) + 2, params.Ny)
    
    Nz_min = max(math.floor( (z - r + (int(params.PeriodicBoundaries) + 1)*params.zmax)/params.pixelsize )    , 0)
    Nz_max = min(math.floor( (z + r + (int(params.PeriodicBoundaries) + 1)*params.zmax)/params.pixelsize ) + 2, params.Nz)
    
    # Create a small binary image with only the considered sphere
    bw_object = np.sqrt( (params.xgrid[Nx_min:Nx_max, Ny_min:Ny_max, Nz_min:Nz_max] - x)**2 +\
                         (params.ygrid[Nx_min:Nx_max, Ny_min:Ny_max, Nz_min:Nz_max] - y)**2 +\
                         (params.zgrid[Nx_min:Nx_max, Ny_min:Ny_max, Nz_min:Nz_max] - z)**2 ) <= r*params.dil_coeff
                         
    return bw_object, [Nx_min, Nx_max, Ny_min, Ny_max, Nz_min, Nz_max]



def __RodMask(x, y, z, wx, wy, wz, l, r, params):
    """ Return a small binary image of a rod (i.e. spherocylinder) of center (``x``,``y``,``z``), orientation vector (wx,wy,w), length ``l`` and radius
        ``r``, as well as the coordinates [Nx_min, Nx_max, Ny_min, Ny_max, Nz_min, Nz_max] of this image in the whole picture.
    """
    # Compute the smallest box [Nx_min,Nx_max[ x [Ny_min,Ny_max[ x [Nz_min,Nz_max[ entirely containing this spherocylinder. Note the left-hand
    # inclusion and right-hand exclusion to accommodate Python indexing. Grid's index are between 0 and Nx-1 (both included), so the range is cut-off
    # to [0,Nx[.
    Nx_min = max(math.floor( (x - abs(wx)*l/2.0 - r + (int(params.PeriodicBoundaries) + 1)*params.xmax)/params.pixelsize )    , 0)
    Nx_max = min(math.floor( (x + abs(wx)*l/2.0 + r + (int(params.PeriodicBoundaries) + 1)*params.xmax)/params.pixelsize ) + 2, params.Nx)
    
    Ny_min = max(math.floor( (y - abs(wy)*l/2.0 - r + (int(params.PeriodicBoundaries) + 1)*params.ymax)/params.pixelsize )    , 0)
    Ny_max = min(math.floor( (y + abs(wy)*l/2.0 + r + (int(params.PeriodicBoundaries) + 1)*params.ymax)/params.pixelsize ) + 2, params.Ny)
    
    Nz_min = max(math.floor( (z - abs(wz)*l/2.0 - r + (int(params.PeriodicBoundaries) + 1)*params.zmax)/params.pixelsize )    , 0)
    Nz_max = min(math.floor( (z + abs(wz)*l/2.0 + r + (int(params.PeriodicBoundaries) + 1)*params.zmax)/params.pixelsize ) + 2, params.Nz)
    
    # Compute the orthogonal projection of each point of the reduced grid onto the central segment of the spherocylinder
    # For each point of the reduced grid, compute its orthogonal projection onto the central segment of the spherocylinder
    p = (params.xgrid[Nx_min:Nx_max, Ny_min:Ny_max, Nz_min:Nz_max] - x)*wx +\
        (params.ygrid[Nx_min:Nx_max, Ny_min:Ny_max, Nz_min:Nz_max] - y)*wy +\
        (params.zgrid[Nx_min:Nx_max, Ny_min:Ny_max, Nz_min:Nz_max] - z)*wz
    
    # For each point of the reduced grid, compute the distance between its projection and itself
    d = np.sqrt( (params.xgrid[Nx_min:Nx_max, Ny_min:Ny_max, Nz_min:Nz_max] - x)**2 +\
                 (params.ygrid[Nx_min:Nx_max, Ny_min:Ny_max, Nz_min:Nz_max] - y)**2 +\
                 (params.zgrid[Nx_min:Nx_max, Ny_min:Ny_max, Nz_min:Nz_max] - z)**2 - p**2 )
    
    # Create a small binary image with the central cylinder
    bw_object1 = (np.abs(p) <= l/2.0) & (d <= r)
    # Create a small binary image with the two end-point spheres
    bw_object2 = np.sqrt(d**2 + (np.abs(p) - l/2.0)**2) <= r
    # Sum up the two previous images to obtain a spherocylinder
    bw_object = bw_object1 | bw_object2
                         
    return bw_object, [Nx_min, Nx_max, Ny_min, Ny_max, Nz_min, Nz_max]



def __FindVirtualDuplicate(xin, yin, zin, per, params):
    """ Return position of the center of the ``per``-th duplicate of an object located at (``xin``,``yin``,``zin``).
        per = 0 : real position
        per = 1 : transposition through the closest face in the x direction
        per = 2 : transposition through the closest face in the y direction
        per = 3 : transposition through the closest face in the z direction
        per = 4 : transposition through the closest edge in the xy direction
        per = 5 : transposition through the closest edge in the xz direction
        per = 6 : transposition through the closest edge in the yz direction
        per = 7 : transposition through the closest vertex
    """
    # Tranlation through the closest face in the x direction
    xper = xin - 2*math.copysign(params.xmax, xin)
    # Tranlated position of the object through the closest y-edge
    yper = yin - 2*math.copysign(params.ymax, yin)
    # Tranlated position of the object through the closest z-edge
    zper = zin - 2*math.copysign(params.zmax, zin)
    
    if per==0:
        xout = xin  
        yout = yin
        zout = zin
    elif per==1:
        xout = xper
        yout = yin
        zout = zin
    elif per==2:
        xout = xin
        yout = yper
        zout = zin
    elif per==3:
        xout = xin
        yout = yin
        zout = zper
    elif per==4:
        xout = xper
        yout = yper
        zout = zin
    elif per==5:
        xout = xper
        yout = yin
        zout = zper
    elif per==6:
        xout = xin
        yout = yper
        zout = zper
    elif per==7:
        xout = xper
        yout = yper
        zout = zper
    
    return xout, yout, zout
    
    
    
def MapRegionsUsingWatershed(bw, pixelsize, smooth_coeff):
    """ Return a labeled map of the different regions in the binary image ``bw``, i.e. a numpy.ndarray of the same shape than ``bw`` with 0's for 
        background and different strictly positive integers for each region. Each region corresponds to a cluster of spheres. 
        
        To identify the regions, this function first process the binary image ``bw`` into a grayscale image/ topographic map using the procedure 
        described in
        ****** matlab page ******************
        
        then apply a watershed algorithm to this grayscale image using its local peaks as regions' seeds.
    """
    # For each pixel of the input binary image, compute the Euclidean distance to the nearest zero pixel, using ``pixelsize`` as the length of the 
    # pixels in each direction.
    distance_map = scim.distance_transform_edt(bw, sampling=pixelsize)
    
    # Apply h-minimum transform to smooth the distance map.
    # --- This function was designed to mimic the behavior of Matlab's imhmin function on 3D images.
    distance_map_smoothed = - imhmin(distance_map, smooth_coeff*pixelsize) + 1.0
    
    # Find the local peaks in the smoothed distance map and return a 3D binary image with 1's at the position of local peaks and 0's elsewhere. 
    local_max_map = local_maxima(distance_map_smoothed)
    
    # Label the different peaks using connected component analysis with full connectivity in the three directions. The labeled pixels will serve as
    # seeds for the watershed process.
    seed_for_watershed = scim.label(local_max_map, np.ones((3,3,3)))[0]
    
    # Apply watershed algorithm.
    ClusterMap = watershed(-distance_map_smoothed, seed_for_watershed, mask=bw)
    
    return ClusterMap



## Copyright (C) 2017 Hartmut Gimpel <hg_code@gmx.de>
def imhmin(im, h, conn=None):
    """ @deftypefn  {Function File} {} @ imhmin (@var{im}, @var{h})
        @deftypefnx {Function File} {} @ imhmin (@var{im}, @var{h}, @var{conn})
        Caculate the morphological h-minimum transform of an image @var{im}.
    
        This function removes all regional minima in the grayscale image @var{im} whose depth is less or equal to the given threshold level @var{h}, and
        it increases the depth of the remaining regional minima by the value of @var{h}. (A "regional minimum" is defined as a connected component of
        pixels with an equal pixel value that is less than the value of all its neighboring pixels. And the "depth" of a regional minimum can be thought
        of as minimum pixel value difference between the regional minimum and its neighboring maxima.)
    
        The input image @var{im} needs to be a real and nonsparse numeric array (of any dimension), and the height parameter @var{h} a non-negative
        scalar number.
    
        The definition of "neighborhood" for this morphological operation can be set with the connectivity parameter @var{conn}, which defaults to 8 for
        2D images, to 26 for 3D  images and to @code{conn(ndims(n), "maximal")} in general. @var{conn} can be given as scalar value or as a boolean matrix
        (see @code{conndef} for details).
    
        The output is a transformed grayscale image of same type and shape as the input image @var{im}.
    
        @seealso{imhmax, imregionalmin, imextendedmin, imreconstruct}
        @end deftypefn
        
        Algorithm:
        * The 'classical' reference for this morphological h-minimum function is the book "Morphological Image Analysis" by P. Soille (Springer, 2nd
          edition, 2004), chapter 6.3.4 "Extended and h-extrema".
          It says: "This [h-maximum] is achieved by performing the reconstruction by dilation of [a grayscale image] f from f-h:
                        HMAX_h(f) = R^delta_f (f - h)
                    [...]
                    The h-minima [...] transformations are defined by analogy:
                        HMIN_h(f) = R^epsilon_f (f + h)".
        * A more easily accessible reference is for example the following web page by Régis Clouard:
              https://clouard.users.greyc.fr/Pantheon/experiments/morphology/index-en.html#extremum
          It says: "It is defined as the [morphological] reconstruction by erosion of [a grayscale image] f increased by a height h."
          (We will call the grayscale image im instead of f.)
    """
    
    # Retrieve input parameters, set default value
    if conn==None:
        conn = np.ones((3,)*im.ndim)
    else:
        if not isinstance(conn,np.ndarray):
            raise ValueError("Connectivity must be of type numpy.ndarray")
        else:
            dim = conn.ndim
            valid = True
            for i in range(dim):
                if (conn.shape[i]!=3):
                    valid = False
            if not valid:
                raise TypeError("Connectivity must be an array with all dimensions of size 3")
            
            if (conn.dtype.kind not in np.typecodes["AllInteger"]):
                raise ValueError("Connectivity must be an array of integers")
            elif ( (conn[(1,)*dim] != 1) or ( (np.unique(conn)!=[0, 1]) and (np.unique(conn)!=[1]) )):
                raise ValueError("Connectivity must be an array with only 0 or 1 as values, and 1 at its center")
    
      
    # Check input parameters
    if ( (not isinstance(im,np.ndarray)) or (im.dtype.kind not in np.typecodes["AllFloat"]) or issparse(im) ):
        raise TypeError("imhmin: IM must be a real and nonsparse numeric array")
        
    if ((type(h)!=int) and (type(h)!=float)):
        raise TypeError("imhmin: H must be a non-negative scalar number")
    elif (h<0):
        raise ValueError("imhmin: H must be non-negative")
      
    # Do the actual calculation
    # (Calculate dilations of the inverse image, instead of erosions of the original image, because this is what imreconstruct can do.)
    im2 = imreconstruct((im-h), im, conn)
    im2 = 1 - im2
    
    return im2



def imreconstruct(marker, mask, conn):
    enter = True
    while (enter or (not np.all(marker == previous))):
        enter = False
        previous = marker
        marker = scim.maximum_filter(marker,footprint=conn)
        if (marker.dtype.kind == 'bool'):
            marker = (marker and mask)
        else:
            marker = np.minimum(marker,mask)
    return marker



def ClusterizeFromMap(SpheresSet, ClusterMap, params):
    ## """Add to the original ``SpheresSet`` DataFrame a column labeled ``header`` and containing the cluster index attributed to each sphere
    ##    according to the ``ClusterMap``, with clusters continuously numbered starting from 0 and non-attributed objects denoted by -1.
    ## """
    """ Return an array referencing the cluster index attributed to each element of ``SpheresSet`` according to the ``ClusterMap``, with clusters 
        continuously numbered starting from 0 and non-attributed objects denoted by -1.
    """
    if params.PeriodicBoundaries:
        # Create array in which to store the cluster index of each object and its duplicates
        DuplicatedClusterIndex = np.zeros((params.Nspheres,8), dtype=int)
                                          
        # Retrieve cluster index for each object and its duplicates
        for index in range(params.Nspheres):
            for per in range(8):
                # Compute position of the 'per'-th duplicate of object 'index'
                [xper,yper,zper] = __FindVirtualDuplicate(SpheresSet["X"][index], SpheresSet["Y"][index], SpheresSet["Z"][index], per, params)
                # Retrieve object mask
                object_mask, [Nx_min,Nx_max,Ny_min,Ny_max,Nz_min,Nz_max] = __SphereMask(xper, yper, zper, SpheresSet["R"][index], params)
                # Using this mask, retrieve watershed information over the object
                object_seg = ClusterMap[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max]
                object_seg = object_seg[object_mask]
                # Find the most common value attributed to the pixels of the considered object by the watershed segmentation
                DuplicatedClusterIndex[index,per] = np.argmax(np.bincount(object_seg))
        
        # Merge the various instances of the clusters extending through the border of the domain
        ClusterIndex, SpheresSetTranslatedPosition = __MergePeriodicCluster(SpheresSet, DuplicatedClusterIndex, params)
                                                
    else:
        # Create array in which to store the cluster index of each object
        ClusterIndex = np.zeros(params.Nspheres, dtype=int)
    
        # Retrieve cluster index for each object
        for index in range(params.Nspheres):
            # Retrieve object mask
            object_mask, [Nx_min,Nx_max,Ny_min,Ny_max,Nz_min,Nz_max] = __SphereMask(SpheresSet["X"][index], SpheresSet["Y"][index],\
                                                                                    SpheresSet["Z"][index], SpheresSet["R"][index], params)
            # Using this mask, retrieve watershed information over the object
            object_seg = ClusterMap[Nx_min:Nx_max,Ny_min:Ny_max,Nz_min:Nz_max]
            object_seg = object_seg[object_mask]
            # Find the most common value attributed to the pixels of the object by the watershed segmentation
            ClusterIndex[index] = np.argmax(np.bincount(object_seg))
        
        SpheresSetTranslatedPosition = None
    
    # Renumber clusters continuously starting from 0, while suppressing too small clusters
    ClusterIndex = __SimplifyCluster(ClusterIndex, params)
    
    return ClusterIndex, SpheresSetTranslatedPosition



def __MergePeriodicCluster(SpheresSet, ClusterIndex, params):
    """ Merge the various instances of the clusters extending through the border of the domain and return a filtered array containing the definitive 
        cluster index of each element of SpheresSet.
        
        Only useful is the domain have periodic boundary condition.
    """
    PeriodicClusterIndex = - np.ones(params.Nspheres, dtype=int)
    ClusterRedirection = np.arange(np.max(ClusterIndex))
    SpheresSetTranslatedPosition = np.zeros( (params.Nspheres,3) )
    
    # Identify the real clusters, that is the clusters passing through the central (not duplicated) part of the system
    list_cluster_real, list_cluster_size = np.unique(ClusterIndex[:,0],return_counts=True)
    
    # Sort clusters by their size, in descending order 
    sorting_index = list_cluster_size.argsort()
    list_cluster_real = list_cluster_real[sorting_index[::-1]]
    
    # Iterate over all real clusters
    for c in list_cluster_real:
        # For each object...
        for index in range(params.Nspheres):
            # ...still not definitely attributed to a cluster...
            if PeriodicClusterIndex[index] == -1:
                # ...check if one of its duplicates...
                for per in range(8):
                    # ...pertain to cluster number c
                    if ClusterIndex[index,per] == c:
                        # If True, attribute this object to c or whatever other cluster c redirect to
                        PeriodicClusterIndex[index] = ClusterRedirection[c]
                        # Save the position of the duplicate that made the connection
                        SpheresSetTranslatedPosition[index,:] = __FindVirtualDuplicate(SpheresSet["X"][index], SpheresSet["Y"][index],\
                                                                                       SpheresSet["Z"][index], per, params)
                        # Note that the cluster this object was originally attributed to in fact redirect to c
                        ClusterRedirection[ ClusterIndex[index,0] ] = ClusterRedirection[c]
                        # Once connection has been done, no need to check the other duplicates
                        break
    
    return PeriodicClusterIndex, SpheresSetTranslatedPosition



def __SimplifyCluster(ClusterIndex, params):  
    """ Suppress the clusters smaller (in term of number of objects) than ``MinSize`` 
        
        Also simplify cluster indexing by renumbering the remaining clusters continuously starting from 1. To avoid confusion with 
        the usual Python indexing, non-attributed objects are denoted by -1 instead of 0.
    """
    # Initialize number of valid clusters (N.B : cluster indexing starts from 1)
    Ncluster = 0
    # Initialize attribution of objects to cluster : non-attributed objects are denoted by -1
    SimplifiedClusterIndex = - np.ones(params.Nspheres, dtype=int)
    # Identify existing clusters and compute the number of objects in each of them
    list_cluster, list_cluster_size = np.unique(ClusterIndex, return_counts=True)
    
    # Iterate over all existing clusters
    for index in range(len(list_cluster)):
        # Check if the number of objects in this cluster is greater or equal to threshold
        if list_cluster_size[index] >= params.MinSize:
            # Increment number of valid clusters
            Ncluster += 1
            # Renumber all objects in this cluster (N.B : cluster numbering start from 0)
            SimplifiedClusterIndex = np.where(ClusterIndex==list_cluster[index], Ncluster, SimplifiedClusterIndex)
    
    return SimplifiedClusterIndex