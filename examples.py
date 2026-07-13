from SegTool import WatershedSegmentation

## Fig 12
WatershedSegmentation("image", "demo_data/bananas.tiff", savefile="bananas_default")


## Fig 13
WatershedSegmentation("dataset", "demo_data/default.csv")


## Fig 14
WatershedSegmentation("image", "demo_data/bananas.tiff", savefile="bananas_withsep",
                      InputRods="demo_data/bananas_separators.tiff")


## Fig 15
WatershedSegmentation("dataset", "demo_data/default.csv", savefile="default_seg_withsep",
                      InputRods="demo_data/default_separators.csv")


## Fig 16
WatershedSegmentation("dataset", "demo_data/resolution.csv", savefile="resolution1", resolution=1)

WatershedSegmentation("dataset", "demo_data/resolution.csv", savefile="resolution5", resolution=5)


## Fig 18
WatershedSegmentation("dataset", "demo_data/domain_size.csv", savefile="domain_size_auto")

WatershedSegmentation("dataset", "demo_data/domain_size.csv", savefile="domain_size_auto2",
                      PeriodicBoundaries=True)

WatershedSegmentation("dataset", "demo_data/domain_size.csv", savefile="domain_size_xmax4.2",
                      PeriodicBoundaries=True, xmax=4.2) # This will print a warning 

WatershedSegmentation("dataset", "demo_data/domain_size.csv", savefile="domain_size_xmax4.2_ymax3",
                      PeriodicBoundaries=True, xmax=4.2, ymax=3)

WatershedSegmentation("dataset", "demo_data/domain_size.csv", savefile="domain_size_xmax4.2_ymax3_zmax3.5",
                      PeriodicBoundaries=True, xmax=4.2, ymax=3, zmax=3.5)


## Fig 19
WatershedSegmentation("dataset", "demo_data/scaling_up.csv", savefile="scaling_up1") # default value of dil_coeff is 1.0

WatershedSegmentation("dataset", "demo_data/scaling_up.csv", savefile="scaling_up1.2", dil_coeff=1.2)

WatershedSegmentation("dataset", "demo_data/scaling_up.csv", savefile="scaling_up1.2_withsep",
                      dil_coeff=1.2,
                      InputRods="demo_data/scaling_seps.csv")

WatershedSegmentation("dataset", "demo_data/scaling_down.csv", savefile="scaling_down1") # default value of dil_coeff is 1.0

WatershedSegmentation("dataset", "demo_data/scaling_down.csv", savefile="scaling_down0.9", dil_coeff=0.9)


## Fig 20
WatershedSegmentation("image", "demo_data/bananas.tiff", savefile="bananas_pixelsize1-1-6", pixelsize=[1,1,6])


## Fig 21
WatershedSegmentation("dataset", "demo_data/smoothing.csv", savefile="smoothing0", smooth_coeff=0)
WatershedSegmentation("dataset", "demo_data/smoothing.csv", savefile="smoothing1") # default value of smooth_coeff is 1.0
WatershedSegmentation("dataset", "demo_data/smoothing.csv", savefile="smoothing5", smooth_coeff=5)


## Fig 22
WatershedSegmentation("image", "demo_data/bananas.tiff", savefile="bananas_smoothing0", smooth_coeff=0)
WatershedSegmentation("image", "demo_data/bananas.tiff", savefile="bananas_smoothing1") # default value of smooth_coeff is 1.0
WatershedSegmentation("image", "demo_data/bananas.tiff", savefile="bananas_smoothing10", smooth_coeff=10)


## Graphical abstract
WatershedSegmentation("image", "demo_data/mouse_adipose_tissue.tif", savefile="mouse_adipose_tissue_seg",
                      pixelsize=[1.84, 1.84, 1], smooth_coeff=10, MinSize=400000)

WatershedSegmentation("dataset", "dataset_spheres.csv", savefile="dataset_spheres_seg",
                      InputRods="dataset_rods.csv",
                      PeriodicBoundaries=True, xmax=15, ymax=15, zmax=15,
                      resolution=3, dil_coeff=1.5, smooth_coeff=0.5, MinSize=10)