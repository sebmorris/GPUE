from gen_data import *

step = 100
for i in range(0,500):
    print(i)
    #item = proj_2d(xDim, yDim, zDim,"/media/james/ExtraDrive1/GPUE/data_elong","wfc_ev", (i*step))
    item = proj_2d(xDim, yDim, zDim,"/media/james/ExtraDrive1/GPUE/data_elong_w0.5n","wfc_ev", (i*step))

    if (i % 10 == 0):
        #item = var(xDim, yDim, zDim,"/media/james/ExtraDrive1/GPUE/data_elong","Edges_%s" %(i*step))
        #to_vtk(item, xDim, yDim, zDim, "/media/james/ExtraDrive1/GPUE/data_elong", "Edges_%s.vtk" % (i*step))

        item = var(xDim, yDim, zDim,"/media/james/ExtraDrive1/GPUE/data_elong_w0.5n","Edges_%s" %(i*step))
        to_vtk(item, xDim, yDim, zDim, "/media/james/ExtraDrive1/GPUE/data_elong_w0.5n", "Edges_%s.vtk" % (i*step))

    #item = proj_2d(xDim, yDim, zDim,"/media/james/ExtraDrive1/GPUE/data_elong_w0.5","wfc_ev", (i*10000))
    #item = proj_2d(xDim, yDim, zDim,"/media/james/ExtraDrive1/GPUE/data_elong_w1","wfc_ev", (i*10000))
    #item = proj_2d(xDim, yDim, zDim,"/media/james/ExtraDrive1/GPUE/data_elong_w1.5","wfc_ev", (i*10000))


