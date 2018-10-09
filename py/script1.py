from gen_data import *

f = open("w40_output.dat", "w")
for i in range(0,1):
    comx, comy = wfc_com(256, 256, 256, "data",
                         "wfc", i*1000)
    f.write("%s\t%s" % (comx, comy))
f.close()
