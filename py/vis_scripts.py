import gen_data
'''
proj_var2d(xDim, yDim, zDim, "data", "fieldz")
proj_2d(xDim, yDim, zDim,"data","wfc",50000)
#proj_2d(xDim, yDim, zDim,"data","wfc",1000000)
#proj_k2d(xDim, yDim, zDim,"data","wfc",500000)

item_wfc = wfc_density(xDim, yDim, zDim,"data","wfc",50000)
#item_wfc = wfc_density(xDim, yDim, zDim,"data","wfc",1000000)
item_phase = wfc_phase(xDim, yDim, zDim,"data","wfc",50000)
#item_phase = wfc_phase(xDim, yDim, zDim,"data","wfc",1000000)
#item_var = var(xDim, yDim, zDim,"data","By_0")
#item2_var = var(xDim, yDim, zDim,"data","Edges_0")

#to_bvox(item_wfc, xDim, yDim, zDim, 1, "test_wfc.bvox")
#to_bvox(item_phase, xDim, yDim, zDim, 1, "test_phase.bvox")
#to_bvox(item_var, xDim, yDim, zDim, 1, "test_var.bvox")
#to_bvox(item2_var, xDim, yDim, zDim, 1, "test_edges.bvox")
'''
'''

print("iterating through data...")
# Writing out the potential
#item_pot = var(xDim, yDim, zDim, "data", "V_0")
#to_bvox(item_pot, xDim, yDim, zDim, 1, "test_pot.bvox")
#item_gauge = var(xDim, yDim, zDim, "data", "Az_0")
#to_bvox(item_gauge, xDim, yDim, zDim, 1, "test_gauge.bvox")
#item_edges = var(xDim, yDim, zDim,"data","Edges_0")
#to_bvox(item_edges, xDim, yDim, zDim, 1, "test_edges.bvox")
#to_vtk(item_edges, xDim, yDim, zDim, 1, "test_edges.vtk")
for i in range(0,101):
#for i in [86, 143, 195, 95, 152, 204]:
#for i in [4, 40]:
    num = i * 100
    proj_2d(xDim, yDim, zDim, "data", "wfc_ev", num)
    #item = wfc_density(xDim, yDim, zDim, "data", "wfc_ev", num)
    #item_ph = wfc_phase(xDim, yDim, zDim, "data", "wfc", num)
    #to_bvox(item, xDim, yDim, zDim, 1, "wfc_%s.bvox" %num)
    #to_vtk(item, xDim, yDim, zDim, 1, "wfc_%s.vtk" %num)
    #to_bvox(item_ph, xDim, yDim, zDim, 1, "wfc_ph_%s.bvox" %num)

'''
'''

for i in range(0,21):
    num = i * 1
    print(num)
    item = var(xDim, yDim, zDim, "data", "Edges_%s"%num)
    to_bvox(item, xDim, yDim, zDim, 1, "EDGES_%s.bvox" %num)

'''
'''
proj_var2d(xDim, yDim, zDim, "data", "fieldz")
item = var(xDim, yDim, zDim, "data", "fieldz")
to_bvox(item, xDim, yDim, zDim, 1, "HE11check.bvox")
'''

'''
proj_2d(xDim, yDim, zDim, "data", "wfc_ramp", 100000)
#item_edges = var(xDim, yDim, zDim,"data","Edges_0")
#to_bvox(item_edges, xDim, yDim, zDim, 1, "test_edges.bvox")
'''

'''
print("start")
item_edges = var(xDim, yDim, zDim,"data","Edges_std")
to_bvox(item_edges, xDim, yDim, zDim, 1, "edges_std.bvox")
to_vtk(item_edges, xDim, yDim, zDim, 1, "edges_std.vtk")
print("starting linpol")
item_edges = var(xDim, yDim, zDim,"data","Edges_linpol")
to_bvox(item_edges, xDim, yDim, zDim, 1, "edges_linpol.bvox")
to_vtk(item_edges, xDim, yDim, zDim, 1, "edges_linpol.vtk")
'''

'''
item = wfc_density(xDim, yDim, zDim, "data", "wfc", 100000)
to_vtk(item, xDim, yDim, zDim, 1, "wfc.vtk")
to_bvox(item, xDim, yDim, zDim, 1, "wfc_check.bvox")
'''

#proj_phase_2d(xDim, yDim, zDim, "data", "wfc", 660)
#proj_phase_2d(xDim, yDim, zDim, "data", "wfc", 320)
#proj_phase_2d(xDim, yDim, zDim, "data", "wfc", 1000)
#item_pot = var(xDim, yDim, zDim, "data", "V_0")
#to_bvox(item_pot, xDim, yDim, zDim, 1, "test_pot.bvox")
#item_edges = var(xDim, yDim, zDim,"data","Edges_0")
#to_bvox(item_edges, xDim, yDim, zDim, 1, "test_edges.bvox")
#item = wfc_density(xDim, yDim, zDim,"data","wfc", 0)
#to_bvox(item, xDim, yDim, zDim, 1, "test_wfc.bvox")

'''
item_gx = var(xDim, yDim, zDim,"data","Ax_0")
to_bvox(item_gx, xDim, yDim, zDim, 1, "Ax_0.bvox")
item_gy = var(xDim, yDim, zDim,"data","Ay_0")
to_bvox(item_gy, xDim, yDim, zDim, 1, "Ay_0.bvox")
item_gz = var(xDim, yDim, zDim,"data","Az_0")
to_bvox(item_gz, xDim, yDim, zDim, 1, "Az_0.bvox")
item_V = var(xDim, yDim, zDim,"data","V_0")
to_bvox(item_V, xDim, yDim, zDim, 1, "V_0.bvox")
item_K = var(xDim, yDim, zDim,"data","K_0")
to_bvox(item_K, xDim, yDim, zDim, 1, "K_0.bvox")
'''
'''
for i in range(0,11):
    num = i*100000
    print(num)
    item = var(xDim, yDim, zDim, "data", "Edges_%s" %num)
    to_bvox(item, xDim, yDim, zDim, 1, "EDGES_%s.bvox" %num)

'''
'''
num = 500000
proj_2d(xDim, yDim, zDim, "data", "wfc", num)
item = wfc_density(xDim, yDim, zDim, "data", "wfc", num)
item_edges = var(xDim, yDim, zDim, "data", "Edges_0")
to_bvox(item, xDim, yDim, zDim, 1, "wfc_%s.bvox" %num)
to_bvox(item_edges, xDim, yDim, zDim, 1, "Edges_0.bvox")
to_vtk(item, xDim, yDim, zDim, 1, "wfc_%s.vtk" %num)

'''
'''
proj_var2d(xDim, yDim, zDim, "data", "V_0")
#proj_2d(xDim, yDim, zDim, "data", "wfc_ev", 0)
'''
#item = var(xDim, yDim, zDim,"data","Thresh_0")
#to_vtk(item, xDim, yDim, zDim, 1, "test.vtk")
#item = var(xDim, yDim, zDim,"data","Edges_0")
#to_vtk(item, xDim, yDim, zDim, 1, "test_wfc.vtk")

comx, comy = wfc_com(xDim, yDim, zDim, "data", "wfc", 0)
print(comx)
print(comy)

