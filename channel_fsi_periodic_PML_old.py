
import gmsh

gmsh.initialize()
gmsh.model.add("fsi_occ")

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
meshfile_dir = "../../meshes/"
meshfile     = meshfile_dir + "fsi_occ.msh"
L =  0.05
H =  0.02

s_width  = L/32   # solid width
s_height = 0.75*H    # solid height
s_pos_x  = L/2    # solid center (x)

pml_length = 0.0 #0.05

lc  = s_width/2.0   # mesh size

tol = 1e-6

PML = pml_length > tol
# -----------------------------------------------------------------------------
# OCC geometry: fluid, solid, PML
# -----------------------------------------------------------------------------
fluid = gmsh.model.occ.addRectangle(0, 0, 0, L, H)

x0 = s_pos_x - s_width/2

solid = gmsh.model.occ.addRectangle(x0, 0, 0, s_width, s_height)

if PML:
    pml = gmsh.model.occ.addRectangle(L, 0, 0, pml_length, H)
    objects = [(2, fluid)]
    tools = [(2, solid), (2, pml)]
else:
    objects = [(2, fluid)]
    tools = [(2, solid)]

# -----------------------------------------------------------------------------
# Fragment -> guarantees shared interface nodes!
# -----------------------------------------------------------------------------
gmsh.model.occ.fragment(objects, tools)
gmsh.model.occ.synchronize()

# -----------------------------------------------------------------------------
# Classify regions
# -----------------------------------------------------------------------------
# Identify pieces
entities = gmsh.model.getEntities(2)

# Map surfaces correctly
fluid_surface = None
solid_surface = None
pml_surface   = None

for dim, tag in entities:
    x, y, _ = gmsh.model.occ.getCenterOfMass(dim, tag)

    if s_pos_x - s_width/2 < x < s_pos_x + s_width/2 and y < s_height + tol:
        solid_surface = tag
    elif  pml_length > 0 and x > L:
          pml_surface = tag
    else:
        fluid_surface = tag
print(fluid_surface, solid_surface,  pml_surface)
# -----------------------------------------------------------------------------
# Classify boundaries
# -----------------------------------------------------------------------------
# Helper functions
#
def curves_of_surface(s):
    return [c[1] for c in gmsh.model.getBoundary([(2, s)], oriented=False)]

def center_of_curve(c):
    return gmsh.model.occ.getCenterOfMass(1, c)

fluid_curves = curves_of_surface(fluid_surface)
obst_curves  = curves_of_surface(solid_surface)

if PML: pml_curves   = curves_of_surface(pml_surface)

inlet  = []
outlet = []
walls  = []
fluid_obst_interface = []
obstacle_fixed       = []

for c in fluid_curves:
    x, y, _ = center_of_curve(c)

    if abs(x) < tol:
        inlet.append(c)
    elif abs(x - L) < tol:
         outlet.append(c)
    elif abs(y) < tol or abs(y - H) < tol:
        walls.append(c)
    else:
        fluid_obst_interface.append(c)

for c in obst_curves:
    x, y, _ = center_of_curve(c)
    if abs(y) < tol:
        obstacle_fixed.append(c)
# -----------------------------------------------------------------------------
# Physical groups
# -----------------------------------------------------------------------------
gmsh.model.addPhysicalGroup(2, [fluid_surface], 1)
gmsh.model.setPhysicalName(2, 1, "fluid")

gmsh.model.addPhysicalGroup(2, [solid_surface], 2)
gmsh.model.setPhysicalName(2, 2, "solid")

if PML:
    gmsh.model.addPhysicalGroup(2, [pml_surface], 3)
    gmsh.model.setPhysicalName(2, 3, "pml")

gmsh.model.addPhysicalGroup(1, inlet, 11)
gmsh.model.setPhysicalName(1, 11, "Inlet")

gmsh.model.addPhysicalGroup(1, outlet, 12)
gmsh.model.setPhysicalName(1, 12, "Outlet")

gmsh.model.addPhysicalGroup(1, walls, 13)
gmsh.model.setPhysicalName(1, 13, "Walls")

gmsh.model.addPhysicalGroup(1, fluid_obst_interface, 14)
gmsh.model.setPhysicalName(1, 14, "FluidObstacleInterface")

gmsh.model.addPhysicalGroup(1, obstacle_fixed, 15)
gmsh.model.setPhysicalName(1, 15, "ObstacleFixed")
# -----------------------------------------------------------------------------
# Periodic boundary condition (inlet <-> outlet)
# -----------------------------------------------------------------------------
# Translation vector (x + L, y)
translation = [
    1, 0, 0, L,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
]

if pml_length < tol:
  gmsh.model.mesh.setPeriodic(1, outlet, inlet, translation)

# ------------------------
# Mesh
# ------------------------
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
gmsh.model.mesh.generate(2)
gmsh.write(meshfile)

gmsh.finalize()
