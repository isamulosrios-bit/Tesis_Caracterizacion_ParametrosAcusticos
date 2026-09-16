import gmsh

gmsh.initialize()
gmsh.model.add("fsi_occ")

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
meshfile_dir = "../../meshes/"
meshfile     = meshfile_dir + "fsi_occ_PML.msh"
L =  0.1
H =  0.02

s_width  = L/32    # solid width
s_height = 0.75*H  # solid height
s_pos_x  = L/2     # solid center (x)

pml_length = 2.0*L

lc  = s_width/2.0
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
    tools   = [(2, solid), (2, pml)]
else:
    objects = [(2, fluid)]
    tools   = [(2, solid)]

# -----------------------------------------------------------------------------
# Fragment and synchronize
# -----------------------------------------------------------------------------
outDimTags, outDimTagsMap = gmsh.model.occ.fragment(objects, tools)
gmsh.model.occ.synchronize()

# -----------------------------------------------------------------------------
# Identify surfaces by provenance
# -----------------------------------------------------------------------------
# New entities coming from the original fluid
fluid_from_obj = {tag for (dim, tag) in outDimTagsMap[0] if dim == 2}
# New entities coming from the original solid
solid_from_tool = {tag for (dim, tag) in outDimTagsMap[1] if dim == 2}

if PML:
    pml_from_tool = {tag for (dim, tag) in outDimTagsMap[2] if dim == 2}
else:
    pml_from_tool = set()

# Overlap (solid) appears in both fluid_from_obj and solid_from_tool
# → remove it from the fluid set
fluid_surfaces = list(fluid_from_obj - solid_from_tool - pml_from_tool)
solid_surfaces = list(solid_from_tool)
pml_surfaces   = list(pml_from_tool)

print("fluid_surfaces:", fluid_surfaces)
print("solid_surfaces:", solid_surfaces)
print("pml_surfaces:", pml_surfaces)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def curves_of_surface(s):
    return [c[1] for c in gmsh.model.getBoundary([(2, s)], oriented=False)]

def center_of_curve(c):
    # IMPORTANT: use OCC API here (this is what you had originally)
    return gmsh.model.occ.getCenterOfMass(1, c)

# -----------------------------------------------------------------------------
# Classify boundaries
# -----------------------------------------------------------------------------
fluid_curves = []
for s in fluid_surfaces:
    fluid_curves.extend(curves_of_surface(s))
fluid_curves = list(set(fluid_curves))

inlet  = []
outlet = []
walls  = []
fluid_obst_interface = []

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

solid_curves = []
for s in solid_surfaces:
    solid_curves.extend(curves_of_surface(s))
solid_curves = list(set(solid_curves))

obstacle_fixed = []
for c in solid_curves:
    x, y, _ = center_of_curve(c)
    if abs(y) < tol:
        obstacle_fixed.append(c)

# -----------------------------------------------------------------------------
# Physical groups
# -----------------------------------------------------------------------------
gmsh.model.addPhysicalGroup(2, fluid_surfaces, 1)
gmsh.model.setPhysicalName(2, 1, "fluid")

gmsh.model.addPhysicalGroup(2, solid_surfaces, 2)
gmsh.model.setPhysicalName(2, 2, "solid")

if PML and pml_surfaces:
    gmsh.model.addPhysicalGroup(2, pml_surfaces, 3)
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

# Periodic BC
translation = [
    1, 0, 0, L,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
]

if pml_length < tol:
    gmsh.model.mesh.setPeriodic(1, outlet, inlet, translation)

# Mesh
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
gmsh.model.mesh.generate(2)
gmsh.write(meshfile)

gmsh.finalize()