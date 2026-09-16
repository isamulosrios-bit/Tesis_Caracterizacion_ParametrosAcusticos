#!/usr/bin/env python3
"""
FEniCSx simulation script.
Solves: linear Euler fluid in a channel with linear elastic obstacle
Method: Mixed element prescribing homogeneous Dirichlet bcs for 
        - pressure dofs in the solid part
        - displacement dofs in the fluid part
        in order to avoid orphan dofs
"""
import os
import numpy as np
import ufl, dolfinx, basix.ufl

from mpi4py      import MPI
from petsc4py    import PETSc

import dolfinx.io.gmsh
from dolfinx      import fem, io 

from dolfinx.fem.petsc import LinearProblem

from enum import IntEnum

from material             import IsotropicLinearElastic
from compute_transmission import compute_acoustic_power, compute_transmission
from pml                  import helmholtz_pml_form, helmholtz_pml_coefficients
from pml                  import compute_pml_start_and_length_x

#------------------------------------------------------------------------------
# COMPLEX NUMBERS ACTIVE?
#------------------------------------------------------------------------------
if np.dtype(PETSc.ScalarType).kind !="c":
   print("ERROR. Run 'source dolfinx-complex-mode' in docker env first")
   quit()
print("UFL VERSION:", ufl.__version__)
#------------------------------------------------------------------------------
# PARAMETERS 
#------------------------------------------------------------------------------
# Usamos una ruta relativa limpia asumiendo que el archivo .msh está directamente en la carpeta Malla/
mesh_file    = "Malla/fsi_occ_PML.msh"  
res_dir      = "./"
res_filename = res_dir + "channel_euler_FSI"

ipol_order   = 1
gdim         = 2

pml        = False

p_in_value = 1.0 +0.0j                                    # inlet pressure Pa

freq = np.arange(1, 5000, 100)                    # frequency range, Hz
 
pml_eps      = 1e-02
absorp_param = 1e-04
#------------------------------------------------------------------------------
# MATERIALS CONSTANTS 
#------------------------------------------------------------------------------
Emod, nu  = 5e4,0.3                            # solid Pa, -
rho_s     = 50.0                               # kg/m^3
rho_0, c  = 1.225, 340                         # fluid (air) kg/m^3, m/s
#------------------------------------------------------------------------------
# HELPERS
#------------------------------------------------------------------------------
class TAG(IntEnum):
    FLUID, SOLID, PML    = 1,2,3
    Inlet, Outlet, Walls = 11,12,13
    Gamma_fs, Gamma_u    = 14, 15
#------------------------------------------------------------------------------
# READ MESH
#------------------------------------------------------------------------------
if not os.path.exists(mesh_file):
    # Si falla, buscamos de forma recursiva o en la ruta de trabajo actual el archivo para autocompletar
    found_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file == "fsi_occ_PML.msh":
                found_files.append(os.path.join(root, file))
    if found_files:
        mesh_file = found_files[0]
        print(f"*** Archivo encontrado automáticamente en: {mesh_file}")
    else:
        raise FileNotFoundError(f"No se encuentra 'fsi_occ_PML.msh' en ninguna subcarpeta de {os.getcwd()}. Verifica que el archivo exista.")

mesh_data = dolfinx.io.gmsh.read_from_msh(
    mesh_file, MPI.COMM_WORLD, 0, gdim=gdim)

domain = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags

if np.any(cell_tags.values == TAG.PML):
    pml        = True
    pml_start, pml_length = compute_pml_start_and_length_x(domain, cell_tags,TAG.PML)
    print("Mesh contains PML cells!")
    print("PML starts at x[0]=", pml_start)
    print("PML  x - length   =", pml_length)
    pml_param = 3*c/ pml_length*np.log(1/pml_eps)
    print("PML alpha_max = ", pml_param)
else:
    print("No PML cells.")

tdim = domain.topology.dim 
fdim = domain.topology.dim - 1

fluid_cells      = cell_tags.find(TAG.FLUID)
solid_cells      = cell_tags.find(TAG.SOLID)

if pml: pml_cells = cell_tags.find(TAG.PML)

interface_facets  = facet_tags.find(TAG.Gamma_fs)
inlet_facets      = facet_tags.find(TAG.Inlet)
outlet_facets     = facet_tags.find(TAG.Gamma_fs)
clamped_facets    = facet_tags.find(TAG.Gamma_u)

print("*** READ MSH DONE")
#------------------------------------------------------------------------------
# ELEMENTS AND  FUNCTION SPACES 
#------------------------------------------------------------------------------
Fluid_Element = basix.ufl.element("Lagrange", domain.basix_cell(), degree=1)
Solid_Element = basix.ufl.element("Lagrange", domain.basix_cell(), degree=1, shape=(gdim,))

Mixed_Element = basix.ufl.mixed_element([Fluid_Element, Solid_Element])

W = fem.functionspace(domain, Mixed_Element)  

print("*** FUNCTION SPACES DONE")
#------------------------------------------------------------------------------
# DIRICHLET BOUNDARY CONDITIONS
#------------------------------------------------------------------------------
W_F = W.sub(0)   # fluid (pressure) subspace
W_S = W.sub(1)   # solid (displacement) subspace

V_F, V_F_to_W_F  = W_F.collapse()
V_S, V_S_to_W_S  = W_S.collapse()

u_zero_func = fem.Function(V_S)
u_zero_func.x.array[:] = 0.0 + 0.0j

clamped_dofs = fem.locate_dofs_topological((W_S, V_S), fdim,
                                           clamped_facets)
bc_clamped = fem.dirichletbc(u_zero_func,clamped_dofs, W_S)

bcs=[bc_clamped]
print("*** DIRICHLET BOUNDARY CONDITIONS INLET AND CLAMPED DONE")
#------------------------------------------------------------------------------
# DIRICHLET BOUNDARY CONDITIONS ARTIFICIAL 
#------------------------------------------------------------------------------
domain.topology.create_connectivity(tdim, tdim)

p_dofs_solid = fem.locate_dofs_topological(W_F, tdim,solid_cells)
u_dofs_fluid = fem.locate_dofs_topological((W_S, V_S), tdim, fluid_cells)

p_dofs_interface = fem.locate_dofs_topological(W_F, fdim, interface_facets)
u_dofs_interface = fem.locate_dofs_topological((W_S, V_S), fdim,
                                               interface_facets)  

p_dofs_solid_only = np.setdiff1d(p_dofs_solid, p_dofs_interface)

u_dofs_fluid_only = [np.setdiff1d(u_dofs_fluid[0], u_dofs_interface[0]),
                     np.setdiff1d(u_dofs_fluid[1], u_dofs_interface[1])]

p_zero      = fem.Constant(domain, 0.0+0.0j)
bc_p_solid  = fem.dirichletbc(p_zero,     p_dofs_solid_only, W_F)
bc_u_fluid  = fem.dirichletbc(u_zero_func,u_dofs_fluid_only, W_S)

bcs = bcs + [bc_p_solid, bc_u_fluid]

if pml:
     u_dofs_pml = fem.locate_dofs_topological((W_S, V_S), tdim, pml_cells)
     bc_u_pml   = fem.dirichletbc(u_zero_func,u_dofs_pml, W_S)
     bcs        = bcs + [bc_u_pml]

#------------------------------------------------------------------------------
# MEASURES AND INDICATOR FUNCTIONS FOR INTERFACE INTEGRATION 
#------------------------------------------------------------------------------
dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags) # exterior
dS = ufl.Measure("dS", domain=domain, subdomain_data=facet_tags) # interior facets

if pml: 
        ds_out = ufl.Measure("dS", domain=domain, subdomain_data=facet_tags)
else:
        ds_out = ds

Q0 = fem.functionspace(domain, ("DG", 0)) 

chi_f = fem.Function(Q0)                  
chi_s = fem.Function(Q0)                  

chi_f.x.array[:] = 0.0
chi_s.x.array[:] = 0.0                    

chi_f.x.array[fluid_cells] = 1.0          
chi_s.x.array[solid_cells] = 1.0          

chi_f.x.scatter_forward()                 
chi_s.x.scatter_forward()

print("*** INDICATOR FUNCTIONS FOR INTERFACE CONDITIONS DONE")
#------------------------------------------------------------------------------
# VARIATIONAL PROBLEM 
#------------------------------------------------------------------------------
elast = IsotropicLinearElastic(E=Emod, nu=nu,
                               dim=domain.geometry.dim,   
                               plane_stress=False )

p, u = ufl.TrialFunctions(W)
q, v = ufl.TestFunctions(W)

zero     = fem.Constant(domain, PETSc.ScalarType(0))
zero_vec = fem.Constant(domain, PETSc.ScalarType(np.zeros(gdim)))
omega    = fem.Constant(domain, PETSc.ScalarType(0))
k        = fem.Constant(domain, PETSc.ScalarType(0))

n = ufl.FacetNormal(domain)

a_f_Omega  =         ufl.inner(ufl.grad(p), ufl.grad(q))*dx(TAG.FLUID) \
            - k**2 *ufl.inner(p,q)*dx(TAG.FLUID)

if pml:
        A, B      = helmholtz_pml_coefficients(domain, omega, pml_start,pml_param)

        a_f_right = (ufl.inner(A * ufl.grad(p), ufl.grad(q))
                 - k**2 * B * p * ufl.conj(q) ) * dx(TAG.PML)
else:
      a_f_right =  1.0j*k*ufl.inner(p,q)*ds(TAG.Outlet)   

a_f_Gamma_in = 1.0j*k*ufl.inner(p,q)*ds(TAG.Inlet) 

a_s_Omega    =   ufl.inner(elast.sigma(u), elast.eps(v))*dx(TAG.SOLID) \
             - rho_s*omega**2*ufl.inner(u,v) *dx(TAG.SOLID)

a_fsi_disp = -rho_0 * omega**2 * (
    chi_s('+') * ufl.inner(u('+'), n('+')) * ufl.conj(q('+'))
  + chi_s('-') * ufl.inner(u('-'), n('-')) * ufl.conj(q('-'))
) * dS(TAG.Gamma_fs)

a_fsi_trac = rho_0 * (
    chi_f('+') * p('+') * ufl.inner(n('+'), v('+'))
  + chi_f('-') * p('-') * ufl.inner(n('-'), v('-'))
) * dS(TAG.Gamma_fs)

L_f = ufl.inner(zero, q) * dx(TAG.FLUID)

L_s = ufl.inner(zero_vec, v)*dx(TAG.SOLID)

L_f_Gamma_in  = 2j*k*p_in_value*ufl.conj(q) * ds(TAG.Inlet)

a_fsi = a_f_Omega + a_s_Omega +a_f_right + a_fsi_disp +  a_fsi_trac + \
        a_f_Gamma_in     
L_fsi = L_f + L_s + L_f_Gamma_in

sol      = fem.Function(W)
sol.name = "solution"

petsc_options={"ksp_type": "preonly",
               "pc_type": "lu",
               "pc_factor_mat_solver_type": "mumps"}

problem = LinearProblem(a_fsi,L_fsi,bcs=bcs, u=sol, 
                        petsc_options=petsc_options)
                            
print("*** VARIATIONAL PROBLEM DONE")
#------------------------------------------------------------------------------
# FREQUENCY LOOP 
#------------------------------------------------------------------------------
T_values = np.zeros(len(freq), dtype=float)

for nf in range(0, len(freq)):
    k.value     = 2 * np.pi * freq[nf] / c
    omega.value = 2 * np.pi * freq[nf]
    
    if nf % 10 == 0:
       print(f"*** LOOP {nf} with omega: {omega.value:.1f} k: {k.value:.1f} ")
    problem.solve()

    A_mat = problem.A
    b_vec = problem.b

    p_sol, u_sol = sol.split()
    
    P_in  = compute_acoustic_power(sol, domain, omega, rho_0, ds,     TAG.Inlet)
    P_out = compute_acoustic_power(sol, domain, omega, rho_0, ds_out, TAG.Outlet)  

    T_values[nf] = P_out/P_in

print("*** FREQUENCY LOOP DONE")

data = np.column_stack((freq, T_values))

transmission_data_file = res_dir + "transmission.txt"

np.savetxt(transmission_data_file, data, header="Frequency(Hz)   Transmission",
           comments="")

p_sol = sol.sub(0).collapse()
u_sol = sol.sub(1).collapse()

p_sol.name = "pressure"
u_sol.name = "displacement"

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, res_filename + "_p.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(p_sol)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, res_filename + "_u.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_sol)