import ufl, basix.ufl
import numpy as np
from mpi4py  import MPI
from dolfinx import mesh, fem

def compute_pml_start_and_length_x(domain, cell_tags, tag_value):
    """
    Returns the minimal x[0] coordinate among cells marked with tag_value
    and the x[0] - length of the domain
    """

    #  Get PML cell indices
    pml_cells = cell_tags.find(tag_value)

    if len(pml_cells) == 0:
        raise ValueError("No PML cells found.")

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)

    # Cell → vertex connectivity
    c2v = domain.topology.connectivity(tdim, 0)

    # Collect all vertex indices belonging to PML cells
    vertices = np.unique(
        np.concatenate([c2v.links(cell) for cell in pml_cells])
    )

    #  Extract coordinates
    x_coords = domain.geometry.x[vertices, 0]
    
    pml_start  = np.min(x_coords)
    pml_end    = np.max(x_coords)
    pml_length = pml_end - pml_start
    return pml_start, pml_length

def helmholtz_pml_coefficients(domain, omega, pml_start, alpha_max):
    """
    Construct PML coefficients A (matrix) and B (scalar)
    for a Cartesian Helmholtz PML in 3D.

    Parameters
    ----------
    domain    : dolfinx.mesh.Mesh
    omega     : float Angular frequency        
    pml_start : float Coordinate where PML starts (same for x,y,z)        
    alpha_max : float Maximum damping strength
        
    Returns
    -------
    A : ufl.Matrix Anisotropic diffusion tensor        
    B : ufl.Expr  Complex-valued scalar mass coefficient       
    """
    gdim = domain.geometry.dim

    x = ufl.SpatialCoordinate(domain)

    def alpha(xi):
        return ufl.conditional(ufl.gt(xi, pml_start),
                               alpha_max * (xi - pml_start)**2, 0.0)

    alpha1 = alpha(x[0])
    alpha2 = alpha(x[1])

    s1 = 1 + alpha1 / (1j * omega)
    s2 = 1 + alpha2 / (1j * omega)

    if gdim > 2:    
                alpha3 = alpha(x[2])
                s3     = 1 + alpha3 / (1j * omega)

                A = ufl.as_matrix([
                    [s2*s3/s1, 0,          0         ],
                    [0,        s1*s3/s2,   0         ],
                    [0,        0,          s1*s2/s3  ]])

                B = s1 * s2 * s3
    else:           
                A = ufl.as_matrix([ [s2/s1 , 0   ],[ 0   , s1/s2]])
                B = s1*s2

    return A, B

def helmholtz_pml_form(p, q, domain, k, omega, pml_start, alpha_max, dx_PML):
    """
    Return the PML-modified weak form for Helmholtz equation.

    Solves:
      -div(A grad p) - k^2 B p

    Parameters
    ----------
    p, q    : ufl.TrialFunction, ufl.TestFunction
    domain  : dolfinx.mesh.Mesh
    k       : float  Wavenumber
    omega   : float  Angular frequency
    pml_start : float
    alpha_max : float
    dx_PML    : integration measure

    Returns
    -------
    a : ufl.Form Bilinear form contribution
    """

    A, B = helmholtz_pml_coefficients(domain, omega, pml_start, alpha_max)

    a = (ufl.inner(A * ufl.grad(p), ufl.grad(q))
        - k**2 * B * p * q ) * dx_PML

    return a

def main():
    comm = MPI.COMM_WORLD

    print("Testing module for PML")
    Nx, Ny, Lx, Ly  = 4, 2, 2.0, 1.0

    domain = mesh.create_rectangle(comm, [[0.0, 0.0], [Lx, Ly]],
                                         [Nx, Ny],
                                         cell_type=mesh.CellType.triangle)

    gdim = domain.geometry.dim
    tdim = domain.topology.dim

    # --- locate cells ---
    left_cells  = mesh.locate_entities(domain, tdim,lambda x: x[0] <= Lx/2)
    right_cells = mesh.locate_entities(domain, tdim,lambda x: x[0] > Lx/2)

    # --- build meshtags ---
    indices = np.hstack([left_cells, right_cells])
    values  = np.hstack([
                        np.full(len(left_cells), 0, dtype=np.int32),
                        np.full(len(right_cells), 3, dtype=np.int32)])

    cell_tags = mesh.meshtags(domain, tdim, indices, values)

    pml_start     = Lx/2
    alpha_max_val = 2.0
    omega_val     = 1  

    k_val     = omega_val/300

    dx        = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    
    AA, BB    = helmholtz_pml_coefficients(domain, omega_val, pml_start, alpha_max_val)
    print("PML A:", AA)
    print("PML B:", BB)

    Fluid_Element = basix.ufl.element("Lagrange", domain.basix_cell(), degree=1)
    Solid_Element = basix.ufl.element("Lagrange", domain.basix_cell(), degree=1, shape=(gdim,))

    Mixed_Element = basix.ufl.mixed_element([Fluid_Element, Solid_Element])

    W = fem.functionspace(domain, Mixed_Element)  

    print("*** FUNCTION SPACES DONE")

    p, u = ufl.TrialFunctions(W)
    q, v = ufl.TestFunctions(W)

  
    aa =  helmholtz_pml_form(p, q, domain, k_val, omega_val, pml_start, alpha_max_val, dx(3))

    print("PML form contribution:\n",aa)

if __name__ == "__main__":
    main()