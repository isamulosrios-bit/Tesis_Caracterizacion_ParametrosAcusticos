
import ufl

from dolfinx import fem
from mpi4py  import MPI
    

def compute_transmission(sol, domain, omega, rho_0, ds,
                         inlet_tag, outlet_tag):
    """
    Compute transmission coefficient for time-harmonic acoustics.

    Returns:
        T, P_in, P_out
    """

    # Extract pressure (first subspace)
    p = sol.sub(0)

    # Constants
    rho0 = fem.Constant(domain, complex(rho_0))
    omega_c = fem.Constant(domain,  complex(omega))

    # Acoustic velocity
    v = -(1/(1j * omega_c * rho0)) * ufl.grad(p)

    # Intensity
    I = 0.5 * ufl.real(p * ufl.conj(v))

    # Normal
    n = ufl.FacetNormal(domain)
    flux = ufl.dot(I, n)

    # Assemble powers
    P_in  = fem.assemble_scalar(fem.form(flux * ds(inlet_tag)))
    P_out = fem.assemble_scalar(fem.form(flux * ds(outlet_tag)))

    # MPI reduction
    P_in  = domain.comm.allreduce(P_in,  op=MPI.SUM)
    P_out = domain.comm.allreduce(P_out, op=MPI.SUM)

    P_in  = P_in.real
    P_out = P_out.real
    # Optional: fix sign if needed
    if P_in < 0:
        P_in = -P_in

    T = P_out / P_in if abs(P_in) > 1e-14 else 0.0

    return T, P_in, P_out

def compute_acoustic_power(sol, domain, omega, rho_0, ds, FACET_TAG):
    """
    Compute power integral for time-harmonic acoustics.

    Returns:
        Power
    """
    integral_type = ds.integral_type()
    
    # Extract pressure (first subspace)
    p = sol.sub(0)

    # Constants
    rho0    = fem.Constant(domain, complex(rho_0))
    omega_c = fem.Constant(domain,  complex(omega))

    # Acoustic velocity
    v = -(1/(1j * omega_c * rho0)) * ufl.grad(p)

    # Intensity
    I = 0.5 * ufl.real(p * ufl.conj(v))

    # Normal
    n = ufl.FacetNormal(domain)
    if integral_type == "exterior_facet":
              flux = ufl.dot(I, n)
    else: 
              flux = ufl.dot(I("+"), n("+"))

    # Assemble powers
    P   = fem.assemble_scalar(fem.form(flux * ds(FACET_TAG)))

    # MPI reduction
    P  = domain.comm.allreduce(P,  op=MPI.SUM)


    P  = P.real

    # Optional: fix sign if needed
    if P < 0: P = -P

    return P