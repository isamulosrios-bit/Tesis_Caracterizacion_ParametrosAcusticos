import dolfinx
from mpi4py import MPI
from dolfinx import mesh
from dolfinx.io import XDMFFile

# 1. Crear la malla (Cubo de 10x10x10)
malla = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)

# 2. Confirmación en consola
print(f"DOLFINx funcionando. Celdas creadas: {malla.topology.index_map(3).size_global}")

# 3. GUARDAR EL ARCHIVO DIRECTAMENTE EN EL ESCRITORIO DE WINDOWS
ruta_escritorio = "/mnt/c/Users/isamu/OneDrive/Desktop/Temas Ing. Construcción UACH/Tesis_Poroelasticidad/resultados xdmf/resultado.xdmf"
with XDMFFile(MPI.COMM_WORLD, ruta_escritorio, "w") as xdmf:
    xdmf.write_mesh(malla)
    print("Archivo 'resultado.xdmf' generado con éxito en el Escritorio.")