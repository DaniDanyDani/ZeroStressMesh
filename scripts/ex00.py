import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

import matplotlib.pyplot as plt
import numpy as np
from fenics import *

from src.guccionematerial import GuccioneMaterial

# -----------------------------------------------------------------------------
# CONFIGURAÇÕES DO COMPILADOR
# -----------------------------------------------------------------------------
parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["cpp_optimize"] = True


def load_ellipsoid_data():
    """
    Carrega a malha, marcadores de contorno e campos de microestrutura.
    
    Returns:
        mesh (Mesh): Malha computacional do FEniCS.
        mf (MeshFunction): Função que armazena os marcadores de contorno.
        numbering (dict): Dicionário mapeando os nomes para os IDs físicos.
        fibers (list): Lista com as funções de direção (fiber, sheet, cross_sheet).
    """
    mesh_path = os.path.join(DATA_DIR, "mesh.xml")
    facet_path = os.path.join(DATA_DIR, "facet_function.xml")

    mesh = Mesh(MPI.comm_world, mesh_path)
    mf = MeshFunction("size_t", mesh, facet_path)
    

    numbering = {
        "BASE": 10,
        "ENDO": 30,
        "EPI": 40
    }

    fiber_element = VectorElement(
        family="Quadrature",
        cell=mesh.ufl_cell(),
        degree=4,
        quad_scheme="default"
    )
    fiber_space = FunctionSpace(mesh, fiber_element)
    
    fiber = Function(fiber_space, os.path.join(DATA_DIR, "fiber.xml"))
    sheet = Function(fiber_space, os.path.join(DATA_DIR, "sheet.xml"))
    cross_sheet = Function(fiber_space, os.path.join(DATA_DIR, "cross_sheet.xml"))

    fibers = [fiber, sheet, cross_sheet]
    
    return mesh, mf, numbering, fibers


def compute_cavity_volume(mesh, mf, numbering, u=None):
    """
    Calcula o volume da cavidade (Endocárdio) usando o teorema da divergência.
    """
    X = SpatialCoordinate(mesh) 
    N = FacetNormal(mesh)

    if u is not None:
        I = Identity(3)
        F = I + grad(u)
        J = det(F)
        vol_form = (-1.0 / 3.0) * dot(X + u, J * inv(F).T * N)
    else:
        vol_form = (-1.0 / 3.0) * dot(X, N)

    ds_endo = Measure('ds', domain=mesh, subdomain_data=mf)

    return assemble(vol_form * ds_endo(numbering["ENDO"]))


# -----------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# -----------------------------------------------------------------------------

# 1. Carregamento de dados
mesh, boundary_markers, numbering, fibers = load_ellipsoid_data()

# 2. Espaço de Função para o Deslocamento (Vetor, Elemento Linear P1)
V = VectorFunctionSpace(mesh, 'P', 1)

# Redefine a medida de integração de contorno (ds) usando os marcadores
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

# 3. Condições de Contorno de Dirichlet
# Trava o movimento (clamp) na BASE do ventrículo (deslocamento zero)
clamp = Constant((0.0, 0.0, 0.0))
bc = DirichletBC(V, clamp, boundary_markers, numbering["BASE"])
bcs = [bc]

# 4. Variáveis do Problema Não-Linear
u = Function(V)       # Solução (deslocamento incógnito)
v = TestFunction(V)   # Função teste associada à formulação fraca

# 5. Cinemática das Grandes Deformações
I = Identity(3)
F = I + grad(u)       # Gradiente de Deformação
F = variable(F)       # Declarado como variável para permitir derivação simbólica

# 6. Modelo de Material
# Passa as direções ortotrópicas (e1, e2, e3) lidas dos arquivos XML
mat = GuccioneMaterial(
    e1=fibers[0], 
    e2=fibers[1], 
    e3=fibers[2], 
    kappa=1e3, 
    Tactive=0.0
)

# Energia de deformação e cálculo do Primeiro Tensor de Tensão de Piola-Kirchhoff (P)
psi = mat.strain_energy(F)
P = diff(psi, F)

# 7. Cargas Externas (Pressão no Endocárdio)
p_endo = Constant(0.0)  # Valor inicial de pressão (será incrementado)
N = FacetNormal(mesh)

Gext = p_endo * inner(v, det(F) * inv(F).T * N) * ds(numbering["ENDO"]) 

# 8. Formulação Fraca (Resíduo = Trabalho Interno - Trabalho Externo = 0)
R = inner(P, grad(v)) * dx + Gext 

# 9. Configuração de Incremento de Carga (Step-wise loading)
pressure_steps = 20
target_pressure = 10.0

pressures = np.linspace(0, target_pressure, pressure_steps)
volumes = np.zeros_like(pressures)

# Arquivo para visualizar no ParaView
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
disp_file = File(os.path.join(RESULTS_DIR, "ex00/u.pvd"))

# 10. Loop de Solução
for step in range(pressure_steps):
    # Atualiza a constante de pressão no solver
    p_endo.assign(pressures[step])
    
    print(f"Resolvendo passo {step+1}/{pressure_steps} com P = {pressures[step]:.2f}")
    
    # Resolve o problema não linear. O FEniCS usa Newton-Raphson por trás dos panos.
    solve(R == 0, u, bcs)
    
    # Armazena o volume e salva a malha deformada para o passo atual
    volumes[step] = compute_cavity_volume(mesh, boundary_markers, numbering, u)
    disp_file << u

# 11. Plotagem da Curva Pressão-Volume (PV Loop passivo)
plt.plot(volumes, pressures, marker='o')
plt.xlabel('Volume')
plt.ylabel('Pressão')
plt.title('Relação Pressão-Volume (Inflação Passiva)')
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "ex00/PxV.png"))
# plt.show()