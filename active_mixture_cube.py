import matplotlib.pyplot as plt
import numpy as np
from fenics import *
from mixture import *

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = 4



# Setup the mesh and the function space for the solutions
mesh = UnitCubeMesh(4,4,4)
V = VectorFunctionSpace(mesh, "Lagrange", 2)


# Define functions
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration

# Mark boundary subdomains
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)

xy0 = CompiledSubDomain("near(x[0],0) && near(x[1],0)")
xz0 = CompiledSubDomain("near(x[0],0) && near(x[2],0)")


boundary_markers = MeshFunction("size_t", mesh,mesh.topology().dim() - 1)
boundary_markers.set_all(0)
left.mark(boundary_markers, 1) #not strictly needed
right.mark(boundary_markers, 2)

# Redefine boundary measure
ds = Measure('ds',domain=mesh,subdomain_data=boundary_markers)

"""
Define Dirichlet boundaries (avoid rigid body motion)
u_x = 0 for x = 0
u_y = 0 for x=0 && y=0
u_z = 0 for x=0 && z=0
"""
bcx = DirichletBC(V.sub(0), Constant(0), left)
bcy = DirichletBC(V.sub(1), Constant(0),xy0,"pointwise")
bcz = DirichletBC(V.sub(2), Constant(0),xz0,"pointwise")

bcs = [bcx,bcy,bcz]

d = len(u)
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
F = variable(F)

# Tissue microstructure
f0 = as_vector([ 1.0, 0.0, 0.0 ]) #fiber
n0 = as_vector([ 0.0, 1.0, 0.0 ]) #sheet_normal


a_coll = 568    #J/kg
b_coll = 11.2   #unitless
collagen = Fiber(f0,parameters = {"a_f":a_coll, "b_f":b_coll})
#collagen2 = Fiber(n0,parameters = {"a_f":a_coll, "b_f":b_coll})


a_myo  = 7.6     #J/kg
b_myo  = 11.4    #unitless
max_active = 25    #kPa
active = ActiveTension(force_amp = max_active)
myocytes = ActiveFiber(f0,active,parameters = {"a_f":a_myo, "b_f":b_myo})

kappa_vol = Constant(150)    #kPa
#a_mat = Constant(35.0)
#b_mat = Constant(8.023)
#matrix = FungBackground(parameters = {"a":a_mat, "b":b_mat, "kappa":kappa_vol}) 

c1 = Constant(72) 
matrix = NeoHookeanBackground(parameters = {"c1":c1, "kappa":kappa_vol}) 

constituents = [collagen, matrix, myocytes]
mass_fractions = [Constant(0.1), Constant(0.3), Constant(0.6)]

mixture = Mixture(constituents, mass_fractions)

psi = mixture.strain_energy(F)

P = diff(psi,F) # first Piola-Kirchhoff stress tensor

p_right = Constant(0.0) #the pressure load (zero for now)

# Definition of the weak form:
N = FacetNormal(mesh)
Gext = p_right * inner(v, det(F)*inv(F)*N) * ds(2) #ds(2) = right boundary
R = inner(P,grad(v))*dx + Gext 


# The middle point on the right boundary
point0 = np.array([1.0,0.5,0.5])

disp_file = File("unit_cube/u.pvd")

time = np.linspace(0,1000,200)
d0 = np.zeros(3)                #displacement at point0
disp = np.zeros_like(time) #array to store displacement for all steps

for n, t in enumerate(time):
    myocytes.update_active(t)
    solve(R == 0, u, bcs)
    
    #evaluate displacement at point defined above
    u.eval(d0,point0)
    disp[n] = d0[0]
    disp_file << u


print(f'Minimum stretch ratio = {1+min(disp)}')


plt.plot(time,disp+1)
plt.xlabel('Time (ms)')
plt.ylabel('Fiber stretch ($\lambda$)')
plt.show()