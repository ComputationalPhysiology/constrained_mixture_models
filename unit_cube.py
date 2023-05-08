import matplotlib.pyplot as plt
import numpy as np
import math
from fenics import *
from material import HolzapfelOgden


# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = 4


class ActiveTension:
    def __init__(self,force_amp=1.0, t_start=50, tau1=20, tau2=110):
        self.force_amp = force_amp
        self.t_start = t_start
        self.tau1 = tau1
        self.tau2 = tau2

    def __call__(self, t):
        force_amp, t_start = self.force_amp, self.t_start
        tau1, tau2 = self.tau1, self.tau2

        beta = -math.pow(tau1/tau2, -1/(1 - tau2/tau1)) + math.pow(tau1/tau2,\
                -1/(-1 + tau1/tau2))
        force = ((force_amp)*(np.exp((t_start - t)/tau1) -\
                np.exp((t_start - t)/tau2))/beta) 
        
        #works for t array or scalar:
        force = force*(t>=t_start) + 0.0*(t<t_start)
        return force
    
    def plot(self,show=True):
        t = np.linspace(0,1000,1001)
        f = self(t)
        plt.plot(t,f)
        plt.xlabel('Time (ms)')
        plt.ylabel('Force (normalized)')
        plt.title('Reference active tension')
        if show:
            plt.show()


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

mat_params = HolzapfelOgden.default_parameters() 
print(type(mat_params))



mat_params["b_n"] = Constant(0.0)
mat_params["a_n"] = Constant(0.0)
mat_params["b_fn"] = Constant(0.0)
mat_params["a_fn"] = Constant(0.0)



mat = HolzapfelOgden(f0,n0,parameters={"a_n":0.0,"a_fn":0.0})


psi = mat.strain_energy(F)

P = diff(psi,F) # first Piola-Kirchhoff stress tensor

p_right = Constant(0.0) #the pressure load (zero for now)

# Definition of the weak form:
N = FacetNormal(mesh)
Gext = p_right * inner(v, det(F)*inv(F)*N) * ds(2) #ds(2) = right boundary
R = inner(P,grad(v))*dx + Gext 

#N = FacetNormal(mesh)
#Gext = p_right * inner(v, det(F)*inv(F)*N) * ds(2) #ds(2) = left boundary
#R = inner(P,grad(v))*dx + Gext 

# Step-wise loading (for plotting and convergence)
load_steps = 10
target_load = 1000.0

# The middle point on the right boundary
point0 = np.array([1.0,0.5,0.5])

d0 = np.zeros(3)                #displacement at point0
disp = np.zeros(load_steps) #array to store displacement for all steps

# Define and loop over load steps:
loads = np.linspace(0,target_load,load_steps)

disp_file = File("unit_cube/u.pvd")

for step in range(load_steps):
    # A stretch is a negative pressure
    p_right.assign(-loads[step])
    
    #solve the problem:
    solve(R == 0, u, bcs)
    
    #evaluate displacement at point defined above
    u.eval(d0,point0)
    disp[step] = d0[0]
    disp_file << u

#u.eval(d0,cp0)

cp0 = np.zeros(3)
cp1 = np.array([0,0,1])
cp2 = np.array([0,1,0])

checkpoints = [cp0,cp1,cp2,point0]

for cp in checkpoints:
    u.eval(d0,cp)
    print(f'Displacement at point ({cp[0]:1.2f},{cp[1]:1.2f},{cp[2]:1.2f}): ({d0[0]:1.4f},{d0[1]:1.4f},{d0[2]:1.4f})')
       
#displacement on x-axis, load on y-axis
#plt.figure(1)
plt.plot(disp,loads)
plt.xlabel('Displacement of point (1.0,0.5,0.5)')
plt.ylabel('Applied pressure load')

plt.show()