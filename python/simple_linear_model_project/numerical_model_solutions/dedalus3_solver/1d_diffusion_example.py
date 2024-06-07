import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Parameters
Lx = 10
Nx = 256
alpha = 0.1

dealias = 3/2
stop_sim_time = 20
timestepper = d3.RK222
timestep = 2e-3
dtype = np.float64

# CFL = c*timestep/(Lx/Nx)
# print(f"CFL = {CFL:0.3f}")

# Bases
xcoord =d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=dtype)
xbasis = d3.RealFourier(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)

# Fields
u = dist.Field(name='u', bases=xbasis)

# Substitutions
dx = lambda A: d3.Differentiate(A, xcoord)

# Problem
problem = d3.IVP([u], namespace=locals())
problem.add_equation("dt(u) - alpha*dx(dx(u)) = 0")

# Initial Conditions
x = dist.local_grid(xbasis)
u['g'] = np.exp(-((x-5)/1)**2)
# u['g'] = np.cos(2*2*np.pi*x/Lx)

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

snapshots = solver.evaluator.add_file_handler('diffusion_snapshots', sim_dt=10*timestep)
snapshots.add_task(u, name='velocity')


# Main loop
u.change_scales(1)
u_list = [np.copy(u['g'])]
t_list = [solver.sim_time]
while solver.proceed:
    solver.step(timestep)
    if solver.iteration % 500 == 0:
        logger.info(f"Iteration={solver.iteration}, Time={solver.sim_time:0.2f}/{stop_sim_time}, dt={timestep}")
    if solver.iteration % 25 == 0:
        u.change_scales(1)
        u_list.append(np.copy(u['g']))
        t_list.append(solver.sim_time)

plt.figure(figsize=(16,6))
plt.pcolormesh(x.ravel(), np.array(t_list), np.array(u_list))
plt.xlim(0, Lx)
plt.ylim(0, stop_sim_time)
plt.savefig('1d_diffusion_example.png', dpi=300)

