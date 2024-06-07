import numpy as np
import dedalus.public as d3

import sys
sys.path.insert(0, '/home/disk/eos7/sressel/research/thesis-work/python/auxiliary_functions/')
import ipynb.fs.full.mjo_mean_state_diagnostics as mjo

import logging
logger = logging.getLogger(__name__)

# Parameters
GRAVITY = 9.81                           # g [m/s^2]
EQUIVALENT_DEPTH = 250.                  # H [m]
CORIOLIS_PARAMETER = 2.29e-11            # ß [m^-1 s^-1]
EARTH_RADIUS = 6371.0072e3               # R_e [m]
AIR_DENSITY = 1.225                      # ρ_a [kg m^-3]
WATER_DENSITY = 997                      # ρ_w [kg m^-3]
LATENT_HEAT = 2260000                    # L_v [J kg^-1 K^-1]
SPECIFIC_HEAT = 1004                     # c_p [J kg^-1]
SECONDS_PER_DAY = 24*3600
GROSS_DRY_STABILITY = 3.12e4 

Lx = 2*np.pi*EARTH_RADIUS
Ly = 7000e3
Nx = 256
Ny = 128
gravity_wave_phase_speed = np.sqrt(GRAVITY*EQUIVALENT_DEPTH)
length_scale = (gravity_wave_phase_speed/CORIOLIS_PARAMETER)**(1/2)

dealias = 3/2
stop_sim_time = 10*SECONDS_PER_DAY
timestepper = d3.RK222
timestep = 500
dtype = np.float64

total_iterations = stop_sim_time//timestep

CFL = gravity_wave_phase_speed*timestep/(Lx/Nx)
print(f"CFL = {CFL:0.3f}")

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.Chebyshev(coords['y'], size=Ny, bounds=(-Ly, Ly), dealias=dealias)
x, y = dist.local_grids(xbasis, ybasis)

# Fields
u = dist.Field(name='u', bases=(xbasis, ybasis))
T = dist.Field(name='T', bases=(xbasis, ybasis))
tau_u1 = dist.Field(name='tau_u1', bases=xbasis)
tau_T1 = dist.Field(name='tau_T1', bases=xbasis)
tau_u2 = dist.Field(name='tau_u2', bases=xbasis)
tau_T2 = dist.Field(name='tau_T2', bases=xbasis)

# ncc_y = dist.Field(bases=ybasis)
# ncc_y['g'] = y


# Substitutions
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])

# Problem
problem = d3.IVP([u, T, tau_u1, tau_u2, tau_T1, tau_T2], namespace=locals())

# Tau polynomials
tau_basis = ybasis.derivative_basis(1)
p_u1 = dist.Field(bases=tau_basis)
p_T1 = dist.Field(bases=tau_basis)
p_u2 = dist.Field(bases=tau_basis)
p_T2 = dist.Field(bases=tau_basis)

p_u1['c'][-1] = 1
p_T1['c'][-2] = 2
p_u2['c'][-3] = 3
p_T2['c'][-4] = 4

problem.add_equation("dt(u) + (gravity_wave_phase_speed**2/GROSS_DRY_STABILITY)*dx(T) + tau_u1*p_u1 + tau_u2*p_u2 = 0")
# problem.add_equation("CORIOLIS_PARAMETER*ncc_y*u + (gravity_wave_phase_speed**2/GROSS_DRY_STABILITY)*dy(T)= 0 ")
problem.add_equation("dt(T) + (GROSS_DRY_STABILITY)*dx(u) + tau_T1*p_T1 + tau_T2*p_T2 = 0")
problem.add_equation("u(y=0) = 0")
problem.add_equation("u(y=Ly) = 0")
problem.add_equation("T(y=0) = 0")
problem.add_equation("T(y=Ly) = 0")

# Initial Conditions
n_wavelengths = 2
initial_wavenumber = 2*np.pi*n_wavelengths/Lx

u['g'] = gravity_wave_phase_speed*np.real(
    mjo.parabolic_cylinder_function(y/length_scale, 0)
    * np.exp(1j*initial_wavenumber*x)
)

# <T>(x,y,t=0)
T['g'] = (GROSS_DRY_STABILITY/gravity_wave_phase_speed**2)*gravity_wave_phase_speed**2*np.real(
    mjo.parabolic_cylinder_function(y/length_scale, 0)
    * np.exp(1j*initial_wavenumber*x)
)
    
# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.5*SECONDS_PER_DAY)
snapshots.add_task(u, name='zonal_velocity')
snapshots.add_task(T, name='column_temperature')

# Main loop
try: 
    logger.info("Starting main loop")
    while solver.proceed:
        solver.step(timestep)
        if solver.iteration % 100 == 0:
            logger.info(
                f"Iteration={solver.iteration}/{total_iterations}, "
                + f"Time={solver.sim_time/SECONDS_PER_DAY:0.2f}/{stop_sim_time/SECONDS_PER_DAY} days, "
                + f"dt={timestep} sec"
            )
except:
    logger.error('Exception raised, triggering end of main loop')
    raise
finally:
    solver.log_stats()
