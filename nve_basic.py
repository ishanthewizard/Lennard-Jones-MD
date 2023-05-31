import numpy as np
import gsd.hoomd
import math
from tqdm import tqdm

# Initial parameters
n_particle = 10  # number of particle
temp = 1.0  # temperature in reduced units
box = 5.3
epsilon = 1.0  # LJ epsilon
sigma = 1.0  # LJ sigma
dt = 0.005  # time step for integration
t_total = 100  # total time
nsteps = np.rint(t_total / dt).astype(np.int32)

# I recommend adding a part that reads in parameters from a file to change things as needed

print("Input parameters")
print("Number of particles %d" % n_particle)
print("Initial temperature %8.8e" % temp)
print("Box size %8.8e" % box)
print("epsilon %8.8e" % epsilon)
print("sigma %8.8e" % sigma)
print("dt %8.8e" % dt)
print("Total time %8.8e" % t_total)
print("Number of steps %d" % nsteps)

# Constant box properties
vol = box**3.0
rho = n_particle / vol

radii = np.zeros((n_particle, 3), dtype=np.float64)
velocities = np.zeros((n_particle, 3), dtype=np.float64)
forces = np.zeros((n_particle, 3), dtype=np.float64)
# note: generally easy to keep units for radius and velocities in box=1 units
# and specifically -0.5 to 0.5 space


# Initialize configuration
# Radii
# FCC lattice
def fcc_positions(n_particle, box):
    from itertools import product

    # round-up to nearest fcc box
    cells = np.ceil((n_particle / 4.0) ** (1.0 / 3.0)).astype(np.int32)
    cell_size = box / cells
    radius_ = np.empty((n_particle, 3))
    r_fcc = np.array(
        [
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
        ],
        dtype=np.float64,
    )
    i = 0
    for ix, iy, iz in product(
        list(range(cells)), repeat=3
    ):  # triple loop over unit cells
        for a in range(4):  # 4 atoms in a unit cell
            radius_[i, :] = r_fcc[a, :] + np.array([ix, iy, iz]).astype(
                np.float64
            )  # 0..nc space
            radius_[i, :] = radius_[i, :] * cell_size / box  # 0..1
            i = i + 1
            if i == n_particle:  # break when we have n_particle in our box
                return radius_


radii = fcc_positions(n_particle, box)
radii = radii - 0.5  # convert to -0.5 to 0.5 space for ease with PBC

# Procedure to initialize velocities


# You can get energy and the pressure for free out of this calculation if you do it right
def force_calc(box, radii, sigma, epsilon):
    # Evaluate forces
    # Using LJ potential
    # u_lj(r_ij) = 4*epsilon*[(sigma/r_ij)^12-(sigma/r_ij)^6]

    # Calculate pair distances
    dr = (radii[:, np.newaxis, :] - radii[np.newaxis, :, :]) * box

    # Calculate squared distances
    dr_sq = np.sum(dr**2, axis=-1)

    # Mask the diagonal to avoid division by zero (self-interaction)
    mask = np.eye(n_particle, dtype=bool)
    dr_sq[mask] = 1.0

    # Calculate the magnitudes of forces using the Lennard-Jones potential
    inv_dr6 = (sigma**2 / dr_sq) ** 3
    f_mag = (48 * epsilon / dr_sq) * (inv_dr6**2 - 0.5 * inv_dr6)
    # Set the diagonal back to zero
    f_mag[mask] = 0.0

    forces = np.sum(f_mag[:, :, np.newaxis] * dr, axis=1)

    return forces


# Function to dump simulation frame that is readable in Ovito
# Also stores radii and velocities in a compressed format which is nice
def create_frame(radii, velocities, sigma, box, frame):
    # Particle positions, velocities, diameter

    radii = box * radii
    partpos = radii.tolist()
    velocities = velocities.tolist()
    diameter = sigma * np.ones((n_particle,))
    diameter = diameter.tolist()

    # Now make gsd file
    s = gsd.hoomd.Frame()
    s.configuration.step = frame
    s.particles.N = n_particle
    s.particles.position = partpos
    s.particles.velocity = velocities
    s.particles.diameter = diameter
    s.configuration.box = [box, box, box, 0, 0, 0]
    return s


t = gsd.hoomd.open(name="test.gsd", mode="w")
n_dump = 100  # dump for configuration

# Open file to dump log file
f = open("log.txt", "a+")
print("step", file=f)


def calc_properties():
    # Calculate properties of interest in this function
    nice_thermo_dynamic_property = 0
    return nice_thermo_dynamic_property


thermo = calc_properties()
print("%d" % (0), file=f)

# NVE integration
# Equilibration
for step in tqdm(range(nsteps)):
    # Velocity Verlet algorithm
    velocities = velocities + 0.5 * dt * forces
    radii = radii + dt * velocities / box
    # Applying PBC needed here
    forces = force_calc(box, radii, sigma, epsilon)
    velocities = velocities + 0.5 * dt * forces

    # dump frame
    if step % n_dump == 0:
        t.append(create_frame(radii, velocities, sigma, box, step / n_dump))

# Things left to do
# g_r variables
# way to compute g_r
# I recommend analyzing diffusion coefficient from the trajectories you dump

f.close()
