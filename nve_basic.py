import numpy as np
import gsd.hoomd
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

# Initial parameters
n_particle = 256  # number of particle
temp = 1.8  # temperature in reduced units
box = 5.3
epsilon = 1  # LJ epsilon
sigma = 1  # LJ sigma
dt = 0.005  # time step for integration
t_total = 100  # total time
nsteps = np.rint(t_total / dt).astype(np.int32)
delta_r = 0.01

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

pair_dist = []


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
velocities = np.random.normal(0, np.sqrt(temp), (n_particle, 3))
velocities -= np.mean(velocities, axis=0)


def get_temp():
    return np.sum(velocities**2) / (3 * n_particle - 3)


print("TEMPERATURE:", get_temp())


# You can get energy and the pressure for free out of this calculation if you do it right


def force_calc(box, radii, sigma, epsilon):
    # Evaluate forces
    # Using LJ potential
    # u_lj(r_ij) = 4*epsilon*[(sigma/r_ij)^12-(sigma/r_ij)^6]s
    # Calculate pair distances
    dr = (radii[:, np.newaxis, :] - radii[np.newaxis, :, :]) * box
    pair_dist.append(np.sqrt(np.sum(dr**2, axis=-1)))
    # # minumum image convention
    dr = dr - box * np.rint(dr / box)

    # Calculate squared distances
    dr_sq = np.sum(dr**2, axis=-1)

    # return np.zeros((n_particle, 3)), 0, 0

    # Mask the diagonal to avoid division by zero (self-interaction)
    mask = np.eye(n_particle, dtype=bool)
    dr_sq[mask] = 1.0

    # Calculate the magnitudes of forces using the Lennard-Jones potential
    inv_dr6 = (sigma**2 / dr_sq) ** 3
    w_vec = 48 * epsilon * (inv_dr6**2 - 0.5 * inv_dr6)
    # Set the diagonal back to zero
    w_vec[mask] = 0.0
    f_mag = w_vec / dr_sq

    forces = np.sum(f_mag[:, :, np.newaxis] * dr, axis=1)

    # Calculate the potential energy
    potential_energy = np.sum(4 * epsilon * (inv_dr6**2 - inv_dr6))
    # We set the diagonal of the potential energy to zero as we do not want to include self-interactions
    potential_energy -= n_particle * 4 * epsilon * (sigma**12 - sigma**6)
    # Calculate the kinetic energy
    kinetic_energy = 0.5 * np.sum(np.linalg.norm(velocities, axis=1) ** 2)
    total_energy = potential_energy + kinetic_energy

    internal_virial = -1 / 6 * np.sum(f_mag)
    pressure = internal_virial + get_temp() * n_particle / box**3

    return (forces, total_energy, pressure)


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
# f = open("log.txt", "a+")
# print("step", file=f)


def calc_properties():
    # Calculate properties of interest in this function
    nice_thermo_dynamic_property = 0
    return nice_thermo_dynamic_property


thermo = calc_properties()
# print("%d" % (0), file=f)

# NVE integration
# Equilibration
glob_step = 0
total_energies = []
all_pressures = []
for step in tqdm(range(nsteps)):
    glob_step = step
    # Velocity Verlet algorithm
    velocities = velocities + 0.5 * dt * forces
    radii = radii + dt * velocities / box

    # Applying PBC
    radii = np.where(np.abs(radii) > 0.5, (radii + 0.5) % 1 - 0.5, radii)

    forces, total_energy, pressure = force_calc(box, radii, sigma, epsilon)
    velocities = velocities + 0.5 * dt * forces

    # dump frame
    total_energies.append(total_energy)
    all_pressures.append(pressure)

    # pair distribution function

    if step % 1000 == 0 and step > 10000:
        distances = np.array(pair_dist).flatten()
        distances = distances[distances != 0]

        density = n_particle / box**3
        # Calculate the histogram and divide by the number of samples to get probability density
        hist, bins = np.histogram(
            distances, bins=np.arange(0, box / 2, delta_r), density=True
        )

        # Calculate the bin centers
        bin_centers = (bins[1:] + bins[:-1]) / 2

        # Calculate the volume of the spherical shells
        shell_volume = 4 * np.pi / 3 * ((bin_centers + delta_r) ** 3 - bin_centers**3)

        # Calculate the ideal gas density for each bin
        n_ideal_gas = density * shell_volume

        # Calculate the radial distribution function
        g_r = hist / n_ideal_gas

        # Plot the radial distribution function
        plt.plot(bin_centers + 0.5 * delta_r, g_r)
        plt.xlabel("r")
        plt.ylabel("g(r)")
        # Save the figure to a file. Update the filename with each iteration to create a series of images
        plt.savefig("plot.png")

        # Clear the figure so the next plot doesn't overlap with this one
        plt.clf()
        pair_dist = []

    if step % 10 == 0:
        pair_dist = []

        # end stuff here

    if step % (n_dump // 2) == 0:
        print("TEMP:", get_temp())
        t.append(create_frame(radii, velocities, sigma, box, step / n_dump))

# Things left to do
# g_r variables
# way to compute g_r
# I recommend analyzing diffusion coefficient from the trajectories you dump


def plot_quantity(vals, label):
    # Create an array or list for the step numbers
    steps = range(len(vals))

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, vals, label=label)

    # Add labels and title
    plt.xlabel("Steps")
    plt.ylabel("label")
    plt.title("Total Energy per Step")

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()


plot_quantity(total_energies, "Total Energy")
plot_quantity(all_pressures, "Pressure")

f.close()
