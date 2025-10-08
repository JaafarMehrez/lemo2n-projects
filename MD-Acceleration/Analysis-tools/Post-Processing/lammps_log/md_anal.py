import MDAnalysis as mda
from MDAnalysis.analysis import rdf, msd

u = mda.Universe('silica.data', 'dump.lammpstrj')  # topology + trajectory
print("Frames:", len(u.trajectory))
# compute center-of-mass or MSD, etc.
# Example: iterate frames and compute average charge per frame (if you dumped charge as 'charge' in dump)
avg_charges = []
for ts in u.trajectory:
    charges = u.atoms.get_array('charges')  # depends on how dump labels it; may need small adapt.
    avg_charges.append(charges.mean())
# plot avg_charges vs frame with matplotlib..

