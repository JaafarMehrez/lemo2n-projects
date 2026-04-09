from pylab import *
from ase.build import bulk
from ase.io import write

a = 2.951
c = 4.679
Ti_UC = bulk("Ti", "hcp", a=a, c=c)

write("model.xyz", Ti_UC)

kpoints = ["GKMGA"]
with open("kpoints.in", "w") as f:
    for kp in kpoints:
        path = Ti_UC.cell.bandpath(path=kp, npoints=0)
        for label in kp:
            k = path.special_points[label]
            f.write(f"{k[0]:.3f} {k[1]:.3f} {k[2]:.3f} {label}\n")
        f.write("\n")
