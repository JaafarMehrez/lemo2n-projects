import ase

trj = ase.io.read('dump_1.lammpstrj', index=':')
print(trj[0].arrays.keys())

