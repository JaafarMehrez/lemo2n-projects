import glob
import os

# Mapping from Moltemplate ID -> Element Symbol
# Based on your system.data analysis
type_to_element = {
    100: "C", 
    101: "H",
    120: "O",
    121: "H",
    177: "N",
    181: "H",
    178: "C",
    179: "O",
    180: "C",
    95:  "H",
}

def convert_file(filename):
    output_filename = "fixed_" + filename
    print(f"Converting {filename} -> {output_filename} ...")
    
    with open(filename, 'r') as fin, open(output_filename, 'w') as fout:
        in_atoms_section = False
        
        for line in fin:
            if line.startswith("ITEM: ATOMS"):
                in_atoms_section = True
                # Help ASE by renaming 'type' to 'element' in the header
                # Original: ITEM: ATOMS id type xu ...
                # New:      ITEM: ATOMS id element xu ...
                fout.write(line.replace("type", "element"))
                continue
            
            if line.startswith("ITEM:") and in_atoms_section:
                in_atoms_section = False
            
            if in_atoms_section:
                parts = line.split()
                # Structure: id type xu yu zu vx vy vz
                # Type is at index 1
                try:
                    type_id = int(parts[1])
                    if type_id in type_to_element:
                        parts[1] = type_to_element[type_id]
                except (ValueError, IndexError):
                    pass # Line might be malformed or already string
                
                fout.write(" ".join(parts) + "\n")
            else:
                fout.write(line)

if __name__ == "__main__":
    files = glob.glob("production_run_*.lammpstrj")
    if not files:
        print("No production_run_*.lammpstrj files found!")
    
    for f in files:
        convert_file(f)
    print("Done! Use the 'fixed_' files for preprocessing.")
