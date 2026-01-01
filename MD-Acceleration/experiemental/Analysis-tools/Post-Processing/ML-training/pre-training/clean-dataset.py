import os
import glob

def clean_extxyz_text(file_path):
    print(f"Processing: {file_path}")

    # Create a temporary output file
    temp_file = file_path + ".tmp"

    with open(file_path, 'r') as f_in, open(temp_file, 'w') as f_out:
        for line in f_in:
            # 1. Handle the Header Line
            if "Properties=" in line:
                # Remove the type definition from the header string
                # We handle both cases where it might be in the middle or end
                line = line.replace(":type:I:1", "")
                line = line.replace("type:I:1:", "")
                f_out.write(line)
                continue

            # 2. Handle Atom Data Lines
            parts = line.split()

            # Heuristic: Atom lines contain data and are not the "atom count" line.
            # Based on your file, atom lines have > 10 columns.
            # The 'type' is the 5th column (index 4) -> Species, X, Y, Z, Type, Vx...
            if len(parts) > 10:
                # Remove the 5th column (index 4)
                # Before: [O, x, y, z, 2, vx, ...]
                # After:  [O, x, y, z, vx, ...]
                parts.pop(4)

                # Reconstruct the line ensuring standard spacing
                new_line = " ".join(parts) + "\n"
                f_out.write(new_line)
            else:
                # This catches the "Number of Atoms" line (usually just "472")
                f_out.write(line)

    # Replace original file with the cleaned temporary file
    os.replace(temp_file, file_path)
    print(f" -> Cleaned {os.path.basename(file_path)}")

if __name__ == "__main__":
    # Adjust path if your folder name is different
    dataset_dir = "data"

    # Find all .extxyz files
    files = glob.glob(os.path.join(dataset_dir, "*.extxyz"))

    print(f"Found {len(files)} files to clean.")
    for f in files:
        clean_extxyz_text(f)
    print("All done.")
