import os

# === CONFIGURATION ===
LABELS_DIR = r"C:\Users\João Matias\Desktop\val\labels"

# Mapping: old class ID -> new class ID
CLASS_MAPPING = {
    0: 0,  # Car
    1: 1,  # Bus
    3: 2,  # Van
    4: 3   # Other
}

# Process each label file
total_files = 0
remapped_files = 0
for filename in os.listdir(LABELS_DIR):
    if not filename.endswith('.txt'):
        continue

    total_files += 1
    filepath = os.path.join(LABELS_DIR, filename)
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    modified = False
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        old_class = int(parts[0])
        if old_class not in CLASS_MAPPING:
            print(f"Warning: file {filename} has unknown class {old_class}, skipping this line")
            continue
        new_class = CLASS_MAPPING[old_class]
        if new_class != old_class:
            modified = True
        new_lines.append(' '.join([str(new_class)] + parts[1:]))
    
    # Rewrite the file with remapped class IDs
    if modified:
        with open(filepath, 'w') as f:
            f.write('\n'.join(new_lines) + '\n')
        remapped_files += 1

print(f"Processed {total_files} label files.")
print(f"Remapped classes in {remapped_files} files.")