import os
from collections import Counter

# Folder containing label files
LABEL_FOLDER = r"C:\val\labels"

# Counter for class occurrences
class_counts = Counter()

# Loop through all label files
for filename in os.listdir(LABEL_FOLDER):
    if filename.endswith(".txt"):
        filepath = os.path.join(LABEL_FOLDER, filename)

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue

                class_id = line.split()[0]  # first value = class
                class_counts[class_id] += 1

# =====================
# PRINT RESULTS
# =====================
print("\nClass distribution:\n")

total = sum(class_counts.values())

for cls in sorted(class_counts.keys(), key=int):
    count = class_counts[cls]
    pct = (count / total) * 100 if total > 0 else 0
    print(f"Class {cls}: {count} objects ({pct:.2f}%)")

print("\nTotal objects:", total)