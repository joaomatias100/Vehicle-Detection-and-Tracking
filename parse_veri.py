import xml.etree.ElementTree as ET
import os

# --- 1. SET YOUR PATHS ---
veri_base_path = r"C:\VeRi\VeRi" 
xml_file = os.path.join(veri_base_path, "train_label.xml")
images_folder = os.path.join(veri_base_path, "image_train")

def parse_veri_labels(xml_path, image_dir):
    # FIX: Read the file using 'gb2312' encoding first to avoid the ValueError
    with open(xml_path, 'r', encoding='gb2312') as f:
        xml_string = f.read()
    
    # Parse the XML from the string instead of the file path
    root = ET.fromstring(xml_string)
    dataset_mapping = []
    
    for item in root.findall('Items/Item'):
        image_name = item.get('imageName')
        full_image_path = os.path.join(image_dir, image_name)
        
        if os.path.exists(full_image_path):
            dataset_mapping.append({
                "image_path": full_image_path,
                "vehicle_id": int(item.get('vehicleID')),
                "color_id": int(item.get('colorID')),
                "type_id": int(item.get('typeID'))
            })
    
    print(f"✅ Success! Mapped {len(dataset_mapping)} images.")
    return dataset_mapping

# Run the mapping
if __name__ == "__main__":
    if os.path.exists(xml_file):
        training_data = parse_veri_labels(xml_file, images_folder)
    else:
        print(f"❌ Still can't find file at: {xml_file}")