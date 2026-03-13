import os
import shutil

# === CONFIGURAÇÃO DE CAMINHOS ===
SOURCE_ROOT = r"C:\images"
DEST_FOLDER = r"C:\Users\João Matias\Desktop\test_images"

# Cria a pasta de destino se ela não existir, seguindo a prática dos seus scripts anteriores
os.makedirs(DEST_FOLDER, exist_ok=True)

print("🚀 Iniciando a cópia e renomeação das imagens...")

# 1. Percorrer cada pasta MVI (ex: MVI_39031)
for mvi_folder in os.listdir(SOURCE_ROOT):
    folder_path = os.path.join(SOURCE_ROOT, mvi_folder)
    
    # Garantir que estamos processando apenas diretórios
    if os.path.isdir(folder_path):
        print(f"📁 Processando pasta: {mvi_folder}")
        
        # 2. Percorrer cada imagem dentro da pasta MVI
        for img_name in os.listdir(folder_path):
            if img_name.lower().startswith("img") and img_name.lower().endswith(".jpg"):
                
                # Extrair o número do frame (ex: img00001.jpg -> 00001)
                frame_number = img_name.replace("img", "").replace(".jpg", "").replace(".JPG", "")
                
                # 3. Criar o novo nome seguindo o padrão dos seus labels
                # Exemplo final: MVI_39031_frame00001.jpg
                new_filename = f"{mvi_folder}_frame{frame_number}.jpg"
                
                src_path = os.path.join(folder_path, img_name)
                dst_path = os.path.join(DEST_FOLDER, new_filename)
                
                # 4. Copiar o arquivo (shutil.copy2 mantém os metadados originais)
                shutil.copy2(src_path, dst_path)

print(f"\n✅ Concluído! Todas as imagens foram consolidadas em: {DEST_FOLDER}")