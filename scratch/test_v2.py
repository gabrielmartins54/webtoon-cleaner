import easyocr
import cv2
import numpy as np
import os
import sys

# Ensure utf-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Change directory to the EasyOCR folder to use its images
os.chdir(r"C:\Users\dev\easyOCR\EasyOCR")

image_path = '07.jpg'
output_path = 'resultado_webtoon_v2.jpg'

if not os.path.exists(image_path):
    print(f"ERRO: Imagem {image_path} não encontrada!")
    exit()

print("Carregando leitor customizado (ko_webtoon_v2)...")
try:
    # We use the global ~/.EasyOCR/user_network/ we set up
    reader = easyocr.Reader(['ko'], recog_network='ko_webtoon_v2')
    print("Modelo v2 carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    exit()

print(f"Processando imagem: {image_path}")
img = cv2.imread(image_path)
if img is None:
    print("ERRO ao carregar imagem com OpenCV.")
    exit()

# Executar detecção e reconhecimento
result = reader.readtext(image_path)

print("\n--- RESULTADOS v2 ---")
for (bbox, text, prob) in result:
    points = np.array(bbox).astype(np.int32)
    cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=3)
    print(f"Texto: {text} | Confiança: {prob:.4f}")

# Salvar resultados
with open('resultado_v2.txt', 'w', encoding='utf-8') as f:
    for (bbox, text, prob) in result:
        f.write(f"Texto: {text} | Confiança: {prob:.4f}\n")

cv2.imwrite(output_path, img)
print(f"\nSucesso! Texto salvo em: resultado_v2.txt")
print(f"Máscara visual salva em: {output_path}")
