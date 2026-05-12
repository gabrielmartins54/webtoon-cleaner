# Webtoon Cleaner (PT-BR)

Pipeline para limpeza de texto em páginas de webtoon/manhwa com:
- detecção de regiões de texto,
- geração de máscara,
- inpainting base (LaMa/ONNX),
- refinamento opcional por modelos generativos (SD 1.5, SDXL Turbo, FLUX),
- exportação final em PSD/PSB + máscara PNG.

O frontend é feito em Next.js e o backend em FastAPI.

## O Que Este Projeto Faz

- Recebe um `.zip`/`.rar` com imagens (`.jpg`, `.jpeg`, `.png`).
- Processa cada página em tiles verticais (ideal para tiras longas).
- Remove texto preservando estilo visual da arte.
- Gera:
  - arquivo PSD/PSB com camadas `Original` e `Cleaned`,
  - máscara consolidada (`*_mask.png`),
  - `.zip` final para download.

## Arquitetura

- `backend/main.py`: backend principal (uso local e proxy pelo frontend).
- `backend/main_local.py`: variante com upload opcional para Google Drive.
- `backend/advanced_redrawer.py`: engines de refinamento generativo.
- `backend/text_mask_detector.py`: detector ONNX de máscara de texto.
- `backend/onnx_lama_inpainter.py`: runtime ONNX para LaMa.
- `backend/modal_backend.py`: deploy serverless no Modal.
- `frontend/`: interface web + proxy `/api/upload` e `/api/download`.

## Requisitos

- Windows com GPU NVIDIA e CUDA funcional.
- Python 3.11.
- Node.js 20+.
- VRAM recomendada:
  - mínimo: 8 GB (LaMa/ONNX + EasyOCR/ONNX mask).
  - recomendado: 16 GB+ para refinamento SD/FLUX.

Importante:
- O backend está configurado para exigir CUDA em runtime.
- Sem CUDA válida, o processamento será interrompido com erro explícito.

## Instalação

## 1) Backend

Na raiz do projeto:

```powershell
python -m venv backend/.venv
backend/.venv/Scripts/activate
pip install -r requirements.txt
```

Como `requirements.txt` da raiz referencia o backend, isso instala:
- FastAPI,
- PyTorch + torchvision + torchaudio,
- diffusers/transformers,
- onnxruntime-gpu,
- EasyOCR,
- Ultralytics/SAM2,
- demais dependências do pipeline.

## 2) Frontend

```powershell
cd frontend
npm install
```

Opcional com pnpm:

```powershell
pnpm install
```

## Execução Local

## 1) Subir backend

Em um terminal:

```powershell
cd backend
../backend/.venv/Scripts/python.exe main.py
```

Backend padrão: `http://127.0.0.1:8000`.

## 2) Subir frontend

Em outro terminal:

```powershell
cd frontend
npm run dev
```

Frontend padrão: `http://127.0.0.1:3001`.

## Como Usar (Fluxo)

1. Abra o frontend.
2. Envie um `.zip`/`.rar` com as páginas.
3. Escolha `Mask Engine`.
4. Escolha `Inpaint Model`.
5. Opcional: informe `Webtoon LoRA Path`.
6. Clique em `Process Images`.
7. O frontend baixa automaticamente o `.zip` final.

## Engines de Máscara

`onnx_textmask`:
- detector ONNX + refinamento SAM2 (quando disponível),
- melhor equilíbrio de precisão/velocidade.

`easyocr`:
- usa EasyOCR com `ko_webtoon_v2` para localizar texto,
- pode refinar com SAM2,
- útil quando o detector ONNX ou RT-DETR não capturam bem certos estilos.

`rtdetr`:
- detector por objeto de texto,
- rápido, mas tende a ser menos preciso em efeitos de glow/blur.

## Modelos de Inpainting (Dropdown)

`Default LaMa (simple-lama)`:
- base robusta e rápida para remoção de texto.

`MangaCleaner lama.onnx (local path)`:
- usa runtime ONNX para inpainting LaMa custom.

`SDXL Fast Inpaint (Turbo-style, fastest)`:
- preset de refinamento com SDXL Turbo.

`SD1.5 Webtoon Inpaint (fast practical)`:
- preset com `runwayml/stable-diffusion-inpainting`,
- melhor custo/benefício para consistência de webtoon,
- aceita LoRA opcional via `Webtoon LoRA Path`.

`FLUX Fill Inpaint (best quality)`:
- melhor coerência global em muitos casos,
- mais pesado e mais lento.

## Variáveis de Ambiente Importantes

Backend:
- `APP_DATA_DIR`: diretório base de `uploads/processed`.
- `FRONTEND_ORIGINS`: CORS permitido.
- `MASK_ENGINE_DEFAULT`: engine padrão de máscara.
- `INPAINT_MODEL_DEFAULT`: modelo padrão do dropdown.
- `REDRAW_STEPS_DEFAULT`: steps padrão.
- `TEXT_MASK_MODEL_PATH`: caminho do modelo ONNX de máscara.
- `RTDETR_MODEL_ID`: modelo RT-DETR principal (default `ko_webtoon_v2`).
- `RTDETR_FALLBACK_MODEL_ID`: fallback RT-DETR.
- `EASYOCR_RECOG_NETWORK`: default `ko_webtoon_v2`.
- `EASYOCR_LANG_LIST`: default `ko`.
- `EASYOCR_USER_NETWORK_DIR`: pasta com rede custom do EasyOCR.

Frontend:
- `NEXT_PUBLIC_API_BASE_URL`: URL base do backend.
- `API_BASE_URL`: usado no proxy server-side.

## Deploy no Modal

Arquivo de entrada:
- `backend/modal_backend.py`

Pré-requisitos:
- conta no Modal,
- secret com `HF_TOKEN` (para modelos gated no Hugging Face, quando necessário),
- volume para cache/dados.

Exemplo de deploy:

```powershell
cd backend
modal deploy modal_backend.py
```

Configurações relevantes no deploy:
- `MODAL_DATA_VOLUME_NAME`
- `MODAL_HF_SECRET_NAME`
- `MODAL_GPU`

## Estrutura de Entrada e Saída

Entrada:
- `.zip`/`.rar` com imagens.

Saída por página:
- `nome.psd` ou `nome.psb` (quando dimensão excede limite de PSD),
- `nome_mask.png`.

Saída final:
- `job_clean.zip`.

## Troubleshooting

`No images were processed`:
- verifique se os arquivos no zip são `.jpg/.jpeg/.png`,
- confirme se o backend encontrou imagens no diretório extraído.

`CUDA is required but unavailable`:
- verifique driver NVIDIA e instalação do PyTorch com CUDA,
- confirme `torch.cuda.is_available()` retornando `True`.

Download inicial muito lento:
- primeira execução baixa pesos de modelos (SD/FLUX) do Hugging Face,
- chamadas seguintes ficam mais rápidas com cache.

Máscara pega regiões fora do texto:
- troque entre `onnx_textmask`, `easyocr` e `rtdetr`,
- teste `easyocr` com `ko_webtoon_v2`,
- ajuste modelo de inpaint (SD1.5 costuma ser mais estável para webtoon).

## Segurança e Observações

- Não versionar credenciais reais (`credentials.json`, tokens, chaves).
- Modelos gated no Hugging Face exigem acesso e token autorizado.
- Para uso comercial, revise licenças dos modelos escolhidos.

## Licenças de Terceiros

Este projeto integra bibliotecas e modelos de terceiros (PyTorch, diffusers, EasyOCR, Ultralytics, Hugging Face, etc.).  
Consulte as licenças de cada dependência e modelo antes de distribuição.
