'use client';
import { useState } from 'react';

export default function Home() {
  const customOnnxLamaPath = 'C:\\Users\\dev\\Downloads\\MangaCleaner_GPU\\MangaCleaner_GPU\\models\\lama.onnx';
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string>('');
  const [maskEngine, setMaskEngine] = useState<'onnx_textmask' | 'rtdetr' | 'easyocr'>('onnx_textmask');
  const [inpaintModel, setInpaintModel] = useState<string>('default');
  const [redrawDenoise, setRedrawDenoise] = useState<number>(0.35);
  const [redrawLoraPath, setRedrawLoraPath] = useState<string>('');

  const remoteApiBaseUrlRaw = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000';
  const remoteApiBaseUrl = remoteApiBaseUrlRaw.replace(/\/+$/, '');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setStatus('Select zip/rar first.');
      return;
    }

    const formData = new FormData();
    const isSdxlFast = inpaintModel === 'sdxl_fast_inpaint';
    const isFluxFill = inpaintModel === 'flux_fill_inpaint';
    const isSd15Webtoon = inpaintModel === 'sd15_webtoon_inpaint';
    const usesDiffusionInpaint = isSdxlFast || isFluxFill || isSd15Webtoon;
    const redrawEngine = isFluxFill ? 'flux' : isSd15Webtoon ? 'sd15' : 'sdxl';
    const redrawSteps = isSdxlFast ? 3 : isFluxFill ? 20 : isSd15Webtoon ? 14 : 16;
    const redrawGuidance = isSdxlFast ? 0.0 : isFluxFill ? 20.0 : isSd15Webtoon ? 6.0 : 6.5;

    formData.append('file', file);
    formData.append('mask_engine', maskEngine);
    formData.append('inpaint_model', inpaintModel);
    formData.append('high_quality_redraw', String(usesDiffusionInpaint));
    formData.append('redraw_engine', redrawEngine);
    formData.append('redraw_denoise', String(usesDiffusionInpaint ? redrawDenoise : 0.35));
    formData.append('redraw_steps', String(redrawSteps));
    formData.append('redraw_guidance', String(redrawGuidance));
    if (redrawLoraPath.trim()) {
      formData.append('redraw_lora_path', redrawLoraPath.trim());
    }

    const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

    const tryUpload = async () => {
      return fetch('/api/upload', {
        method: 'POST',
        body: formData,
        cache: 'no-store',
      });
    };

    setStatus('Uploading...');
    try {
      let res: Response;
      try {
        res = await tryUpload();
      } catch (err) {
        if (!(err instanceof Error) || (err.name !== 'AbortError' && err.name !== 'TypeError')) throw err;
        setStatus('Network interruption detected, retrying upload...');
        await delay(3000);
        res = await tryUpload();
      }

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      const data = await res.json();
      const rawDownloadPath = data.download_url ?? (data.filename ? `/download/${data.filename}` : null);
      if (rawDownloadPath) {
        const downloadUrl = new URL(rawDownloadPath, window.location.origin).toString();
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = data.filename ?? 'cleaned.zip';
        document.body.appendChild(link);
        link.click();
        link.remove();
        setStatus(`Success: ${data.message} (mask=${data.mask_engine}, redraw=${data.redraw_engine}, HQ=${data.high_quality_redraw}) Download started.`);
      } else {
        setStatus(`Success: ${data.message} (mask=${data.mask_engine}, redraw=${data.redraw_engine}, HQ=${data.high_quality_redraw})`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setStatus(`Upload failed: ${message}`);
      console.error(err);
    }
  };

  return (
    <div className="min-h-screen bg-neutral-900 text-white p-10 flex flex-col items-center">
      <h1 className="text-4xl font-bold mb-8 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
        Comic Clean AI
      </h1>

      <div className="bg-neutral-800 p-8 rounded-2xl shadow-xl w-full max-w-md border border-neutral-700">
        <p className="mb-4 text-xs text-neutral-400 break-all">
          Upload proxy: /api/upload
        </p>
        <p className="mb-4 text-xs text-neutral-500 break-all">
          Remote backend: {remoteApiBaseUrl}
        </p>
        <label className="block mb-4">
          <span className="text-sm font-semibold mb-2 block text-neutral-300">Upload Zip/Rar</span>
          <input
            type="file"
            accept=".zip,.rar"
            onChange={handleFileChange}
            className="block w-full text-sm text-neutral-400
              file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-purple-600 file:text-white
              hover:file:bg-purple-700
              cursor-pointer"
          />
        </label>

        <div className="mb-4 grid grid-cols-2 gap-3">
          <label className="text-xs text-neutral-300">
            Mask Engine
            <select
              value={maskEngine}
              onChange={(e) => setMaskEngine(e.target.value as 'onnx_textmask' | 'rtdetr' | 'easyocr')}
              className="mt-1 w-full rounded-lg border border-neutral-700 bg-neutral-900 px-2 py-2 text-sm text-neutral-100"
            >
              <option value="onnx_textmask">ONNX TextMaskDetector (recommended)</option>
              <option value="easyocr">EasyOCR (ko_webtoon_v2 + SAM2 refine)</option>
              <option value="rtdetr">RT-DETR (fast)</option>
            </select>
          </label>

          <label className="text-xs text-neutral-300 col-span-2">
            Inpaint Model
            <select
              value={inpaintModel}
              onChange={(e) => setInpaintModel(e.target.value)}
              className="mt-1 w-full rounded-lg border border-neutral-700 bg-neutral-900 px-2 py-2 text-sm text-neutral-100"
            >
              <option value="default">Default LaMa (simple-lama)</option>
              <option value={customOnnxLamaPath}>MangaCleaner `lama.onnx` (local path)</option>
              <option value="sdxl_fast_inpaint">SDXL Fast Inpaint (Turbo-style, fastest)</option>
              <option value="sd15_webtoon_inpaint">SD1.5 Webtoon Inpaint (fast practical)</option>
              <option value="flux_fill_inpaint">FLUX Fill Inpaint (best quality)</option>
            </select>
          </label>

          <label className="text-xs text-neutral-300 col-span-2">
            Webtoon LoRA Path (optional)
            <input
              type="text"
              value={redrawLoraPath}
              onChange={(e) => setRedrawLoraPath(e.target.value)}
              placeholder="C:\\models\\webtoon_lora.safetensors"
              className="mt-1 w-full rounded-lg border border-neutral-700 bg-neutral-900 px-2 py-2 text-sm text-neutral-100"
              disabled={inpaintModel !== 'sd15_webtoon_inpaint'}
            />
          </label>

          <label className="text-xs text-neutral-300">
            Inpaint Strength (0.2-0.55)
            <input
              type="number"
              min={0.2}
              max={0.55}
              step={0.01}
              value={redrawDenoise}
              onChange={(e) => setRedrawDenoise(Number(e.target.value))}
              className="mt-1 w-full rounded-lg border border-neutral-700 bg-neutral-900 px-2 py-2 text-sm text-neutral-100"
              disabled={!(inpaintModel === 'sdxl_fast_inpaint' || inpaintModel === 'flux_fill_inpaint' || inpaintModel === 'sd15_webtoon_inpaint')}
            />
          </label>
        </div>

        <button
          onClick={handleUpload}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-xl transition-all duration-200 active:scale-95"
        >
          Process Images
        </button>

        {status && (
          <div className="mt-6 p-4 bg-neutral-900 rounded-xl border border-neutral-700">
            <p className="text-sm text-center text-neutral-300">{status}</p>
          </div>
        )}
      </div>
    </div>
  );
}
