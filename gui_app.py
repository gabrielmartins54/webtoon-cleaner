"""
WebtoonCleaner — Desktop GUI App
Envia imagens para o backend GPU (Modal) e baixa o resultado limpo.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import requests

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL = "https://abraaor047--webtoon-cleaner-fastapi-app.modal.run"
APP_VERSION  = "1.0.0"
APP_TITLE    = "Webtoon Cleaner AI"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG      = "#0d0d14"
C_SURFACE = "#13131e"
C_CARD    = "#18182a"
C_BORDER  = "#2a2a42"
C_ACCENT  = "#7c6fff"
C_ACCENT2 = "#4f8aff"
C_GREEN   = "#22d3a0"
C_RED     = "#ff5b6e"
C_YELLOW  = "#fbbf24"
C_TEXT    = "#e8e8f4"
C_MUTED   = "#888899"


# ─────────────────────────────────────────────────────────────────────────────
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("560x780")
        self.minsize(520, 720)
        self.configure(fg_color=C_BG)
        self.resizable(True, True)

        # Center on screen
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        x = (sw - 560) // 2
        y = (sh - 780) // 2
        self.geometry(f"560x780+{x}+{y}")

        self._selected_file: Path | None = None
        self._processing = False

        self._build_ui()

    # ── UI Build ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Scrollable main container
        self.main = ctk.CTkScrollableFrame(self, fg_color=C_BG, scrollbar_button_color=C_BORDER)
        self.main.pack(fill="both", expand=True, padx=0, pady=0)

        inner = self.main
        pad = {"padx": 28}

        # ── Header ──
        header = ctk.CTkFrame(inner, fg_color="transparent")
        header.pack(fill="x", pady=(28, 4), **pad)

        # GPU badge
        badge = ctk.CTkFrame(header, fg_color="#1a1a2e", corner_radius=100,
                              border_width=1, border_color="#3a3a60")
        badge.pack(anchor="center", pady=(0, 12))
        ctk.CTkLabel(badge, text="⚡  GPU Online · Modal Cloud",
                     font=ctk.CTkFont("Segoe UI", 11, "bold"),
                     text_color=C_ACCENT).pack(padx=14, pady=5)

        ctk.CTkLabel(inner, text=APP_TITLE,
                     font=ctk.CTkFont("Segoe UI", 30, "bold"),
                     text_color=C_TEXT).pack(**pad)
        ctk.CTkLabel(inner, text="Remoção automática de texto de manhwa/webtoon com IA",
                     font=ctk.CTkFont("Segoe UI", 12),
                     text_color=C_MUTED).pack(pady=(2, 20), **pad)

        # ── File picker card ──
        file_card = ctk.CTkFrame(inner, fg_color=C_CARD, corner_radius=16,
                                  border_width=1, border_color=C_BORDER)
        file_card.pack(fill="x", pady=(0, 12), **pad)

        ctk.CTkLabel(file_card, text="📦  Arquivo ZIP / RAR",
                     font=ctk.CTkFont("Segoe UI", 13, "bold"),
                     text_color=C_TEXT).pack(anchor="w", padx=18, pady=(16, 4))
        ctk.CTkLabel(file_card, text="Selecione um arquivo .zip ou .rar com as imagens do webtoon",
                     font=ctk.CTkFont("Segoe UI", 11),
                     text_color=C_MUTED).pack(anchor="w", padx=18, pady=(0, 10))

        btn_row = ctk.CTkFrame(file_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=18, pady=(0, 16))

        ctk.CTkButton(btn_row, text="Selecionar Arquivo",
                      font=ctk.CTkFont("Segoe UI", 12, "bold"),
                      fg_color=C_ACCENT, hover_color="#6659ee",
                      corner_radius=10, height=38,
                      command=self._pick_file).pack(side="left")

        self.file_label = ctk.CTkLabel(btn_row, text="Nenhum arquivo selecionado",
                                        font=ctk.CTkFont("Segoe UI", 11),
                                        text_color=C_MUTED, wraplength=280, anchor="w")
        self.file_label.pack(side="left", padx=12, fill="x", expand=True)

        # ── Settings card ──
        settings_card = ctk.CTkFrame(inner, fg_color=C_CARD, corner_radius=16,
                                      border_width=1, border_color=C_BORDER)
        settings_card.pack(fill="x", pady=(0, 12), **pad)

        ctk.CTkLabel(settings_card, text="⚙  Configurações",
                     font=ctk.CTkFont("Segoe UI", 13, "bold"),
                     text_color=C_TEXT).pack(anchor="w", padx=18, pady=(16, 12))

        # Mask engine
        self._add_label(settings_card, "Mask Engine (Detector de Texto)")
        self.mask_var = ctk.StringVar(value="onnx_textmask")
        self.mask_menu = ctk.CTkOptionMenu(
            settings_card,
            values=["onnx_textmask", "easyocr", "rtdetr"],
            variable=self.mask_var,
            fg_color=C_SURFACE, button_color=C_ACCENT, button_hover_color="#6659ee",
            dropdown_fg_color=C_CARD, dropdown_hover_color=C_BORDER,
            font=ctk.CTkFont("Segoe UI", 12),
            corner_radius=10, height=36,
        )
        self.mask_menu.pack(fill="x", padx=18, pady=(0, 12))

        # Inpaint model
        self._add_label(settings_card, "Modelo de Inpainting")
        self.model_var = ctk.StringVar(value="default")
        self.model_options = {
            "LaMa — padrão (mais rápido)": "default",
            "SDXL Fast Inpaint — Turbo": "sdxl_fast_inpaint",
            "SD1.5 Webtoon Inpaint — balanceado": "sd15_webtoon_inpaint",
            "FLUX Fill Inpaint — melhor qualidade": "flux_fill_inpaint",
        }
        self.model_display_var = ctk.StringVar(value="LaMa — padrão (mais rápido)")
        self.model_menu = ctk.CTkOptionMenu(
            settings_card,
            values=list(self.model_options.keys()),
            variable=self.model_display_var,
            fg_color=C_SURFACE, button_color=C_ACCENT, button_hover_color="#6659ee",
            dropdown_fg_color=C_CARD, dropdown_hover_color=C_BORDER,
            font=ctk.CTkFont("Segoe UI", 12),
            corner_radius=10, height=36,
            command=self._on_model_change,
        )
        self.model_menu.pack(fill="x", padx=18, pady=(0, 12))

        # Strength slider
        self._add_label(settings_card, "Força do Inpainting (0.20 – 0.55)")
        strength_row = ctk.CTkFrame(settings_card, fg_color="transparent")
        strength_row.pack(fill="x", padx=18, pady=(0, 12))

        self.strength_var = ctk.DoubleVar(value=0.35)
        self.strength_slider = ctk.CTkSlider(
            strength_row, from_=0.20, to=0.55, variable=self.strength_var,
            progress_color=C_ACCENT, button_color=C_ACCENT, button_hover_color="#6659ee",
            fg_color=C_BORDER, number_of_steps=35,
            command=lambda v: self.strength_val_label.configure(text=f"{v:.2f}"),
        )
        self.strength_slider.pack(side="left", fill="x", expand=True)
        self.strength_val_label = ctk.CTkLabel(strength_row, text="0.35",
                                                font=ctk.CTkFont("Segoe UI", 12, "bold"),
                                                text_color=C_ACCENT, width=40)
        self.strength_val_label.pack(side="left", padx=(10, 0))
        self.strength_slider.configure(state="disabled")

        # LoRA path
        self._add_label(settings_card, "LoRA Path (opcional, apenas SD1.5)")
        self.lora_entry = ctk.CTkEntry(
            settings_card, placeholder_text="C:\\models\\webtoon_lora.safetensors",
            fg_color=C_SURFACE, border_color=C_BORDER,
            font=ctk.CTkFont("Segoe UI", 11),
            corner_radius=10, height=36,
        )
        self.lora_entry.pack(fill="x", padx=18, pady=(0, 16))
        self.lora_entry.configure(state="disabled")

        # ── Process button ──
        self.btn_process = ctk.CTkButton(
            inner,
            text="⚡  Processar Imagens na GPU",
            font=ctk.CTkFont("Segoe UI", 14, "bold"),
            fg_color=C_ACCENT, hover_color="#6659ee",
            corner_radius=14, height=52,
            command=self._start_processing,
        )
        self.btn_process.pack(fill="x", pady=(4, 0), **pad)

        # ── Status card ──
        self.status_card = ctk.CTkFrame(inner, fg_color=C_CARD, corner_radius=16,
                                         border_width=1, border_color=C_BORDER)
        self.status_card.pack(fill="x", pady=(12, 4), **pad)
        self.status_card.pack_forget()  # hidden initially

        self.status_title = ctk.CTkLabel(self.status_card, text="",
                                          font=ctk.CTkFont("Segoe UI", 12, "bold"),
                                          text_color=C_TEXT)
        self.status_title.pack(anchor="w", padx=18, pady=(14, 2))

        self.status_label = ctk.CTkLabel(self.status_card, text="",
                                          font=ctk.CTkFont("Segoe UI", 11),
                                          text_color=C_MUTED, wraplength=460, anchor="w",
                                          justify="left")
        self.status_label.pack(anchor="w", padx=18, pady=(0, 6))

        self.progress_bar = ctk.CTkProgressBar(self.status_card,
                                                progress_color=C_ACCENT,
                                                fg_color=C_BORDER, corner_radius=4, height=4)
        self.progress_bar.pack(fill="x", padx=18, pady=(0, 14))
        self.progress_bar.set(0)

        # ── Footer ──
        footer = ctk.CTkFrame(inner, fg_color="transparent")
        footer.pack(pady=(16, 28), **pad)

        ctk.CTkLabel(footer, text=f"Webtoon Cleaner AI v{APP_VERSION}  ·  ",
                     font=ctk.CTkFont("Segoe UI", 10),
                     text_color=C_MUTED).pack(side="left")

        link = ctk.CTkLabel(footer, text="Modal Dashboard",
                             font=ctk.CTkFont("Segoe UI", 10),
                             text_color=C_ACCENT, cursor="hand2")
        link.pack(side="left")
        link.bind("<Button-1>", lambda e: webbrowser.open(
            "https://modal.com/apps/abraaor047/main/deployed/webtoon-cleaner"))

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _add_label(self, parent, text: str):
        ctk.CTkLabel(parent, text=text,
                     font=ctk.CTkFont("Segoe UI", 11, "bold"),
                     text_color=C_MUTED).pack(anchor="w", padx=18, pady=(0, 4))

    def _on_model_change(self, display_name: str):
        self.model_var.set(self.model_options[display_name])
        is_diff = self.model_var.get() in ("sdxl_fast_inpaint", "flux_fill_inpaint", "sd15_webtoon_inpaint")
        is_sd15 = self.model_var.get() == "sd15_webtoon_inpaint"
        self.strength_slider.configure(state="normal" if is_diff else "disabled")
        self.lora_entry.configure(state="normal" if is_sd15 else "disabled")

    def _pick_file(self):
        path = filedialog.askopenfilename(
            title="Selecione o arquivo ZIP/RAR",
            filetypes=[("ZIP/RAR", "*.zip *.rar"), ("Todos", "*.*")],
        )
        if path:
            self._selected_file = Path(path)
            name = self._selected_file.name
            size_mb = self._selected_file.stat().st_size / 1024 / 1024
            self.file_label.configure(
                text=f"📄 {name}  ({size_mb:.1f} MB)",
                text_color=C_ACCENT,
            )

    # ── Processing ────────────────────────────────────────────────────────────
    def _start_processing(self):
        if self._processing:
            return
        if not self._selected_file:
            messagebox.showwarning("Arquivo necessário", "Selecione um arquivo .zip ou .rar primeiro.")
            return
        if not self._selected_file.exists():
            messagebox.showerror("Arquivo não encontrado", f"O arquivo não existe:\n{self._selected_file}")
            return

        self._processing = True
        self.btn_process.configure(state="disabled", text="⏳  Processando...")
        self._show_status("Enviando para a GPU...", C_YELLOW,
                          "Aguarde — cold start pode demorar ~30s na primeira execução.", 0.0, animate=True)

        threading.Thread(target=self._process_thread, daemon=True).start()

    def _process_thread(self):
        try:
            inpaint_model  = self.model_var.get()
            mask_engine    = self.mask_var.get()
            strength       = self.strength_var.get()
            lora_path      = self.lora_entry.get().strip()

            is_sdxl = inpaint_model == "sdxl_fast_inpaint"
            is_flux  = inpaint_model == "flux_fill_inpaint"
            is_sd15  = inpaint_model == "sd15_webtoon_inpaint"
            is_diff  = is_sdxl or is_flux or is_sd15

            redraw_engine   = "flux" if is_flux else "sd15" if is_sd15 else "sdxl"
            redraw_steps    = 3 if is_sdxl else 20 if is_flux else 14 if is_sd15 else 16
            redraw_guidance = 0.0 if is_sdxl else 20.0 if is_flux else 6.0 if is_sd15 else 6.5

            with open(self._selected_file, "rb") as f:
                files = {"file": (self._selected_file.name, f, "application/zip")}
                data  = {
                    "mask_engine":         mask_engine,
                    "inpaint_model":       inpaint_model,
                    "high_quality_redraw": str(is_diff).lower(),
                    "redraw_engine":       redraw_engine,
                    "redraw_denoise":      str(strength if is_diff else 0.35),
                    "redraw_steps":        str(redraw_steps),
                    "redraw_guidance":     str(redraw_guidance),
                }
                if lora_path:
                    data["redraw_lora_path"] = lora_path

                self._update_status_safe("Processando na GPU...", C_YELLOW,
                                         "Imagens sendo limpas na nuvem...", 0.4)

                resp = requests.post(
                    f"{BACKEND_URL}/upload",
                    files=files,
                    data=data,
                    timeout=3600,
                )

            if not resp.ok:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

            result = resp.json()
            download_url = result.get("download_url", "")
            if download_url and not download_url.startswith("http"):
                download_url = BACKEND_URL + download_url

            if download_url:
                self._update_status_safe("Baixando resultado...", C_YELLOW,
                                         "Transferindo arquivo limpo...", 0.85)
                dl_resp = requests.get(download_url, timeout=300, stream=True)
                dl_resp.raise_for_status()

                save_path = self._selected_file.parent / (
                    result.get("filename") or f"{self._selected_file.stem}_clean.zip"
                )
                with open(save_path, "wb") as out:
                    for chunk in dl_resp.iter_content(chunk_size=1024 * 1024):
                        out.write(chunk)

                self._update_status_safe(
                    "✅  Processamento concluído!",
                    C_GREEN,
                    f"Arquivo salvo em:\n{save_path}\n\n"
                    f"mask={result.get('mask_engine', '?')} · "
                    f"model={result.get('inpaint_model', '?')}",
                    1.0,
                )
                os.startfile(str(save_path.parent))
            else:
                self._update_status_safe("✅  Concluído!", C_GREEN,
                                         f"{result.get('message', 'OK')}", 1.0)

        except Exception as exc:
            self._update_status_safe("❌  Erro", C_RED, str(exc), 0.0)
        finally:
            self.after(0, self._reset_button)

    def _reset_button(self):
        self._processing = False
        self.btn_process.configure(state="normal", text="⚡  Processar Imagens na GPU")
        self._anim_running = False

    # ── Status helpers ────────────────────────────────────────────────────────
    _anim_running = False

    def _show_status(self, title: str, color: str, detail: str, progress: float, animate=False):
        self.status_card.pack(fill="x", pady=(12, 4), padx=28)
        self.status_title.configure(text=title, text_color=color)
        self.status_label.configure(text=detail)
        self.progress_bar.configure(progress_color=color)
        self.progress_bar.set(progress)
        if animate:
            self._anim_running = True
            self._animate_progress()

    def _animate_progress(self):
        if not self._anim_running:
            return
        cur = self.progress_bar.get()
        nxt = min(cur + 0.008, 0.88)
        self.progress_bar.set(nxt)
        self.after(200, self._animate_progress)

    def _update_status_safe(self, title: str, color: str, detail: str, progress: float):
        self.after(0, lambda: self._show_status(title, color, detail, progress, animate=False))


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
