"""Stage 2 tail — merge a trained LoRA adapter into its base and register it with ollama.

Runs ON the RTX box (geofm_venv_cu124). Merges to bf16 HF safetensors, then uses ollama's
native safetensors import (no llama.cpp build needed) to create the servable model.

Usage: python merge_and_export.py --base Qwen/Qwen2.5-0.5B-Instruct \
           --adapter ~/distiller_out/0.5b --merged ~/distiller_merged/0.5b \
           --ollama-name tenet-distiller-0.5b
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELFILE = """FROM {merged}
TEMPLATE \"\"\"{{{{- if .Messages }}}}
{{{{- range $i, $_ := .Messages }}}}
{{{{- if eq .Role "system" }}}}<|im_start|>system
{{{{ .Content }}}}<|im_end|>
{{{{ else if eq .Role "user" }}}}<|im_start|>user
{{{{ .Content }}}}<|im_end|>
{{{{ else if eq .Role "assistant" }}}}<|im_start|>assistant
{{{{ .Content }}}}<|im_end|>
{{{{ end }}}}
{{{{- end }}}}<|im_start|>assistant
{{{{ else }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
{{{{ end }}}}\"\"\"
PARAMETER temperature 0
PARAMETER stop <|im_end|>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--ollama-name", required=True)
    args = ap.parse_args()

    print("loading base + adapter ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()
    Path(args.merged).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.merged, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.merged)
    print("merged ->", args.merged, flush=True)

    mf = Path(args.merged) / "Modelfile"
    mf.write_text(MODELFILE.format(merged=args.merged))
    r = subprocess.run(["ollama", "create", args.ollama_name, "-f", str(mf)],
                       capture_output=True, text=True)
    print(r.stdout[-2000:]); print(r.stderr[-2000:])
    print("OLLAMA_CREATE_RC", r.returncode, flush=True)


if __name__ == "__main__":
    main()
