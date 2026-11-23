"""Fetch checkpoints by path or URL. Hugging Face URLs go through huggingface_hub, which caches them and uses your
login (`hf auth login`) for gated or private repositories; other URLs go through torch.hub's cache."""
import os
import re
from pathlib import Path

HF_URL = re.compile(r"https://huggingface\.co/(?P<repo>[^/]+/[^/]+)/resolve/(?P<rev>[^/]+)/(?P<file>.+)")


def local_path(path_or_url):
    m = HF_URL.fullmatch(str(path_or_url).split("?")[0])
    if m:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(m["repo"], m["file"], revision=m["rev"])
    if str(path_or_url).startswith(("http://", "https://")):
        import torch
        target = Path(torch.hub.get_dir()) / "checkpoints" / os.path.basename(str(path_or_url).split("?")[0])
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(str(path_or_url), str(target))
        return str(target)
    return str(path_or_url)
