from pathlib import Path
import os
from whisper import _download, _MODELS


_PATH = os.path.join(Path.home(), '.cache', 'whisper')

models = [
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large"
    ]

for model in models:
    print(f'Downloading model {model}....')
    _download(_MODELS[model],
              _PATH,
              False)
