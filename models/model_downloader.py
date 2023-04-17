"""
При вызове функции в консоли при незагруженных моделях
падает с ошибкой из-за некрректного ключа шифрования SHA256.
Протестировано на модели large-v2. Корректно загрузились tiny и base.
"""


import sys
from pathlib import Path
import os
from whisper import _download, _MODELS


_PATH = os.path.join(Path.home(), '.cache', 'whisper')

models = [
    #"tiny.en",
    #"tiny",
    #"base.en",
    # "base",
    #"small.en",
    #"small",
    #"medium.en",
    "medium",
    #"large"
    ]

for model in models:
    print(f'Downloading model {model}....')
    _download(_MODELS[model],
              _PATH,
              False)
