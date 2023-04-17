"""
При вызове функции в консоли при незагруженных моделях
падает с ошибкой из-за некрректного ключа шифрования SHA256.
Протестировано на модели large-v2  
"""


import sys
from whisper import _download, _MODELS

models = ["tiny", "base", "small", "medium", "large"]

for model in models:
    print(f'Downloading model {model}....')
    _download(_MODELS[model], "~/.cache/whisper", False)
