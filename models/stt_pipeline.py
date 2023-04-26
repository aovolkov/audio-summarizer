import whisper
import os
from tqdm.notebook import tqdm
from whisper.utils import get_writer


def whisper_stt(path_to_data: str,
                path_to_save: str,
                model_name: str = 'small',
                output_formats: list = ['txt', ]
                ) -> None:
    """_summary_

    Args:
        path_to_data (str): a path to raw audio samples
        path_to_save (str): a path to a
        directory for results. (Must exist).
        model_name (str, optional): whisper model name
        (supported: tiny, base, small, medium, large).
        Defaults to 'small'.
        output_formats: (list, optional): format of
        otput text file. Supported: 'txt', 'srt', 'vtt',
        'tsv', 'json'.
        Defaults to ['txt',].

    """
    model = whisper.load_model(model_name)

    for audio in tqdm(os.listdir(path_to_data)):
        audio_file = os.path.join(path_to_data, audio)
        transcription = model.transcribe(audio_file, language='ru')
        for out_format in output_formats:
            whisper_writer = get_writer(out_format, path_to_save)
            whisper_writer(transcription, audio_file)
        print(f'{audio} transcribed')
