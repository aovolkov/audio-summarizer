import whisper
import os
from tqdm.notebook import tqdm
from whisper.utils import get_writer


def whisper_stt(path_to_data: str,
                path_to_save: str,
                model_name: str = 'small',
                ) -> None:
    """_summary_

    Args:
        path_to_data (str): a path to raw audio samples
        path_to_save (str): a path to a directory for results. Directory must exist.
        model_name (str, optional): whisper model name (available: tiny, base, small, medium, large). Defaults to 'small'.
    """
    model = whisper.load_model(model_name)

    for audio in tqdm(os.listdir(path_to_data)):
        audio_file = os.path.join(path_to_data, audio)
        transcription = model.transcribe(audio_file, language='ru')

        # Save as an TXT file
        txt_writer = get_writer("txt", path_to_save)
        txt_writer(transcription, audio_file)

        # Save as an SRT file
        srt_writer = get_writer("srt", path_to_save)
        srt_writer(transcription, audio_file)

        # Save as a VTT file
        vtt_writer = get_writer("vtt", path_to_save)
        vtt_writer(transcription, audio_file)

        # Save as a TSV file
        tsv_writer = get_writer("tsv", path_to_save)
        tsv_writer(transcription, audio_file)

        # Save as a JSON file
        json_writer = get_writer("json", path_to_save)
        json_writer(transcription, audio_file)
