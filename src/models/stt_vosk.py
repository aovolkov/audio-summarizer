import os
import wave
import json
import torch
from tqdm import tqdm
from vosk import Model, KaldiRecognizer
from tqdm.notebook import tqdm
from pydub import AudioSegment


class Vosk:
    def __init__(
            self,
            model_path: str,
            simple_rate: int,
    ):
        """
        :param model_path: Путь к модели
        :param simple_rate: Частота распознавания речи
        """
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, simple_rate)

    def stt(
            self,
            file_path: str
    ):
        """
        :param file_path: Путь к фаилу
        :return: Текст аудио
        """
        # Проверка на существование файла
        if not os.path.isfile(file_path):
            raise FileNotFoundError(os.path.basename(file_path) + " not found")

        wav = AudioSegment.from_wav(file_path)
        if len(wav) == 0:
            return " "
        wav_file = wave.open(file_path, "rb")
        file_size = os.path.getsize(file_path)

        # Проверка на правильность параметров файла
        if wav_file.getnchannels() != 1:
            raise TypeError("Audio file must be WAV format mono!")
        elif wav_file.getcomptype() != "NONE":
            raise TypeError("Compretion type is not supported!")
        elif wav_file.getsampwidth() != 2:
            raise TypeError("Sample width is different!")

        pbar = tqdm(total=file_size)
        transcription = []
        while True:
            data = wav_file.readframes(4000)
            pbar.update(len(data))
            if len(data) == 0:
                pbar.set_description("Transcription finished")
                break
            if self.recognizer.AcceptWaveform(data):
                result_text_dict = json.loads(self.recognizer.Result())
                transcription.append(result_text_dict.get("text", ""))

        final_result = json.loads(self.recognizer.FinalResult())
        transcription.append(final_result.get("text", ""))
        pbar.close()

        return ' '.join(transcription)

    @staticmethod
    def init_te_model(
            repo: str = "snakers4/silero-models",
            model: str = "silero_te"
    ):

        model, example_texts, languages, punct, apply_te = torch.hub.load(
            repo_or_dir=repo,
            model=model
        )
        return apply_te

    def apply_stt_for_chunks(
            self,
            path_to_wav_chunks: str,
            path_to_txt: str,
    ):
        apply_te = Vosk.init_te_model()
        wav_chunks = os.listdir(path_to_wav_chunks)
        for i in range(1, len(wav_chunks)):
            wav_chunk = os.path.join(path_to_wav_chunks, f"chunk_{i}.wav")
            chunk_text = self.stt(wav_chunk)
            chunk_text_corrected = apply_te(chunk_text, lan='ru')
            chunk_text_corrected = f"{chunk_text_corrected} \n"
            with open(path_to_txt, "a") as text_file:
                text_file.write(chunk_text_corrected)