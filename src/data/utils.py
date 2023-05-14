import re
import os
import librosa
import soundfile as sf
import torch
from pydub import AudioSegment

DS_KEYWORD_DICT = {
    "Python": [
        "Бетон", "Петунии", "Бетоне", "Вильтон", "Бетона", "Бетоном",
        "Питоне", "Питон", "Питона", "Степаном", "Капитан"
    ],
    "R": ["Аре", "Ара", "Шаром", "Арии", "Аркс"],
    "SpeechKit": ["печь Кит"],
    "Science": ["Сaйнс", "Сансон", "Санси", "Сайенс", "Сандерса"],
    "Scientist": ["Сантис"],
    "Data": ["Дата", "Дельта", "Дейта", "Дуайта"],
    "Glowbyte": ["Голубая", "Брайтон", "Глобал"],
    "ML": ["Мель", "Эмаль", "Мелли", "Малькам"],
    "Machine": ["Машин"],
    "Learning": ["Ленинг", "Лёнинг", "Дарлинг", "Дарлинга"]
}
SR = 16000


def ogg2wav(path_to_ogg_file: str, path_to_wav_file: str):
    wav, sr = librosa.load(path_to_ogg_file)
    sf.write(path_to_wav_file, wav, SR, 'PCM_16')


class AudioWav:
    def __init__(self, path_to_wav_file: str):
        self.path_to_wav_file = path_to_wav_file
        self.wav_file = AudioSegment.from_wav(self.path_to_wav_file)
        self.sample_rate = self.wav_file.frame_rate

    @staticmethod
    def init_vad(repo: str, model_vad_path: str):
        vad_model, utils = torch.hub.load(
            repo_or_dir=repo,
            model=model_vad_path,
            force_reload=True,
            verbose=False)

        (get_speech_timestamps,
         save_audio,
         read_audio,
         VADIterator,
         collect_chunks) = utils

        return vad_model, get_speech_timestamps, read_audio

    def cut_wav_file(
            self,
            wav_chunks_dir: str
    ):
        vad_model, get_speech_timestamps, read_audio = AudioWav.init_vad(
            'snakers4/silero-vad', 'silero_vad'
            )
        wav = read_audio(
            self.path_to_wav_file, sampling_rate=self.sample_rate
            )
        speech_timestamps = get_speech_timestamps(
            wav, vad_model, sampling_rate=self.sample_rate
            )
        os.makedirs(wav_chunks_dir, exist_ok=True)

        current_frame, j = 0, 0
        for i, chunk in enumerate(speech_timestamps):
            if chunk["end"] // 600000 != j or i == len(speech_timestamps) - 1:
                speech_end = chunk["end"] // 10
                wav_chunk = self.wav_file[current_frame:speech_end]
                if i != len(speech_timestamps) - 1:
                    j = chunk["end"] // 600000
                    current_frame = speech_timestamps[i + 1]["start"] // 10
                wav_chunk.export(
                    os.path.join(wav_chunks_dir, f"chunk_{j}.wav"),
                    format="wav"
                )


class Text:
    def __init__(
            self,
            path_to_txt: str,
            keywords_dict: dict = {},
    ):
        with open(path_to_txt, "r", encoding='utf-8') as file_txt:
            self.text = file_txt.read()

        self.keywords_dict = keywords_dict

    def extend_keyword_dict(self):
        for key, item in self.keywords_dict.items():
            self.keywords_dict[key] += [i.lower() for i in item]

    def replace_keywords(self):
        self.extend_keyword_dict()
        for old_word, new_words in self.keywords_dict.items():
            for new_word in new_words:
                self.text = self.text.replace(new_word, old_word)

    def text2chunks(self, chunk_size: int):
        sentences, text_chunks = [], []
        split_regex = re.compile(r'[.]')
        for line in self.text.strip().splitlines():
            sentences.extend(
                list(
                    filter(
                        lambda t: t, [
                            t.strip() for t in split_regex.split(line)
                            ]
                    )
                )
            )

        tmp = sentences[0]
        for i, sent in enumerate(sentences[1:]):
            if len(tmp.split()) <= chunk_size:
                tmp = ". ".join([tmp, sent])
            elif (len(tmp.split()) > chunk_size) or (i == len(sentences)):
                text_chunks.append(tmp + ".")
                tmp = sent
        return text_chunks

