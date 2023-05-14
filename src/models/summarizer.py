import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
from summarizer import Summarizer
from transformers import T5ForConditionalGeneration, T5Tokenizer


class ExtrSummarizer:
    def __init__(
            self,
            extr_summ_model,
    ):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
            )
        self.extr_config = AutoConfig.from_pretrained(extr_summ_model)
        self.extr_config.output_hidden_states = True
        self.extr_tokenizer = AutoTokenizer.from_pretrained(
            extr_summ_model, config=self.extr_config
            )
        self.extr_model = AutoModel.from_pretrained(
            extr_summ_model,
            config=self.extr_config
        )
        self.extr_summarizer = Summarizer(
            custom_model=self.extr_model, custom_tokenizer=self.extr_tokenizer
            )

        def extractive_summarize(self, text):
            return self.extr_summarizer(text, min_length=60)


class AbsSummarizer:
    def __init__(
            self,
            abs_summ_model,
    ):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
            )
        self.abs_tokenizer = T5Tokenizer.from_pretrained(abs_summ_model)
        self.abs_model = T5ForConditionalGeneration.from_pretrained(
            abs_summ_model
            )
        self.abs_model.to(self.device)

    def abstractive_summarize(self, text):
        input_ids = self.abs_tokenizer.encode(
            text, return_tensors='pt', padding=True,
            )
        summary_ids = self.abs_model.generate(input_ids)
        return self.abs_tokenizer.decode(
            summary_ids[0], skip_special_tokens=True
            )
