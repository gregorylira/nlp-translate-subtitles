from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from rest_framework import status
import torch

class Prediction:
    def __init__(self):
        model_name = "facebook/m2m100_418M"
        device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise use CPU
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe = pipeline("translation", model=self.model, tokenizer=self.tokenizer, device=device)

    def predict(self, request):
        return_dict = dict()
        try:
            input_text = request.data["input_text"]
            src_lang = "en"  # Replace with source language code
            tgt_lang = "fr"  # Replace with target language code
            result = self.pipe(input_text, src_lang=src_lang, tgt_lang=tgt_lang)
            return_dict["prediction"] = result[0]["translation_text"]
            return_dict["status"] = status.HTTP_200_OK
        except Exception as e:
            return_dict["prediction"] = str(e)
            return_dict["status"] = status.HTTP_400_BAD_REQUEST

        return return_dict
