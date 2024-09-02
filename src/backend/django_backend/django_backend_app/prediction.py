from django.http import HttpResponse
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from rest_framework import status
import torch
import tqdm
import subprocess as sp
import tempfile
import json


class Prediction:
    model_name = "facebook/m2m100_418M"
    device = 0 if torch.cuda.is_available() else -1
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline("translation", model=model, tokenizer=tokenizer, device=device)

    def __init__(self):
        self.types_files = ["srt", "vtt", "cap", "scc", "ttml"]
        self.src_lang = "en"
        self.tgt_lang = "pt"

    def predict(self, request):
        try:
            file = request.FILES.get("file")
            self.src_lang = request.data.get("src_lang", "en")
            self.tgt_lang = request.data.get("tgt_lang", "pt")
            new_file = self.read_translate_srt(file)
            new_file = "\n".join(new_file)

            response = HttpResponse(new_file, content_type="text/plain; charset=utf-8")
            response["Content-Disposition"] = (
                "attachment; filename=legendas_traduzidas.srt"
            )
            return response

        except Exception as e:
            return HttpResponse(str(e), status=status.HTTP_400_BAD_REQUEST)

    def get_subtitle(self, request):
        try:
            video = request.FILES.get("video")
            video_path = tempfile.NamedTemporaryFile(delete=False)
            video_path.write(video.read())
            video_path.close()

            command = [
                "ffmpeg",
                "-i",
                video_path.name,
                "-map",
                "s:0",
                "-f",
                "srt",
                "-",
            ]
            out = sp.run(
                command, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True
            )
            subtitle = out.stdout

            response = HttpResponse(subtitle, content_type="text/plain; charset=utf-8")
            response["Content-Disposition"] = "attachment; filename=legendas.srt"

        except Exception as e:
            return HttpResponse(str(e), status=status.HTTP_400_BAD_REQUEST)

    def translate(self, input_text):
        result = self.pipe(
            input_text, src_lang=self.src_lang, tgt_lang=self.tgt_lang, batch_size=8
        )
        return [res["translation_text"] for res in result]

    def read_translate_srt(self, file):
        lines = file.readlines()
        new_lines = []
        batch = []
        for line in tqdm.tqdm(lines):
            line = line.decode("utf-8").strip()

            if line.isdigit() or "-->" in line or not line:
                if batch:
                    new_lines.extend(self.translate(batch))
                    batch = []
                new_lines.append(line)
            else:
                batch.append(line)

        if batch:
            new_lines.extend(self.translate(batch))

        return new_lines
