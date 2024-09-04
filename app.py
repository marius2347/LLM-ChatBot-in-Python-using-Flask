# LLM ChatBot developed by Marius

from flask import Flask, render_template, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch 

app = Flask(__name__)

# pre-trained models for movie classifying sentiment and tokenizers
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# models for summarization and translation
summarizer = pipeline("summarization")
translator_en_to_fr = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
translator_fr_to_en = pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en")
translator_en_to_es = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
translator_es_to_en = pipeline("translation_es_to_en", model="Helsinki-NLP/opus-mt-es-en")
translator_en_to_de = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
translator_de_to_en = pipeline("translation_de_to_en", model="Helsinki-NLP/opus-mt-de-en")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ""
    task = ""
    language = ""
    if request.method == 'POST':
        task = request.form.get('task')
        text = request.form.get('text', '')

        # movie review sentiment task
        if task == "review":
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            result = "The movie was good!" if predicted_class in [4, 5] else "Bad movie..Sorry!"
        
        # summarization task
        elif task == "summarization":
            summary = summarizer(text, max_length=500, min_length=50, do_sample=False)
            result = summary[0]['summary_text']

        # translation task
        elif task == "translation":
            language = request.form.get('language', '')
            if language == "en_to_fr":
                translation = translator_en_to_fr(text)
            elif language == "fr_to_en":
                translation = translator_fr_to_en(text)
            elif language == "en_to_es":
                translation = translator_en_to_es(text)
            elif language == "es_to_en":
                translation = translator_es_to_en(text)
            elif language == "en_to_de":
                translation = translator_en_to_de(text)
            elif language == "de_to_en":
                translation = translator_de_to_en(text)
            result = translation[0]['translation_text']

        return render_template('index.html', text=text, result=result, task=task, language=language)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
