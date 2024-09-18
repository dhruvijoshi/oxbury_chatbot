from flask import Flask, render_template, request, jsonify
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('main.html')

@app.route("/generate", methods=["GET", "POST"])
def generate():
    msg = request.form["msg"]
    input = msg
    return generate_text(input)

def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt ,return_tensors="pt")
    output = model.generate(input_ids, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], max_length=max_length ,skip_special_tokens=True)
    
    if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
    # Truncate to the last complete sentence
    sentences = generated_text.split('.')
    if len(sentences) > 1:
        generated_text = '.'.join(sentences[:-1]) + '.'
        
    return generated_text.strip()


if __name__ == '__main__':
    app.run()