from flask import Flask, render_template, request, jsonify
import torch
import requests
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import spacy
import re
import os

api_key = os.getenv('COMPANY_HOUSE_API')

# Initialising Language model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

retrieved_data = {}
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('main.html')

@app.route("/generate", methods=["POST"])
def generate():
    # Fetching inputs from user
    msg = request.form["msg"]
    nlp = spacy.load("en_core_web_sm")
    msg = nlp(msg)
   
    # Regex to find postcode in input
    postcode_pattern = r'\b[A-Z]{1,2}[0-9][A-Z0-9]? [0-9][ABD-HJLNP-UW-Z]{2}\b'
    c_postcode = re.findall(postcode_pattern, msg.text)
    
    if c_postcode:
        response = call_using_add(c_postcode)
        return response
    
    # Finding company name/ number from input 
    for token in msg:
        if (token.pos_ == 'PROPN' and token.is_alpha == True) or (token.pos_ == 'PROPN' and token.ent_type_ == 'ORG'):
            response = call_using_name(token.text)
            return response

        if token.pos_ == 'NUM' and token.ent_type_ == 'CARDINAL' and len(token.text) == 8:
            response = call_using_number(token.text)
            return response
    
    response = generate_text(msg.text)
    return response


# Generates output for the chatbot
def generate_text(prompt, max_length=100):
    result = []
    
    # If company data is retrieved from API
    if retrieved_data:
        generated_text = f"I found below result:"
        result.append(generated_text)
        for value in retrieved_data.values():
            company_name = value.get('name', 'N/A')
            company_number = value.get('number', 'N/A')
            company_address = value.get('address', 'N/A')

            generated_text = (
                f"<br>Company: {company_name}<br>"
                f"Number: {company_number}<br>"
                f"Location: {company_address}<br>"
            )
            result.append(generated_text)
        retrieved_data.clear()
        return ''.join(result)
    
    # Generates output for queries other than Company details
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        
        output = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, max_length=max_length, num_return_sequences=1)
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        sentences = generated_text.split('.')
        return jsonify(sentences)


# Will be called if there is company name in the input
def call_using_name(name):
    # Fetching data from API
    url = f'https://api.company-information.service.gov.uk/search/companies?q={name}'
    response = requests.get(url, auth=(api_key, ''))
    data = response.json()    
    
    if response.status_code != 200 or 'items' not in data:
        answer = generate_text(data)
        return answer

    # Extracting useful information from API response
    i = 0
    for item in data['items']:
        i += 1
        company_number = item['company_number']
        company_name = item['title']
        address = item.get('address', {})
        premises = address.get('premises', '')
        address_line_1 = address.get('address_line_1', '')
        locality = address.get('locality', '')
        postal_code = address.get('postal_code', '')
        
        full_address = f"{premises} {address_line_1}, {locality}, {postal_code}".strip()
        
        retrieved_data[i] = {'name': company_name, 'number': company_number, 'address': full_address}
    
    answer = generate_text(retrieved_data)
    return answer


# Will be called if there is company number in the input
def call_using_number(c_num):
    # Fetching data from API
    url = f'https://api.company-information.service.gov.uk/search?q={c_num}'
    response = requests.get(url, auth=(api_key, ''))
    data = response.json()
    
    if response.status_code != 200:
        answer = generate_text(data)
        return answer
    
    # Extracting useful information from API response
    i = 0
    for item in data['items']:
        i += 1
        company_number = item['company_number']
        company_name = item['title']
        address = item.get('address', {})
        premises = address.get('premises', '')
        address_line_1 = address.get('address_line_1', '')
        locality = address.get('locality', '')
        postal_code = address.get('postal_code', '')
        
        full_address = f"{premises} {address_line_1}, {locality}, {postal_code}".strip()
        retrieved_data[i] = {'name' : company_name, 'number' : company_number, 'address' : full_address}
    
    answer = generate_text(retrieved_data)
    return answer


# Will be called if there is company's postcode in the input
def call_using_add(c_add):
    # Fetching data from API
    url = f'https://api.company-information.service.gov.uk/search?q={c_add}'
    response = requests.get(url, auth=(api_key, ''))
    data = response.json()

    if response.status_code != 200:
        answer = generate_text(data)
        return answer
    
    # Extracting useful information from API response
    i = 0
    for item in data['items']:
        if item['kind'] == 'searchresults#company':
            i += 1
            company_name = item['title']
            address = item.get('address', {})
            premises = address.get('premises', '')
            address_line_1 = address.get('address_line_1', '')
            locality = address.get('locality', '')
            postal_code = address.get('postal_code', '')
            company_number = item['company_number']

            full_address = f"{premises} {address_line_1}, {locality}, {postal_code}".strip()
            retrieved_data[i] = {'name' : company_name, 'number' : company_number, 'address' : full_address}

    answer = generate_text(retrieved_data)
    return answer


if __name__ == '__main__':
    app.run()
