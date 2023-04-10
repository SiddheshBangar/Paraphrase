import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Define function to load models and tokenizer based on selection

def use_model(model_name):
    if model_name == 'T5':
        model = T5ForConditionalGeneration.from_pretrained('t5-base',
                                                            use_cache=True,
                                                            output_attentions=False,
                                                            max_length=2048,
                                                            num_beams=5,
                                                            no_repeat_ngram_size=1,
                                                            length_penalty=2.0,
                                                            early_stopping=True,
                                                            top_k=50,
                                                            top_p=0.95,
                                                            temperature=1.0,
                                                            do_sample=True)
        tokenizer = T5Tokenizer.from_pretrained('t5-base',
                                                do_lower_case=False,
                                                bos_token_id=0,
                                                eos_token_id=1,
                                                pad_token_id=2,
                                                model_max_length=2048)
    elif model_name == 'PAWS':
        model = AutoModelForSeq2SeqLM.from_pretrained('Vamsi/T5_Paraphrase_Paws',
                                                    use_cache=True,
                                                    output_attentions=False,
                                                    max_length=2048,
                                                    num_beams=5,
                                                    no_repeat_ngram_size=1,
                                                    length_penalty=2.0,
                                                    early_stopping=True,
                                                    top_k=50,
                                                    top_p=0.95,
                                                    temperature=1.0,
                                                    do_sample=True)
        tokenizer = AutoTokenizer.from_pretrained('Vamsi/T5_Paraphrase_Paws',
                                                do_lower_case=False,
                                                bos_token_id=0,
                                                eos_token_id=1,
                                                pad_token_id=2,
                                                model_max_length=2048)
    elif model_name == 'Parrot':
        model = AutoModelForSeq2SeqLM.from_pretrained('prithivida/parrot_paraphraser_on_T5',
                                                    use_cache=True,
                                                    output_attentions=False,
                                                    max_length=512,
                                                    num_beams=10,
                                                    no_repeat_ngram_size=2,
                                                    length_penalty=1.0,
                                                    early_stopping=True,
                                                    top_k=50,
                                                    top_p=0.95,
                                                    temperature=0.8,
                                                    do_sample=True)
        tokenizer = AutoTokenizer.from_pretrained('prithivida/parrot_paraphraser_on_T5',
                                                do_lower_case=False,
                                                bos_token_id=0,
                                                eos_token_id=1,
                                                pad_token_id=2,
                                                model_max_length=512)
    else:
        st.error("Invalid model selection.")
        return None, None
    return model, tokenizer

# Define function to paraphrase text
def paraphrase(model, tokenizer, text):
    input_ids = tokenizer.encode(text, return_tensors='pt', max_length=256, truncation=True)
    output = model.generate(input_ids=input_ids,
                            max_length=100,
                            num_beams=4,
                            early_stopping=True)
    paraphrased_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return paraphrased_text

# Define Streamlit app
def main():
    st.title("Paraphrasing App")
    st.write("Select a model to use:")
    model_name = st.selectbox("Model Selection", options=['T5', 'PAWS', 'Parrot'])
    model, tokenizer = use_model(model_name)
    if model and tokenizer:
        st.write("Enter a sentence to paraphrase:")
        input_text = st.text_area("Input Text", height=100)
        if st.button("Paraphrase"):
            output_text = paraphrase(model, tokenizer, input_text)
            st.write("Paraphrased Text:")
            st.write(output_text.capitalize())

if __name__ == '__main__':
    main()
