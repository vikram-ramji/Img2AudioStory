import os
import requests
import streamlit as st
import openai
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACE_API_KEY')


def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]['generated_text']
    print(text)
    return text


def generate_story(scenario):
    template = f"""
    You are a short storyteller;
    You can generate a story based on a simple narrative, the story should be no more than 60 words;

    CONTEXT: {scenario}
    STORY:
    """
    story = openai.Completion.create(
        model= "text-davinci-003",
        max_tokens=300,
        temperature= 1,
        prompt= template,
        stop=None
    )["choices"][0]["text"]

    print(story)
    return story

# text to speech
def text2speech(message):
    api_url = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
        "inputs": message
    }

    response = requests.post(url=api_url, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)


def main():
    st.set_page_config(page_title="img 2 audio story", page_icon=" ")

    st.header("turn img into audio story")
    uploaded_file = st.file_uploader("choose an img...", type="jpg")
    if uploaded_file is not None:
        byte = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(byte)
        st.image(uploaded_file, caption="uploaded image", use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("Senario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        st.audio("audio.flac")


if __name__ == "__main__":
    main()
