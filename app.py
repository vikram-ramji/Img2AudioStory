import os
import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from hugchat import hugchat
from hugchat.login import Login
from transformers import pipeline

load_dotenv(find_dotenv())

HUGGINGFACEHUB_API_TOKEN = os.environ['HUGGINGFACEHUB_API_TOKEN']


def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]['generated_text']
    return text


def generate_story(scenario):
    template = f"""
    You are a short storyteller;
    You can generate a story based on a simple narrative, the story should be no more than 60 words;

    CONTEXT: {scenario}
    STORY:
    """
    # Create a ChatBot
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())  # or cookie_path="usercookies/<email>.json"
    story = chatbot.chat(template)

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


# Log in to huggingface and grant authorization to huggingchat
sign = Login("vikramramji24@gmail.com", "jV9wy4FVKm4iAxG")
cookies = sign.login()

# Save cookies to usercookies/<email>.json
sign.saveCookies()


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
