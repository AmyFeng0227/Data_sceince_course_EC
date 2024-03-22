import openai
import streamlit as st
import base64
import requests
import os


st.title("Handwritten digit prediction with OpenAI API")
openai_api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=openai_api_key)
picture = st.camera_input("Take a picture of a hand-written digit")

if picture is not None:
    picture_bytes = picture.getvalue()
    picture_b64 = base64.b64encode(picture_bytes).decode("utf-8")

    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
      "model": "gpt-4-vision-preview",  
      "messages": [
        {
          "role": "user",
          "content": [
            {"type": "text",
             "text": "What numbers are in this image? answer in the format of 'X', where X is an arabic number."},
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{picture_b64}"
              }
            }
          ]
        }
      ],
      "max_tokens": 150
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        content_value = response.json()["choices"][0]["message"]["content"]
        digits = content_value.split()
        if len(digits) > 1:
            st.write("The digits are", content_value)
        else:
            st.write("The digit is", content_value)
    else:
        st.error("Failed to analyze the image.")