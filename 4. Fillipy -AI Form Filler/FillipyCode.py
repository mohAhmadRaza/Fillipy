import streamlit as st
import base64
from groq import Groq
from PIL import Image
import os
import json
from io import BytesIO
from TTS.api import TTS
from pydub import AudioSegment
from pydub.playback import play


# Creating GROQ's client from API
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
client = Groq(api_key=GROQ_API_KEY)


# App layout and title
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FEEE91;  /* Change this to your desired color */
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("PDF Filler 🦙")
st.markdown("### AI-Powered Form Filler using Llama 3.2 11b and Llama 3-8b Models 🧑‍💻✨")

st.header("📑 Processed USCIS 📑")
st.markdown("#### The AI will extract to help you and fill out the form accurately from the USCIS Form.")

image_directory = "Images"
image_files = os.listdir(image_directory)
print(image_files)

image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(image_files)

page_responses = []

def convert_image_to_data_url(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # Adjust format if image is not PNG
    base64_encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_encoded}"


if not image_files:
    st.write("No image files found in the directory. Please check the Images folder.")
else:
    page_responses = []
    count = 1
    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        image = Image.open(image_path) 

        # Get the base64 data URL
        data_url = convert_image_to_data_url(image)
        
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are an assistant analyzing a USCIS form..."},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }, {"role": "assistant", "content": ""}],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        assistant_message = completion.choices[0].message.content
        page_responses.append(f"Page : {count}\n{assistant_message}")
        count += 1


# Combine all page responses into one string
combined_response = "\n".join(page_responses)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message((message['role'])):
        st.markdown(message['content'])
    
user_prompt = st.chat_input("As for Llama 3.2 ...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})


    final_completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{
            "role": "user",
            "content": f'''
                        We are giving you a combined string from all pages of a USCIS form.
                        Please keep in check where the page start and ends, user can query like "What is in first page" ?
                        Please reformat and structure it into a well-organized list with proper formatting.

                        The combined response is:
                        "{combined_response}"

                        Store all the fields, like Full name, First Name, Last Name, Middle name, Family name, Account Number, etc., on separate lines.
                        Now, user will ask you about guidance to fill this form. He will also ask you queries related to this form, you have to answer from {combined_response}.
                        Provide a clean, readable, and structured output that would be easy for the user to follow and fill out the form correctly.
                        '''
        },
        *st.session_state.chat_history
        ]
    )

    assisstant_response = final_completion.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": assisstant_response})

    with st.chat_message("assisstant"):
        st.markdown(assisstant_response)

    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
    tts.tts_to_file(text=assisstant_response, file_path="output.wav")
    audio = AudioSegment.from_wav("output.wav")
    play(audio)
