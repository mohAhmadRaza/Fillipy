import streamlit as st
import base64
from groq import Groq
from PIL import Image
import os
import json
from io import BytesIO
from dotenv import load_dotenv

client = Groq(api_key="GROQ_API_KEY")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFCCEA;  /* Change this to your desired color */
    }
    .stSelectbox label {
        font-size: 20px;
        color: black;  /* Label text color */
    }
    .stSelectbox {
        background-color: rgba(0, 0, 0, 0.4); /* Semi-transparent background */
        border-radius: 10px;
        padding: 10px;
    }
    body {
        background: #FFF7D1; /* App background color */
        color: black;
        font-family: 'Arial', sans-serif;
    }
    /* Title Styling */
    .stTitle {
        font-size: 2.5rem;
        text-align: center;
        color: #10375C;
        text-shadow: 2px 2px 4px #000000;
    }
    .assistant-message {
        background-color: #000B58; /* Response background */
        padding: 10px;
        color: white; /* Response text color */
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="stTitle">Fillipy 🦙</h1>', unsafe_allow_html=True)
st.markdown("### AI-Powered Form Filling Helper using Llama 3.2 11b and Llama 3-8b Models 🧑‍💻✨")

options = ["", "USCIS-Form-I-9", "USCIS-Form-I-765"]
selected_option = st.selectbox("Choose Form To Proceed:", options)

if selected_option != "":
    image_directory = selected_option
    image_files = os.listdir(image_directory)
    print(image_files)

    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(image_files)

    page_responses = []

    def convert_image_to_data_url(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG") 
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
                model="llama-3.2-90b-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "You are an assistant analyzing a USCIS Form. You have to extrcat al the information from the given form page. Extract all the info wether it's important to enter, must be enter, all the filds required, all the information required from the user to fill the form (wether optional or necessary), also mention the page number present at the last of page, all the instructions, warnings, information. Means everything. Because your response will be used to help user to fill the form."},
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

    st.markdown(f"## Ready To Help You With {selected_option}")

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
            st.markdown(f'<div class="assistant-message">{assisstant_response}</div>', unsafe_allow_html=True)
