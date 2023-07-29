import streamlit as st
import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import PIL
import os
from PIL import Image, ImageOps
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from oauth2client.service_account import ServiceAccountCredentials
from io import BytesIO
from googleapiclient.http import MediaIoBaseUpload
from gtts import gTTS
import tempfile


@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('model_v2b2.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()




SCOPES  = ['https://www.googleapis.com/auth/drive.file']
SERVICE_ACCOUNT_FILE  = 'file.json'

st.write("<h1 style='text-align: center; background-color: #ebccff; color: #990033;'>Tomato Leaf Diseases Detection</h1>", unsafe_allow_html=True)
st.write("<h1 style='text-align: center; background-color: #ebccff; color: #5c0099;'>টমেটো পাতার রোগ নির্ণয়</h1>", unsafe_allow_html=True)

def text_to_speech(text, lang='en'):
    # Create a temporary file to store the audio
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tts = gTTS(text=text, lang=lang)
        tts.save(tf.name)

    # Return the file path for later use
    return tf.name




file = st.file_uploader("Please upload an Tomato leaf image file. / একটি টমেটো পাতার ছবি আপলোড করুন", type=["jpg", "png", "jpeg"])


st.set_option('deprecation.showfileUploaderEncoding', False)

reference_link = "https://www.usda.gov/"
reference_text = "Tomato Diseases Solutions (English)"
reference_link_bn = "http://www.bari.gov.bd/"
reference_text_bn = "Tomato Diseases Solutions (Bangla)"


if file is None:
    st.text("No tomato leaf image is selected\nকোনো টমেটো পাতার ছবি নির্বাচন করা হয়নি")
else:

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    drive_service = build('drive', 'v3', credentials=credentials)

    # Upload the file to Google Drive
    file_metadata = {'name': file.name, 'parents': ['1ps9JTqK1N1HXVRdmQnLoeKVP1Dam4JuK']}
    media = MediaIoBaseUpload(BytesIO(file.read()), mimetype=file.type)
    response = drive_service.files().create(
        body=file_metadata, media_body=media, fields='id'
    ).execute()

    image = Image.open(file)
    st.image(image, use_column_width=True)

    image = image.resize((256,256), resample=PIL.Image.BICUBIC)
    img_arr = img_to_array(image)
    img_arr = np.expand_dims(img_arr, axis=0)
    pred = model.predict(img_arr)

    class_names = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
    # Get the index of the maximum probability
    pred_index = np.argmax(pred)

    # Map the index to the corresponding class name
    pred_class = class_names[pred_index]

    voice_button_style = """
    height: 100px;
    width: 300px;
    font-size: 50px;
"""



    st.write(f"<div style='text-align: center;'><h3>Result = {pred_class}</h3></div>", unsafe_allow_html=True)


    if pred_class == 'Tomato___Bacterial_spot':

        bac_spot_en = "Result is Tomato Bacterial spot"
        if st.button("🔊 Speak (English Result)"):
        # Convert the text to speech and play the audio
          audio_file_path_en = text_to_speech(bac_spot_en, lang='en-in')
          st.audio(audio_file_path_en, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)


        st.write("<h3 style='text-align: center;'>ফলাফল : ছবিটি হলো টমেটো ব্যাকটেরিয়াল স্পট </h3>", unsafe_allow_html=True)
        Bac_spot = "ফলাফল : ছবিটি হলো টমেটো ব্যাকটেরিয়াল স্পট"
        if st.button("🔊 Speak (Bengali Result)"):
        # Convert the text to speech and play the audio
          audio_file_path_bn = text_to_speech(Bac_spot, lang='bn')
          st.audio(audio_file_path_bn, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)


        english_text_bacterial_spot = """Solution :
1. Use copper-based fungicides for protection.
2. Practice crop rotation to disrupt bacterial life cycle.
3. Maintain garden hygiene by removing infected debris.
4. Plant resistant tomato varieties for long-term control."""

        bangla_text_bacterial_spot = """প্রতিকার :
১. প্রতিরক্ষা হিসেবে কপার-ভিত্তিক ফাংগিসাইড ব্যবহার করুন।
২. ফসল পরিবর্তন অনুশীলন করে ব্যাকটেরিয়ার জীবন চক্র বিচ্ছেদ করুন।
৩. সংক্রান্ত দ্রব্যাঙ্ক সরিয়ে নেওয়ার মাধ্যমে বাগান পরিষ্কার রক্ষণা করুন।
৪. দীর্ঘমেয়াদী নিয়ন্ত্রণের জন্য প্রতিরোধী টমেটো জাত রোপণ করুন।"""

        st.text(english_text_bacterial_spot)


        if st.button("🔊 Speak (English)"):
   
          audio_file_path_en_in = text_to_speech(english_text_bacterial_spot, lang='en-in')
          st.audio(audio_file_path_en_in, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)


        st.text(bangla_text_bacterial_spot)
        if st.button("🔊 Speak (Bengali)"):
        # Convert the text to speech and play the audio
          audio_file_path_bn = text_to_speech(bangla_text_bacterial_spot, lang='bn')
          st.audio(audio_file_path_bn, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)
        # Display the reference with the link
        st.write(f"Reference 1 : [{reference_text}]({reference_link})")
        st.write(f"Reference 2 : [{reference_text_bn}]({reference_link_bn})")

    elif pred_class == 'Tomato___Early_blight':

        early_bli_en = "Result is Tomato Early blight"
        if st.button("🔊 Speak (English Result)"):
        # Convert the text to speech and play the audio
          audio_file_path_en = text_to_speech(early_bli_en, lang='en-in')
          st.audio(audio_file_path_en, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)

        st.write("<h3 style='text-align: center;'>ফলাফল : ছবিটি হলো টমেটো আরলি ব্লাইট </h3>", unsafe_allow_html=True)
        early_bli_bn = "ফলাফল : ছবিটি হলো টমেটো আরলি ব্লাইট"
        if st.button("🔊 Speak (Bengali Result)"):
        # Convert the text to speech and play the audio
          audio_file_path_bn = text_to_speech(early_bli_bn, lang='bn')
          st.audio(audio_file_path_bn, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)
        
        english_text_early_bli = """Solution :
1. Remove infected leaves and destroy them to prevent spread.
2. Apply fungicides labeled for early blight control.
3. Practice crop rotation to reduce disease pressure.
4. Ensure proper spacing and ventilation for plants."""

        bangla_text_early_bli = """প্রতিকার :
১. সংক্রামিত পাতা অপসারণ করুন এবং বিস্তার রোধ করতে তাদের ধ্বংস করুন।
২. প্রাথমিক ব্লাইট নিয়ন্ত্রণের জন্য লেবেলযুক্ত ছত্রাকনাশক প্রয়োগ করুন।
৩. ফসল পরিবর্তন অনুশীলন করে রোগের চাপ কমানো যায়।
৪. গাছের জন্য উচিত স্থান এবং বাতাসচালনা নিশ্চিত করুন।"""

        st.text(english_text_early_bli)


        if st.button("🔊 Speak (English)"):
   
          audio_file_path_en_in = text_to_speech(english_text_early_bli, lang='en-in')
          st.audio(audio_file_path_en_in, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)


        st.text(bangla_text_early_bli)
        if st.button("🔊 Speak (Bengali)"):
        # Convert the text to speech and play the audio
          audio_file_path_bn = text_to_speech(bangla_text_early_bli, lang='bn')
          st.audio(audio_file_path_bn, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)
        # Display the reference with the link
        st.write(f"Reference 1 : [{reference_text}]({reference_link})")
        st.write(f"Reference 2 : [{reference_text_bn}]({reference_link_bn})")

        


    elif pred_class == 'Tomato___Late_blight':

        late_blt_en = "Result is Tomato Late blight"
        if st.button("🔊 Speak (English Result)"):
        # Convert the text to speech and play the audio
          audio_file_path_en = text_to_speech(late_blt_en, lang='en-in')
          st.audio(audio_file_path_en, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)

        st.write("<h3 style='text-align: center;'>ফলাফল : ছবিটি হলো টমেটো লেট ব্লাইট</h3>", unsafe_allow_html=True)
        
        late_blt_bn = "ফলাফল : ছবিটি হলো টমেটো লেট ব্লাইট"
        if st.button("🔊 Speak (Bengali Result)"):
        # Convert the text to speech and play the audio
          audio_file_path_bn = text_to_speech(late_blt_bn, lang='bn')
          st.audio(audio_file_path_bn, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)
        
        english_text_late_bli = """Solution :
1. Apply fungicides containing copper to control tomato late blight.
2. Remove infected leaves and destroy them to prevent disease spread.
3. Choose blight-resistant tomato varieties for planting.
4. Ensure proper spacing and ventilation in tomato plants to reduce humidity."""

        bangla_text_late_bli = """প্রতিকার :
১. টমেটো লেট ব্লাইট নিয়ন্ত্রণে কপার ধাতুকে যোগ করা ফংগিসাইড ব্যবহার করুন।
২. আক্রান্ত পাতা সরিয়ে ফেলে দিন এবং সেগুলি ধ্বংস করে ফেলুন রোগ প্রসারণ প্রতিরোধ করার জন্য।
৩. চয়ন করুন টমেটো সমুদ্রীক প্রকারগুলি কৃষির জন্য যা ব্লাইট রোগের প্রতিরোধ করে।
৪. টমেটো গাছের মধ্যে উপযুক্ত স্থানগুলি প্রদান করুন এবং কাফেরিয়াশীতকতা হ্রাস করার জন্য প্রায়শই বাতাস প্রবেশ করান।"""

        st.text(english_text_late_bli)


        if st.button("🔊 Speak (English)"):
   
          audio_file_path_en_in = text_to_speech(english_text_late_bli, lang='en-in')
          st.audio(audio_file_path_en_in, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)


        st.text(bangla_text_late_bli)
        if st.button("🔊 Speak (Bengali)"):
        # Convert the text to speech and play the audio
          audio_file_path_bn = text_to_speech(bangla_text_late_bli, lang='bn')
          st.audio(audio_file_path_bn, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)
        # Display the reference with the link
        st.write(f"Reference 1 : [{reference_text}]({reference_link})")
        st.write(f"Reference 2 : [{reference_text_bn}]({reference_link_bn})")
        


    elif pred_class == 'Tomato___Leaf_Mold':

        leaf_mold_en = "Result is Tomato Leaf Mold"
        if st.button("🔊 Speak (English Result)"):
        # Convert the text to speech and play the audio
          audio_file_path_en = text_to_speech(leaf_mold_en, lang='en-in')
          st.audio(audio_file_path_en, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)

        st.write("<h3 style='text-align: center;'>ফলাফল : ছবিটি হলো টমেটো লিফ মোল্ড </h3>", unsafe_allow_html=True)

        leaf_mold_bn = "ফলাফল : ছবিটি হলো টমেটো লিফ মোল্ড"
        if st.button("🔊 Speak (Bengali Result)"):
        # Convert the text to speech and play the audio
          audio_file_path_bn = text_to_speech(leaf_mold_bn, lang='bn')
          st.audio(audio_file_path_bn, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)
        
        english_text_leaf_mold = """Solution :
1. Remove infected leaves and destroy them; avoid overhead watering to control tomato leaf mold.
2. Apply fungicides with chlorothalonil as a preventive measure.
3. Ensure proper plant spacing and ventilation for reduced humidity.
4. Use resistant tomato varieties if available."""

        bangla_text_leaf_mold = """প্রতিকার :
১. সংক্রমিত পাতা অপসারণ এবং তাদের ধ্বংস; টমেটো পাতার ছাঁচ নিয়ন্ত্রণ করতে ওভারহেড জল এড়িয়ে চলুন।
২. প্রতিরোধমূলক ব্যবস্থা হিসাবে ক্লোরোথালোনিলের সাথে ছত্রাকনাশক প্রয়োগ করুন।
৩. কম আর্দ্রতার জন্য উদ্ভিদের সঠিক ব্যবধান এবং বায়ুচলাচল নিশ্চিত করুন।
৪. পাওয়া গেলে প্রতিরোধী টমেটোর জাত ব্যবহার করুন।"""

        st.text(english_text_leaf_mold)


        if st.button("🔊 Speak (English)"):
   
          audio_file_path_en_in = text_to_speech(english_text_leaf_mold, lang='en-in')
          st.audio(audio_file_path_en_in, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)


        st.text(bangla_text_leaf_mold)
        if st.button("🔊 Speak (Bengali)"):
        # Convert the text to speech and play the audio
          audio_file_path_bn = text_to_speech(bangla_text_leaf_mold, lang='bn')
          st.audio(audio_file_path_bn, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)
        # Display the reference with the link
        st.write(f"Reference 1 : [{reference_text}]({reference_link})")
        st.write(f"Reference 2 : [{reference_text_bn}]({reference_link_bn})")


    elif pred_class == 'Tomato___Septoria_leaf_spot':

        septo_en = "Result is Tomato Septoria leaf spot"
        if st.button("🔊 Speak (English Result)"):
        # Convert the text to speech and play the audio
          audio_file_path_en = text_to_speech(septo_en, lang='en-in')
          st.audio(audio_file_path_en, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)

        st.write("<h3 style='text-align: center;'>ফলাফল : ছবিটি হলো টমেটো সেপ্টোরিয়াল লিফ স্পট </h3>", unsafe_allow_html=True)

        septo_bn = "ফলাফল : ছবিটি হলো টমেটো সেপ্টোরিয়াল লিফ স্পট"
        if st.button("🔊 Speak (Bengali Result)"):
        # Convert the text to speech and play the audio
          audio_file_path_bn = text_to_speech(septo_bn, lang='bn')
          st.audio(audio_file_path_bn, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)
        
        english_text_septo = """Solution :
1. Remove infected leaves and dispose of them far from the field to manage Septoria leaf spot.
2. Apply fungicides containing chlorothalonil or copper-based products.
3. Water the plants at ground level, avoiding overhead irrigation, to reduce leaf wetness.
4. Crop rotation and planting resistant tomato varieties can be effective preventive measures."""

        bangla_text_septo = """প্রতিকার :
১. সেপ্টোরিয়া পাতার দাগ নিয়ন্ত্রণ করতে সংক্রামিত পাতাগুলি সরিয়ে ফেলুন এবং ক্ষেত থেকে দূরে ফেলে দিন।
২. ক্লোরোথালোনিল বা তামা-ভিত্তিক পণ্যযুক্ত ছত্রাকনাশক প্রয়োগ করুন।
৩. পাতার আর্দ্রতা কমাতে ওভারহেড সেচ এড়িয়ে গাছকে মাটির স্তরে জল দিন।
৪. শস্য আবর্তন এবং রোপণ প্রতিরোধী টমেটো জাত কার্যকর প্রতিরোধমূলক ব্যবস্থা হতে পারে।"""

        st.text(english_text_septo)


        if st.button("🔊 Speak (English)"):
   
          audio_file_path_en_in = text_to_speech(english_text_septo, lang='en-in')
          st.audio(audio_file_path_en_in, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)


        st.text(bangla_text_septo)
        if st.button("🔊 Speak (Bengali)"):
        # Convert the text to speech and play the audio
          audio_file_path_bn = text_to_speech(bangla_text_septo, lang='bn')
          st.audio(audio_file_path_bn, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)
        # Display the reference with the link
        st.write(f"Reference 1 : [{reference_text}]({reference_link})")
        st.write(f"Reference 2 : [{reference_text_bn}]({reference_link_bn})")



    elif pred_class == 'Tomato___Spider_mites Two-spotted_spider_mite':

        spider_en = "Result is Tomato Spider mites Two spotted spider mite"
        if st.button("🔊 Speak (English Result)"):
        # Convert the text to speech and play the audio
          audio_file_path_en = text_to_speech(spider_en, lang='en-in')
          st.audio(audio_file_path_en, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)

        st.write("<h3 style='text-align: center;'>ফলাফল : ছবিটি হলো টমেটো স্পাইডার মাইটস টু-স্পটেড স্পাইডার মাইট </h3>", unsafe_allow_html=True)

        spider_bn = "ফলাফল : ছবিটি হলো টমেটো স্পাইডার মাইটস টু-স্পটেড স্পাইডার মাইট"
        if st.button("🔊 Speak (Bengali Result)"):
        # Convert the text to speech and play the audio
          audio_file_path_bn = text_to_speech(spider_bn, lang='bn')
          st.audio(audio_file_path_bn, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)
        
        english_text_spider = """Solution :
1. Spray plants with a strong jet of water to dislodge spider mites.
2. Apply insecticidal soap or neem oil to control the mite population.
3. Introduce predatory mites or ladybugs to feed on the spider mites.
4. Maintain proper plant hygiene and avoid overcrowding to prevent mite infestations."""

        bangla_text_spider = """প্রতিকার :
১. স্পাইডার মাইটস অপসারণের জন্য একটি শক্তিশালী জেট জল দিয়ে উদ্ভিদ স্প্রে করুন।
২. মাইট জনসংখ্যা নিয়ন্ত্রণ করতে কীটনাশক সাবান বা নিম তেল প্রয়োগ করুন।
৩. স্পাইডার মাইট খাওয়ানোর জন্য শিকারী মাইট বা লেডিবাগের পরিচয় দিন।
৪. সঠিক উদ্ভিদের স্বাস্থ্যবিধি বজায় রাখুন এবং মাইটের উপদ্রব রোধ করতে অতিরিক্ত ভিড় এড়ান।"""

        st.text(english_text_spider)


        if st.button("🔊 Speak (English)"):
   
          audio_file_path_en_in = text_to_speech(english_text_spider, lang='en-in')
          st.audio(audio_file_path_en_in, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)


        st.text(bangla_text_spider)
        if st.button("🔊 Speak (Bengali)"):
        # Convert the text to speech and play the audio
          audio_file_path_bn = text_to_speech(bangla_text_spider, lang='bn')
          st.audio(audio_file_path_bn, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)
        # Display the reference with the link
        st.write(f"Reference 1 : [{reference_text}]({reference_link})")
        st.write(f"Reference 2 : [{reference_text_bn}]({reference_link_bn})")
       


    elif pred_class == 'Tomato___Target_Spot':

        target_en = "Result is Tomato Target Spot"
        if st.button("🔊 Speak (English Result)"):
        # Convert the text to speech and play the audio
          audio_file_path_en = text_to_speech(target_en, lang='en-in')
          st.audio(audio_file_path_en, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)

        st.write("<h3 style='text-align: center;'>ফলাফল : ছবিটি হলো টমেটো টারগেট স্পট </h3>", unsafe_allow_html=True)

        target_bn = "ফলাফল : ছবিটি হলো টমেটো টারগেট স্পট"
        if st.button("🔊 Speak (Bengali Result)"):
        # Convert the text to speech and play the audio
          audio_file_path_bn = text_to_speech(target_bn, lang='bn')
          st.audio(audio_file_path_bn, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)
        
        english_text_target = """Solution :
1. Remove and destroy infected leaves immediately.
2. Apply copper-based fungicides to control the disease.
3. Ensure proper air circulation and avoid overhead watering.
4. Plant resistant tomato varieties."""

        bangla_text_target = """প্রতিকার :
১. সঙ্গে সঙ্গে আক্রান্ত পাতা সরিয়ে নিন এবং ধ্বংস করে দিন।
২. রোগ নিয়ন্ত্রণের জন্য তামা ভিত্তিক ফাংগিসাইড প্রয়োগ করুন।
৩. উপযুক্ত বায়ু পরিপ্রেক্ষিতা নিশ্চিত করুন এবং ওভারহেড ওয়াটারিং এড়ান করুন।
৪. রোগ সহিষ্ণু টমেটো জাতিগুলি গাছ করুন।"""

        st.text(english_text_target)


        if st.button("🔊 Speak (English)"):
   
          audio_file_path_en_in = text_to_speech(english_text_target, lang='en-in')
          st.audio(audio_file_path_en_in, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)


        st.text(bangla_text_target)
        if st.button("🔊 Speak (Bengali)"):
        # Convert the text to speech and play the audio
          audio_file_path_bn = text_to_speech(bangla_text_target, lang='bn')
          st.audio(audio_file_path_bn, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)
        # Display the reference with the link
        st.write(f"Reference 1 : [{reference_text}]({reference_link})")
        st.write(f"Reference 2 : [{reference_text_bn}]({reference_link_bn})")
       



    elif pred_class == 'Tomato___Tomato_Yellow_Leaf_Curl_Virus':

        yellow_en = "Result is Tomato Yellow Leaf Curl Virust"
        if st.button("🔊 Speak (English Result)"):
        # Convert the text to speech and play the audio
          audio_file_path_en = text_to_speech(yellow_en, lang='en-in')
          st.audio(audio_file_path_en, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)

        st.write("<h3 style='text-align: center;'>ফলাফল : ছবিটি হলো টমেটো ইয়েলো লিফ কার্ল ভাইরাস </h3>", unsafe_allow_html=True)
        
        yellow_bn = "ফলাফল : ছবিটি হলো টমেটো ইয়েলো লিফ কার্ল ভাইরাস"
        if st.button("🔊 Speak (Bengali Result)"):
        # Convert the text to speech and play the audio
          audio_file_path_bn = text_to_speech(yellow_bn, lang='bn')
          st.audio(audio_file_path_bn, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)
        
        english_text_yellow = """Solution :
1. Remove and destroy infected plants.
2. Control whiteflies, the virus vector, using insecticides.
3. Plant virus-resistant tomato varieties.
4. Use reflective mulches to deter whiteflies."""

        bangla_text_yellow = """প্রতিকার :
১. আক্রান্ত গাছগুলি সরিয়ে নিন এবং ধ্বংস করে দিন।
২. পোকামাকড় নিয়ন্ত্রণের জন্য কীটনাশক ব্যবহার করুন।
৩. ভাইরাস সহিষ্ণু টমেটো জাতিগুলি গাছ করুন।
৪. সাদা মাছি প্রতিরোধ করতে প্রতিফলিত মালচ ব্যবহার করুন।"""

        st.text(english_text_yellow)


        if st.button("🔊 Speak (English)"):
   
          audio_file_path_en_in = text_to_speech(english_text_yellow, lang='en-in')
          st.audio(audio_file_path_en_in, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)


        st.text(bangla_text_yellow)
        if st.button("🔊 Speak (Bengali)"):
        # Convert the text to speech and play the audio
          audio_file_path_bn = text_to_speech(bangla_text_yellow, lang='bn')
          st.audio(audio_file_path_bn, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)
        # Display the reference with the link
        st.write(f"Reference 1 : [{reference_text}]({reference_link})")
        st.write(f"Reference 2 : [{reference_text_bn}]({reference_link_bn})")
       

        


    elif pred_class == 'Tomato___Tomato_mosaic_virus':

        mos_en = "Result is Tomato mosaic virus"
        if st.button("🔊 Speak (English Result)"):
        # Convert the text to speech and play the audio
          audio_file_path_en = text_to_speech(mos_en, lang='en-in')
          st.audio(audio_file_path_en, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)

        st.write("<h3 style='text-align: center;'>ফলাফল : ছবিটি হলো টমেটো মোজাইক ভাইরাস </h3>", unsafe_allow_html=True)
        mos_bn = "ফলাফল : ছবিটি হলো টমেটো মোজাইক ভাইরাস"
        if st.button("🔊 Speak (Bengali Result)"):
        # Convert the text to speech and play the audio
          audio_file_path_bn = text_to_speech(mos_bn, lang='bn')
          st.audio(audio_file_path_bn, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)
        
        english_text_mos = """Solution :
1. Remove and destroy infected plants.
2. Control aphids, which can spread the virus, using insecticides.
3. Plant virus-resistant tomato varieties.
4. Practice good hygiene and sanitation to prevent the virus's spread."""

        bangla_text_mos = """প্রতিকার :
১. আক্রান্ত গাছগুলি সরিয়ে নিন এবং ধ্বংস করে দিন।
২. কীটনাশক ব্যবহার করে এফিড নিয়ন্ত্রণ করুন, যা ভাইরাস ছড়াতে পারে।
৩. ভাইরাস সহিষ্ণু টমেটো জাতিগুলি গাছ করুন।
৪. ভাইরাস ছড়ানোর প্রতিরোধে ভাল স্বাস্থ্য ও স্যানিটেশন অনুষ্ঠান অনুসরণ করুন।"""

        st.text(english_text_mos)


        if st.button("🔊 Speak (English)"):
   
          audio_file_path_en_in = text_to_speech(english_text_mos, lang='en-in')
          st.audio(audio_file_path_en_in, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)


        st.text(bangla_text_mos)
        if st.button("🔊 Speak (Bengali)"):
        # Convert the text to speech and play the audio
          audio_file_path_bn = text_to_speech(bangla_text_mos, lang='bn')
          st.audio(audio_file_path_bn, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)
        # Display the reference with the link
        st.write(f"Reference 1 : [{reference_text}]({reference_link})")
        st.write(f"Reference 2 : [{reference_text_bn}]({reference_link_bn})")
       
        


    elif pred_class == 'Tomato___healthy':

        hel_en = "Result is Tomato Healthy"
        if st.button("🔊 Speak (English Result)"):
        # Convert the text to speech and play the audio
          audio_file_path_en = text_to_speech(hel_en, lang='en-in')
          st.audio(audio_file_path_en, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)

        st.write("<h3 style='text-align: center;'>ফলাফল : ছবিটি হলো টমেটো হেলদি </h3>", unsafe_allow_html=True)

        hel_bn = "ফলাফল : ছবিটি হলো টমেটো হেলদি"
        if st.button("🔊 Speak (Bengali Result)"):
        # Convert the text to speech and play the audio
          audio_file_path_bn = text_to_speech(hel_bn, lang='bn')
          st.audio(audio_file_path_bn, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)
        
        english_text_hel = """Ways to keep healthy :
1. Provide adequate sunlight, water, and nutrients.
2. Monitor for pests and diseases and take prompt action if any issues arise.
3. Prune the tomato plants to improve air circulation.
4. Use mulch to conserve moisture and suppress weeds."""

        bangla_text_hel = """হেলদি রাখার উপায় :
১. যথাযথ সূর্যের আলো, পানি এবং পুষ্টি সরবরাহ করুন।
২. পোকা এবং রোগের জন্য নজরদারি করুন এবং প্রয়োজনে দ্রুত কর্মব্যবস্থা নিন।
৩. টমেটো গাছগুলি ছাঁটুন যাতে বায়ু পরিপ্রেক্ষিতা উন্নত হয়।
৪. আর্দ্রতা সংরক্ষণ এবং আগাছা দমন করতে মাল্চ ব্যবহার করুন।"""

        st.text(english_text_hel)


        if st.button("🔊 Speak (English)"):
   
          audio_file_path_en_in = text_to_speech(english_text_hel, lang='en-in')
          st.audio(audio_file_path_en_in, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)


        st.text(bangla_text_hel)
        if st.button("🔊 Speak (Bengali)"):
        # Convert the text to speech and play the audio
          audio_file_path_bn = text_to_speech(bangla_text_hel, lang='bn')
          st.audio(audio_file_path_bn, format='audio/mp3')
          st.write(f'<style>.css-1q5b6h4 {{ {voice_button_style} }}</style>', unsafe_allow_html=True)
        # Display the reference with the link
        st.write(f"Reference 1 : [{reference_text}]({reference_link})")
        st.write(f"Reference 2 : [{reference_text_bn}]({reference_link_bn})")
       





