import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ API key not found. Please create a .env file with GEMINI_API_KEY.")
    st.stop()

# Streamlit layout
st.set_page_config(layout="wide")
col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.subheader("")

# Configure Gemini API
genai.configure(api_key="enter_your_api_key") # here enter your gemini api key
model = genai.GenerativeModel('gemini-1.5-flash')

# Webcam settings
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand detector setup
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1,
                        detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up → draw
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, prev_pos, current_pos, (0, 255, 0), thickness=8, lineType=cv2.LINE_AA)
    elif fingers == [1, 0, 0, 0, 0]:  # Thumb up → clear
        canvas = np.zeros_like(canvas)
        return None, canvas, ""  
    return current_pos, canvas, None

def sendToAI(model, canvas, fingers):
    if fingers == [0, 1, 1, 1, 0]:  # Index + Middle + Ring up → send to AI
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (512, 512))
        pil_image = Image.fromarray(resized)
        response = model.generate_content([
            "This is a handwritten math expression. Recognize and solve it clearly.",
            pil_image
        ])
        return response.text
    return None

prev_pos = None
canvas = None
output_text = ""

# Main loop
while run:
    success, img = cap.read()
    if not success:
        st.error("Failed to access camera")
        break

    img = cv2.flip(img, 1)
    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas, cleared = draw(info, prev_pos, canvas)
        if cleared == "":
            output_text = ""
        result = sendToAI(model, canvas, fingers)
        if result:
            output_text = result

    combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(combined, channels="BGR")

    if output_text:
        output_text_area.text(output_text)

    cv2.waitKey(1)
