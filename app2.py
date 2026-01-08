import streamlit as st
import numpy as np
import cv2
from tf_keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Load your trained MNIST model
model = load_model("mnist.h5")

st.title("🖌️ Handwritten Digit Recognition")

# Create a drawable canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",  # Transparent fill
    stroke_width=8,
    stroke_color="black",
    background_color="white",
    width=300,
    height=300,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Recognize Digit"):
    if canvas_result.image_data is not None:
        # Convert RGBA to grayscale
        img = cv2.cvtColor(canvas_result.image_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)

        # Threshold to binary
        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours (multiple digits possible)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            digit = th[y:y+h, x:x+w]

            # Heuristic check: reject if too small or broken
            filled_ratio = cv2.countNonZero(digit) / (w * h)
            if w < 10 or h < 10 or filled_ratio < 0.1:
                results.append(("Not recognised", 0))
                continue

            # Resize to 18x18 then pad to 28x28
            resized_digit = cv2.resize(digit, (18, 18))
            padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

            # Normalize and reshape
            digit_input = padded_digit.reshape(1, 28, 28, 1) / 255.0

            # Predict
            pred = model.predict(digit_input)[0]
            final_pred = np.argmax(pred)
            confidence = int(max(pred) * 100)

            # Confidence threshold check
            if confidence < 50:  # reject low-confidence predictions
                results.append(("Not recognised", confidence))
            else:
                results.append((final_pred, confidence))

        # Sort contours left-to-right so digits appear in order
        contours_sorted = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        st.write("### Predictions:")
        for i, (digit, conf) in enumerate(results):
            if digit == "Not recognised":
                st.write(f"Digit : **Not recognised**")
            else:
                st.write(f"Digit : **{digit}** ({conf}%)")
    else:
        st.warning("Please draw a digit on the canvas first!")