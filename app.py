import streamlit as st
import cv2
import numpy as np

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils

# Define your process_image function here
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    edged = cv2.Canny(blur, 50, 100)
    
    dilate = cv2.dilate(edged, None, iterations=1)
    morph = cv2.erode(dilate, None, iterations=1)
    return gray, blur, edged, morph

# Create a session state to store variables across sessions
if 'current_result_index' not in st.session_state:
    st.session_state.current_result_index = 0

if 'result_images' not in st.session_state:
    st.session_state.result_images = []

def main():
    st.title("Object Measurement")
    with st.sidebar:
        st.header('Settings')
        object_width = st.number_input("Enter object width", min_value=0.1, value=1.0, step=0.1)
        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        # Declare the image variable outside the if block
        image = None
        if image_file:
            image = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image, use_column_width=True, channels='RGB')
            
    
    columns = st.columns(4)
    if image is not None:
        gray, blur, edged, morph = preprocess_image(image)
        for i, img in enumerate([gray, blur, edged,morph]):
            with columns[i]:
                if i == 0:
                    st.image(img, caption="Gray", use_column_width=True, channels='GRAY')
                elif i == 1:
                    st.image(img, caption="Blur", use_column_width=True, channels='GRAY')
                elif i == 2:
                    st.image(img, caption="Edge Detection", use_column_width=True, channels='GRAY')
                else:
                    st.image(img, caption="Morphological Operations", use_column_width=True, channels="GRAY")

    # processed_image = st.button("Process Image")

    if not st.session_state.result_images:
        st.session_state.current_result_index = 0

        result_images = []

        if image is not None:  # Check if the image is not None
            # Process the image and convert it to grayscale
            gray, blur, edged, morph = preprocess_image(image)
            
            cnts = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            print("Total number of contours are: ", len(cnts))

            (cnts, _) = contours.sort_contours(cnts)
            pixelPerMetric = None
            count = 0

            for c in cnts:
                if cv2.contourArea(c) < 300:
                    continue
                count += 1

                orig = image.copy()
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")

                box = perspective.order_points(box)
                cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

                for (x, y) in box:
                    cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)

                cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

                cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
                cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                if pixelPerMetric is None:
                    pixelPerMetric = dB / object_width

                dimA = dA / pixelPerMetric
                dimB = dB / pixelPerMetric

                cv2.putText(orig, "{:.1f}mm".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(orig, "{:.1f}mm".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    
                result_images.append(orig)

        st.session_state.result_images = result_images

    if st.session_state.result_images:
        st.image(st.session_state.result_images[st.session_state.current_result_index], caption="Processed Image with Measurements", use_column_width=True, channels="RGB")
        button1, button2 = st.columns(2)
        previous_button =  button1.button('Previous')
        next_button = button2.button('Next')
        if len(st.session_state.result_images) > 1:
            if previous_button:
                st.session_state.current_result_index = (st.session_state.current_result_index - 1) % len(st.session_state.result_images)
            if next_button:
                st.session_state.current_result_index = (st.session_state.current_result_index + 1) % len(st.session_state.result_images)

if __name__ == "__main__":
    main()
