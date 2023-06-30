import boto3 as boto3
import streamlit as st
import retina
import extract_frames
import cv2
import numpy as np
import aws_client
global kernel_size
global epsilon


def convert_bytes_to_opencv(bytes_image):
    np_img = cv2.imdecode(np.frombuffer(bytes_image, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

def update_masked_image(masked):
    aws_client.upload_image_to_s3(masked)
    st.write("Done uploading.")

    # display masked image
    masked_image_s3 = aws_client.image_from_s3(aws_client.BUCKET_NAME, "masked_people.jpg")
    masked_opencv_img = convert_bytes_to_opencv(masked_image_s3)

    image_placeholder.image(masked_opencv_img)

# Set the page title
st.set_page_config(page_title="Mask Video File - Preview")

# Add a title
st.title("Preview")

slider_value = None


# display unmasked image
unmasked_image_s3 = aws_client.image_from_s3(aws_client.BUCKET_NAME, aws_client.KEY)
unmasked_pil_img = cv2.imdecode(np.frombuffer(unmasked_image_s3, np.uint8), cv2.IMREAD_COLOR)

unmasked_opencv_img = convert_bytes_to_opencv(unmasked_image_s3)

st.image(unmasked_opencv_img, caption='Unmasked Image')
# Create an empty placeholder for the image
image_placeholder = st.empty()


faces_locations = retina.all_faces_locations(unmasked_pil_img)
masked = retina.update_parameters(unmasked_pil_img, (5,5), 10, faces_locations)

frames_files = []


kernel_size = st.slider("Choose blurr", 0, 100)
epsilon = st.slider("Choose coverage", 0, 40)
if st.button("Update"):
    slider_value = (kernel_size, epsilon)
if slider_value is not None:
    with st.spinner("Applying mask..."):
        masked = retina.update_parameters(unmasked_pil_img, (kernel_size, kernel_size), epsilon, faces_locations)
        update_masked_image(masked)

    with st.spinner("Applying mask..."):
        masked = retina.update_parameters(unmasked_pil_img, (kernel_size, kernel_size), epsilon, faces_locations)
        update_masked_image(masked)

# Create a text input field for the video path
video_path = st.text_input("Enter the path to the video file")

# Force the text input field to lose focus
st.write('')


uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

# S3 upload logic
if uploaded_file is not None:
    s3 = boto3.client('s3')
    with st.spinner('Uploading...'):
        unmasked_video_name = uploaded_file.name
        s3.upload_fileobj(uploaded_file, aws_client.BUCKET_NAME, unmasked_video_name)
    st.write('Upload successful!')



# TODO: convert this to work with s3
if st.button("Mask video") and uploaded_file is not None:
    with st.spinner("Extracting frames from video..."):
        extract_frames.extract_frames_from_video(aws_client.get_video_url(unmasked_video_name))
    frames_files = extract_frames.sorted_frames_files(aws_client.BUCKET_NAME, "unmasked_frames/")
    st.write("start mask")

    masked_frames = []

    progress_bar = st.progress(0)


    for idx, frame in enumerate(frames_files):
        frame_image_s3 = aws_client.image_from_s3(aws_client.BUCKET_NAME, frame)
        unmasked_frame_img = cv2.imdecode(np.frombuffer(frame_image_s3, np.uint8), cv2.IMREAD_COLOR)
        faces_locations = retina.all_faces_locations(unmasked_frame_img)
        masked_frames.append(retina.update_parameters(unmasked_frame_img, (kernel_size, kernel_size), epsilon, faces_locations))

        # Update the progress bar
        progress = (idx + 1) / len(frames_files)
        progress_bar.progress(progress)
    st.write("end mask")
