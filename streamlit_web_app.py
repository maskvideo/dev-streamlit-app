import boto3 as boto3
import streamlit as st
import retina
import extract_frames
import cv2
import numpy as np
import aws_client
import PIL.Image

global kernel_size
global epsilon


def convert_bytes_to_opencv(bytes_image):
    np_img = cv2.imdecode(np.frombuffer(bytes_image, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

def update_masked_image(masked):
    aws_client.upload_image_to_s3(masked)
    st.write("")

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
    with st.spinner(""):
        masked = retina.update_parameters(unmasked_pil_img, (kernel_size, kernel_size), epsilon, faces_locations)
        update_masked_image(masked)

    with st.spinner(""):
        masked = retina.update_parameters(unmasked_pil_img, (kernel_size, kernel_size), epsilon, faces_locations)
        update_masked_image(masked)

#upload video to the s3 bucket
uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])
print(uploaded_file)
# S3 upload logic
if uploaded_file is not None:
    s3 = boto3.client('s3', aws_access_key_id=aws_client.aws_access_key_id, aws_secret_access_key=aws_client.aws_secret_access_key)
    with st.spinner('Uploading...'):
        video_name = uploaded_file.name
        s3.upload_fileobj(uploaded_file, aws_client.BUCKET_NAME, video_name)
        # Get the name of the uploaded video
    st.write('Upload successful!')

if st.button("Process video") and uploaded_file is not None:
    with st.spinner("Extracting frames from video..."):
        extract_frames.extract_frames_from_video(aws_client.get_video_url(video_name))
    aws_client.delete_file(video_name)

if st.button("Mask video"):
    with st.spinner("Masking video..."):
        frames_files = extract_frames.sorted_frames_files(aws_client.BUCKET_NAME, "unmasked_frames/")
        st.write("start mask")
        masked_frames = []
        for frame in frames_files:
            print(frame)
            frame_obj = aws_client.s3_client.get_object(Bucket=aws_client.BUCKET_NAME, Key=frame)
            frame_bytes = frame_obj['Body'].read()

            unmasked_pil_img = PIL.Image.open(io.BytesIO(frame_bytes)).convert("RGB")
            unmasked_np_img = np.array(unmasked_pil_img)
            faces_locations = retina.all_faces_locations(unmasked_np_img)
            masked_frame = retina.update_parameters(unmasked_np_img, (kernel_size, kernel_size), epsilon, faces_locations)
            masked_frames.append(masked_frame)      
        st.write("end mask")