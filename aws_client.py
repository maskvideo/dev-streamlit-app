import io
import os
import boto3
import cv2
import numpy as np
import retina

from PIL import Image
BUCKET_NAME = 'face.mask' # replace with your bucket name
KEY = 'people.jpg' # replace with your object key

aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

s3_resource = boto3.resource('s3', aws_access_key_id=aws_access_key_id,
                  aws_secret_access_key=aws_secret_access_key)
s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                  aws_secret_access_key=aws_secret_access_key)


def image_from_s3(bucket, key):

    bucket = s3_resource.Bucket(bucket)
    image = bucket.Object(key)
    img_data = image.get().get('Body').read()

    return img_data


def upload_image_to_s3(pil_img):
    # Convert PIL Image object to bytes
    pil_bytes = io.BytesIO()
    pil_img.save(pil_bytes, format='JPEG')
    pil_bytes.seek(0)

    # Upload bytes to S3 bucket
    s3_client.upload_fileobj(pil_bytes, BUCKET_NAME, 'masked_people.jpg')

    print("done uploading")


def upload_unmasked_frame(frame, path):
    return s3_client.put_object(Bucket=BUCKET_NAME, Key=path, Body=cv2.imencode('.jpg', frame)[1].tobytes())


def get_video_url(file_name):
    expiration_time = 3600  # URL expiration time in seconds (1 hour in this example)

    url = s3_client.generate_presigned_url(
        ClientMethod='get_object',
        Params={'Bucket': BUCKET_NAME, 'Key': file_name},
        ExpiresIn=expiration_time
    )
    return url

def delete_file(file_name):
    s3_client.delete_object(Bucket=BUCKET_NAME, Key=file_name)


def delete_folder(folder_name):
    # List objects in the folder
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=folder_name)

    # Delete each object in the folder
    if 'Contents' in response:
        for obj in response['Contents']:
            s3_client.delete_object(Bucket=BUCKET_NAME, Key=obj['Key'])





def test_run():
    img = cv2.imdecode(np.frombuffer(image_from_s3(BUCKET_NAME, KEY), np.uint8), cv2.IMREAD_COLOR)

    locations = retina.all_faces_locations(img)
    masked = retina.update_parameters(img, (20,20), 10, locations)

    upload_image_to_s3(masked)

