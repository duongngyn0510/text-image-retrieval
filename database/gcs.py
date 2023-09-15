import concurrent.futures

from google.cloud import storage


class GCS:
    def __init__(self, bucket_name, storage_client):
        self.bucket_name = bucket_name
        self.storage_client = storage_client

    def upload_glami_images(self, image_files):
        def upload(image_file):
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(image_file.split("/")[-1])
            blob.upload_from_filename(filename=image_file)

        with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
            executor.map(upload, image_files)

    def upload_images(self, image_files):
        bucket = self.storage_client.bucket(self.bucket_name)
        for image_file in image_files:
            blob = bucket.blob(image_file)
            blob.upload_from_filename(filename=image_file)

    def count_images(self):
        bucket = self.storage_client.bucket(self.bucket_name)
        blobs = list(bucket.list_blobs())
        return len(blobs)

    def get_images_url(self, image):
        bucket = self.storage_client.get_bucket(self.bucket_name)
        blob = bucket.blob(image)
        return blob.public_url


gcs = GCS(
    "fashion_image",
    storage.Client.from_service_account_json("secrets/mle-course-00d1417f42ca.json"),
)
