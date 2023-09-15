terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "4.80.0"
    }
  }
  required_version = "1.5.7"
}

provider "google" {
  project     = var.project_id
  region      = var.region
}

resource "google_storage_bucket" "fashion_image" {
  name          = var.bucket
  location      = var.region
  uniform_bucket_level_access = true
}

resource "google_storage_bucket_iam_member" "public_read" {
  bucket = google_storage_bucket.fashion_image.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}
