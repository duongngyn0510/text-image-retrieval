// Variables to use accross the project
// which can be accessed by var.project_id
variable "project_id" {
  description = "The project ID to host the cluster in"
  default     = "mle-course-398013"
}

variable "region" {
  description = "The region the cluster in"
  default     = "us-central1"
}

variable "bucket" {
  description = "GCS bucket for store image products"
  default     = "fashion_image"
}
