variable "instance_name" {
  description = "Name for the VM instance"
  type        = string
  default     = "mlops-serving"
}

variable "image_name" {
  description = "Chameleon KVM base image"
  type        = string
  default     = "CC-Ubuntu24.04"
}

variable "reservation_id" {
  description = "Active Chameleon reservation ID for the leased KVM flavor"
  type        = string
}

variable "key_pair_name" {
  description = "Name of the SSH keypair already uploaded to Chameleon"
  type        = string
}

variable "network_name" {
  description = "Network to attach the instance to"
  type        = string
  default     = "sharednet1"
}

# ── Teammate service endpoints ───────────────────────────────────────────────

variable "postgres_dsn" {
  description = "PostgreSQL connection string for the ML serving backend"
  type        = string
  default     = "postgresql://mlops_user:mlops_pass@localhost:5432/mlops"
}

variable "mlflow_tracking_uri" {
  description = "MLflow tracking server URL"
  type        = string
  default     = "http://localhost:5000"
}

variable "embedding_service_url" {
  description = "Embedding microservice URL"
  type        = string
  default     = "http://localhost:8001"
}

variable "mlflow_model_uri" {
  description = "MLflow model registry URI"
  type        = string
  default     = "models:/layer1-classifier/latest"
}

variable "layer1_model_kind" {
  type    = string
  default = "hf"
}

variable "layer1_mlflow_run_id" {
  type    = string
  default = "5ff06329d3af4701a4cc83a659a7d07b"
}

variable "layer1_mlflow_artifact_path" {
  type    = string
  default = "minilm"
}

variable "repo_url" {
  description = "Git repository to clone onto the VM before building containers"
  type        = string
  default     = "https://github.com/jayrajpamnani/MLOps-Serving.git"
}

variable "repo_ref" {
  description = "Git branch, tag, or commit to check out on the VM"
  type        = string
  default     = "main"
}

variable "actual_build_node_memory_mb" {
  description = "Node.js heap size for building the custom Actual image"
  type        = number
  default     = 4096
}

variable "docker_platform" {
  description = "Target platform for Docker builds on the deployment host"
  type        = string
  default     = "linux/amd64"
}
