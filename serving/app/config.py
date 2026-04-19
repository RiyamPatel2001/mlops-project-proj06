import os


POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://mlops_user:mlops_pass@postgres.mlops.svc.cluster.local:5432/mlops",
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow-proj06.mlops.svc.cluster.local:5000",
)

MLFLOW_MODEL_URI = os.getenv(
    "MLFLOW_MODEL_URI",
    "models:/layer1-classifier/latest",
)

LAYER1_MODEL_KIND = os.getenv("LAYER1_MODEL_KIND", "hf").lower()

LAYER1_MLFLOW_RUN_ID = os.getenv(
    "LAYER1_MLFLOW_RUN_ID",
    "5ff06329d3af4701a4cc83a659a7d07b",
)
LAYER1_MLFLOW_ARTIFACT_PATH = os.getenv("LAYER1_MLFLOW_ARTIFACT_PATH", "minilm")

LAYER1_HF_MAX_LENGTH = int(os.getenv("LAYER1_HF_MAX_LENGTH", "128"))

EMBEDDING_SERVICE_URL = os.getenv(
    "EMBEDDING_SERVICE_URL",
    "http://embedding-service:8001",
)

LAYER2_SIMILARITY_THRESHOLD = float(os.getenv("LAYER2_SIMILARITY_THRESHOLD", "0.6"))
LAYER2_TOP_K = int(os.getenv("LAYER2_TOP_K", "5"))
LAYER2_MIN_EXAMPLES = int(os.getenv("LAYER2_MIN_EXAMPLES", "2"))

CONFIDENCE_AUTOFILL_THRESHOLD = 0.6

LABEL_CLASSES = [
    "Alcohol", "Charitable Giving", "Childcare", "Clothing", "Dining Out",
    "Education", "Entertainment", "Groceries", "Health Insurance", "Healthcare",
    "Home Improvement", "Household Supplies", "Insurance", "Other",
    "Personal Care", "Pets", "Phone & Internet", "Property Tax",
    "Public Transit", "Reading", "Rent / Mortgage", "Savings", "Streaming",
    "Tobacco", "Transport", "Travel", "Utilities", "Vehicle Insurance",
    "Vehicle Payment",
]
