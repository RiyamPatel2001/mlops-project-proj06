import os


POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://mlops_user:mlops_pass@postgres.mlops.svc.cluster.local:5432/mlops",
)

AUTH_SESSION_TTL_HOURS = int(os.getenv("AUTH_SESSION_TTL_HOURS", "720"))
AUTH_PASSWORD_HASH_ITERATIONS = int(
    os.getenv("AUTH_PASSWORD_HASH_ITERATIONS", "200000")
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow-proj06.mlops.svc.cluster.local:5000",
)

MODEL_REGISTRY_PATH = os.getenv(
    "MODEL_REGISTRY_PATH",
    "/app/models/layer1_registry.json",
)
MODEL_REGISTRY_REFRESH_SECONDS = float(
    os.getenv("MODEL_REGISTRY_REFRESH_SECONDS", "30")
)

LAYER1_HF_MAX_LENGTH = int(os.getenv("LAYER1_HF_MAX_LENGTH", "128"))

TIER_GOOD_MODEL_NAME = os.getenv("TIER_GOOD_MODEL_NAME", "minilm")
TIER_GOOD_MODEL_KIND = os.getenv("TIER_GOOD_MODEL_KIND", "hf").lower()
TIER_GOOD_RUN_ID = os.getenv(
    "TIER_GOOD_RUN_ID",
    "464cacf29c054edca5aa6ddc62f8816a",
)
TIER_GOOD_ARTIFACT_PATH = os.getenv("TIER_GOOD_ARTIFACT_PATH", "minilm")

TIER_FAST_MODEL_NAME = os.getenv("TIER_FAST_MODEL_NAME", "fasttext")
TIER_FAST_MODEL_KIND = os.getenv("TIER_FAST_MODEL_KIND", "fasttext").lower()
TIER_FAST_RUN_ID = os.getenv(
    "TIER_FAST_RUN_ID",
    "e99c0f7fe5554fd584c9efd2162f5572",
)
TIER_FAST_ARTIFACT_PATH = os.getenv("TIER_FAST_ARTIFACT_PATH", "fasttext.bin")

TIER_CHEAP_MODEL_NAME = os.getenv("TIER_CHEAP_MODEL_NAME", "tf_idf_logreg")
TIER_CHEAP_MODEL_KIND = os.getenv("TIER_CHEAP_MODEL_KIND", "sklearn").lower()
TIER_CHEAP_RUN_ID = os.getenv(
    "TIER_CHEAP_RUN_ID",
    "b69934eb9ef14e0a960a5b6345b0d8a4",
)
TIER_CHEAP_ARTIFACT_PATH = os.getenv(
    "TIER_CHEAP_ARTIFACT_PATH",
    "tfidf_logreg.joblib",
)

ROUTER_REQUEST_WINDOW_SECONDS = int(
    os.getenv("ROUTER_REQUEST_WINDOW_SECONDS", "10")
)
ROUTER_BATCH_STICKY_TTL_SECONDS = int(
    os.getenv("ROUTER_BATCH_STICKY_TTL_SECONDS", "900")
)
ROUTER_OVERLOAD_INFLIGHT_THRESHOLD = int(
    os.getenv("ROUTER_OVERLOAD_INFLIGHT_THRESHOLD", "24")
)
ROUTER_OVERLOAD_SUSTAIN_SECONDS = float(
    os.getenv("ROUTER_OVERLOAD_SUSTAIN_SECONDS", "2.0")
)

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
