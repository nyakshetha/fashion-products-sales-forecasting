"""Central configuration: paths, seeds, garment categories and research groupings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "fitting_room_dataset.csv"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
TRENDS_PATH = EXTERNAL_DIR / "google_trends_weekly.csv"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
METRICS_DIR = OUTPUTS_DIR / "metrics"
REPORTS_DIR = OUTPUTS_DIR / "reports"

SEED = 42

# ---- Scale of the (synthetic) study, matching the brief: 1,630 customers / 6,540 trials ----
N_CUSTOMERS = 1630
N_TRIALS = 6540
N_GARMENTS = 420
STUDY_DAYS = 90                 # ~three months
STUDY_START = "2025-01-06"      # a Monday; the dates are simulated
TARGET_CONVERSION = 0.41        # overall purchase rate targeted by the generator (not forced 50:50)

# ---- Garment attributes ----
SIZES = [36, 38, 39, 40, 42, 44, 46, 48]
SIZE_PROBS = [0.06, 0.14, 0.16, 0.24, 0.20, 0.12, 0.06, 0.02]
SLEEVES = ["Half", "Full"]
SLEEVE_PROBS = [0.28, 0.72]
COLORS = ["Beige", "Black", "Blue", "Brown", "Cream", "Green", "Grey", "Khaki", "Light Blue", "Lilac"]
COLOR_PROBS = [0.09, 0.14, 0.16, 0.07, 0.08, 0.08, 0.11, 0.08, 0.13, 0.06]
COLLARS = ["Button-down", "Regular", "Band"]
COLLAR_PROBS = [0.30, 0.55, 0.15]
FITS = ["Slim", "Ultra Slim"]
FIT_PROBS = [0.68, 0.32]
PATTERNS = ["Print", "Patterned", "Graphic", "Checks", "Dots", "Stripes", "Solid", "Textured"]
PATTERN_PROBS = [0.09, 0.08, 0.06, 0.15, 0.06, 0.17, 0.30, 0.09]
PRICE_MIN = 934
PRICE_MAX = 3999

# ---- Feature aggregation (research approach: colour / pattern / size / price-range groups) ----
COLOR_TO_GROUP = {
    "Beige": "warm", "Brown": "warm", "Cream": "warm", "Khaki": "warm",
    "Blue": "cool", "Light Blue": "cool", "Green": "cool", "Lilac": "cool",
    "Black": "neutral", "Grey": "neutral",
}
PATTERN_TO_GROUP = {
    "Solid": "solid", "Textured": "texture",
    "Checks": "geometric", "Dots": "geometric", "Stripes": "geometric",
    "Print": "abstract", "Patterned": "abstract", "Graphic": "abstract",
}
SIZE_TO_GROUP = {36: "S", 38: "S", 39: "M", 40: "M", 42: "L", 44: "L", 46: "XL", 48: "XL"}
COLOR_GROUP_ORDER = ["warm", "cool", "neutral"]
PATTERN_GROUP_ORDER = ["solid", "texture", "geometric", "abstract"]
PRICE_BAND_WIDTH = 500

# ---- Columns ----
ID_COLS = ["Customer_ID", "Garment_ID"]
ATTR_COLS = ["Size", "Sleeve_Type", "Color", "Collar_Type", "Fit", "Pattern", "Price"]
CAT_FEATURES = ["Size", "Sleeve_Type", "Color", "Collar_Type", "Fit", "Pattern"]
TARGET = "Purchased"
CORE_COLUMNS = ID_COLS + ATTR_COLS + ["Trial_Count", TARGET]
ALL_COLUMNS = CORE_COLUMNS + ["Trial_Timestamp", "Trend_Index"]

# ---- Clustering ----
N_CLUSTERS = 6          # optimum reported in the research brief
K_RANGE = list(range(2, 11))

# ---- Purchase-likelihood model ----
PREFERRED_MODEL = "XGBoost"     # project requirement; reported next to the alternatives
CLASS_THRESHOLD = 0.5
TIER_MARGIN = 0.10              # High/Low = base conversion rate +/- this margin
DEMAND_HIGH = 1.20              # relative demand index thresholds
DEMAND_LOW = 0.80

TREND_METADATA_PATH = EXTERNAL_DIR / "google_trends_metadata.json"
TRENDS_DIR = EXTERNAL_DIR

# ---- Google Trends (PyTrends) keywords: one weekly interest series per key ----
TREND_KEYWORDS = {
    "pattern:solid": "plain shirt",
    "pattern:texture": "textured shirt",
    "pattern:geometric": "checks shirt",
    "pattern:abstract": "printed shirt",
    "fit:Slim": "slim fit shirt",
    "fit:Ultra Slim": "super slim fit shirt",
}

# Fashion keyword fallbacks tried if primary keyword returns empty or low data
TREND_KEYWORD_FALLBACKS = {
    "pattern:solid": ["solid shirt", "plain shirt for men", "formal shirt"],
    "pattern:texture": ["textured shirt for men", "linen shirt", "cotton shirt"],
    "pattern:geometric": ["check shirt", "striped shirt", "checks shirt for men"],
    "pattern:abstract": ["printed shirt for men", "graphic shirt", "floral shirt"],
    "fit:Slim": ["slim fit shirt men", "slim shirt"],
    "fit:Ultra Slim": ["skinny fit shirt", "extra slim fit shirt", "slim fit shirt"],
}

# Clean, retailer-facing display names for charts and KPI cards
TREND_DISPLAY_NAMES = {
    "pattern:solid": "Plain / Solid Shirts",
    "pattern:texture": "Textured Shirts",
    "pattern:geometric": "Checks & Stripes",
    "pattern:abstract": "Printed & Graphic",
    "fit:Slim": "Slim Fit Shirts",
    "fit:Ultra Slim": "Super Slim / Skinny",
}

TREND_GEO = "IN"
TREND_TIMEFRAME = "today 3-m"
TREND_CACHE_TTL_DAYS = 7        # Cache lifetime before refreshing live trends

# ---- Dashboard ----
APP_NAME = "StockSense"
SHOW_DEMO_BADGE = False         # Retailer-facing mode; demo badge disabled
