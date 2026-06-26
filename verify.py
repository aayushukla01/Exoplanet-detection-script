import sys

print("--- Checking Core Environment Setup ---")
print(f"Python version: {sys.version.split()[0]}")

# We temporarily omitted Batman so your environment builds cleanly
libraries = [
    ("numpy", "NumPy"),
    ("matplotlib", "Matplotlib"),
    ("astropy", "Astropy"),
    ("lightkurve", "Lightkurve"),
    ("wotan", "Wotan"),
    ("transitleastsquares", "Transit Least Squares (TLS)"),
    ("sklearn", "Scikit-Learn"),
    ("xgboost", "XGBoost")
]

for mod_name, label in libraries:
    try:
        __import__(mod_name)
        print(f"✅ {label}: Ready")
    except ImportError:
        print(f"❌ {label}: Not found")

print("\n--- Verification Complete ---")