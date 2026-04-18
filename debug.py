# debug_full.py - Check all encoders
import joblib

encoders = joblib.load('models/encoders.pkl')

print("=" * 50)
print("ALL ENCODER CLASSES:")
print("=" * 50)

for col, encoder in encoders.items():
    print(f"\n{col}:")
    print(f"  Values: {list(encoder.classes_)}")