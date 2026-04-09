import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_excel("crop_yield.xlsx")

# Encode categorical data
le_crop = LabelEncoder()
le_state = LabelEncoder()
le_season = LabelEncoder()

df["Crop"] = le_crop.fit_transform(df["Crop"])
df["State"] = le_state.fit_transform(df["State"])
df["Season"] = le_season.fit_transform(df["Season"])

# Features
X = df[["Crop", "State", "Season", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]]
y = df["Yield"]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save files
joblib.dump(model, "crop_yield_model.pkl")
joblib.dump((le_crop, le_state, le_season), "encoder.pkl")

print("✅ Model created successfully!")