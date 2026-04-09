from flask import Flask, request, jsonify, render_template, redirect, send_file
from flask_cors import CORS
import joblib
import pandas as pd
import sqlite3

app = Flask(__name__)  # fixed __name__
CORS(app)

# ================= DATABASE =================
def create_db():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("""  
    CREATE TABLE IF NOT EXISTS users(  
        id INTEGER PRIMARY KEY AUTOINCREMENT,  
        username TEXT,  
        password TEXT  
    )  
    """)
    conn.commit()
    conn.close()

create_db()

# ================= LOAD MODEL =================
model = joblib.load("crop_yield_model.pkl")
encoder = joblib.load("encoder.pkl")
if isinstance(encoder, tuple):
    encoder = encoder[0]
df = pd.read_excel("crop_yield.xlsx")
df.columns = df.columns.str.strip()

# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("login.html")

@app.route("/register")
def register():
    return render_template("register.html")


@app.route("/get-options")
def get_options():
    return jsonify({
        "crops": sorted(df["Crop"].dropna().unique().tolist()),
        "states": sorted(df["State"].dropna().unique().tolist()),
        "seasons": sorted(df["Season"].dropna().unique().tolist())
    })

@app.route("/index")
def index():
    return render_template("index.html")

# ================= DOWNLOAD DATASET =================
@app.route('/download_dataset')
def download_dataset():
    return send_file('crop_yield.xlsx', as_attachment=True)

# ================= REGISTER =================
@app.route("/register_user", methods=["POST"])
def register_user():
    username = request.form["username"]
    password = request.form["password"]

    conn = sqlite3.connect("users.db")  
    cur = conn.cursor()  
    cur.execute("INSERT INTO users (username,password) VALUES (?,?)",(username,password))  
    conn.commit()  
    conn.close()  

    return redirect("/")

# ================= LOGIN =================
@app.route("/login_user", methods=["POST"])
def login_user():
    username = request.form["username"]
    password = request.form["password"]

    conn = sqlite3.connect("users.db")  
    cur = conn.cursor()  
    cur.execute("SELECT * FROM users WHERE username=? AND password=?",(username,password))  
    user = cur.fetchone()  
    conn.close()  

    if user:  
        return redirect("/index")  
    else:  
        return "Invalid Login"

# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        # Clean categorical columns
        for col in ["Crop", "Season", "State"]:
            df[col] = df[col].astype(str).str.strip().str.lower()

        # ✅ USE get_dummies (FINAL FIX)
        df = pd.get_dummies(df)

        # match model columns
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)

        # prediction
        prediction = model.predict(df)[0]

        return jsonify({
            "prediction": float(prediction),
            "suitability": "Suitable" if prediction > 50 else "Not Suitable",
            "water": "High" if data["Annual_Rainfall"] < 800 else "Low",
            "pesticide": "Neem Oil"
        })

    except Exception as e:
        return jsonify({"error": str(e)})
# ================= RUN =================
if __name__ == "__main__":
    print("✅ API RUNNING...")
    app.run(debug=True)