from flask import Flask, request, render_template
import numpy as np
import warnings
import pickle
from feature import FeatureExtraction
from pymongo import MongoClient
from datetime import datetime

warnings.filterwarnings('ignore')

# Load the model
file = open("model.pkl", "rb")
gbc = pickle.load(file)
file.close()

app = Flask(__name__)
# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Update with your MongoDB URI if using a cloud service
db = client["phishing_detector"]  # Database name
contact_collection = db["contact_messages"]  # Collection for contact form submissions
url_collection = db["url_checks"]  # Collection for URL check results

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)
        y_pred = gbc.predict(x)[0]  # 1 is safe, -1 is unsafe
        y_pro_phishing = gbc.predict_proba(x)[0, 0]
        y_pro_non_phishing = gbc.predict_proba(x)[0, 1]

        # Store URL check result in MongoDB
        url_check = {
            "url": url,
            "prediction": int(y_pred),  # 1 for safe, -1 for unsafe
            "probability_phishing": float(y_pro_phishing),
            "probability_non_phishing": float(y_pro_non_phishing),
            "timestamp": datetime.utcnow()
        }
        url_collection.insert_one(url_check)

        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing * 100)  # optional
        return render_template('index.html', xx=round(y_pro_non_phishing, 2), url=url)
    return render_template("index.html", xx=-1)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")

        # Basic validation
        if not name or not email or not message:
            return render_template("contact.html", error="All fields are required!")

        # Store the message in MongoDB
        contact_message = {
            "name": name,
            "email": email,
            "message": message,
            "timestamp": datetime.utcnow()
        }
        contact_collection.insert_one(contact_message)

        # Render the contact page with a success message
        return render_template("contact.html", success=True)

    # Render the contact page for GET requests
    return render_template("contact.html")

@app.route("/dashboard")
def dashboard():
    # Fetch recent URL checks from MongoDB
    urls = list(url_collection.find().sort("timestamp", -1))
    
    # Fetch contact messages from MongoDB
    messages = list(contact_collection.find().sort("timestamp", -1))
    
    return render_template("dashboard.html", urls=urls, messages=messages)


if __name__ == "__main__":
    app.run(debug=True)