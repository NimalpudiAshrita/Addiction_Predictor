
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "digital_addiction_dataset.csv")

data = pd.read_csv(DATA_PATH)

le = LabelEncoder()
data["addiction_level"] = le.fit_transform(data["addiction_level"])

X = data.drop("addiction_level", axis=1)
y = data["addiction_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    suggestion = ""

    if request.method == "POST":
        user_data = [[
            float(request.form["screen_time"]),
            float(request.form["social_time"]),
            float(request.form["gaming_time"]),
            float(request.form["work_time"]),
            float(request.form["sleep_hours"]),
            int(request.form["night_usage"])
        ]]

        prediction = model.predict(user_data)
        risk = le.inverse_transform(prediction)[0]
        result = risk

        if risk == "High":
            suggestion = "Reduce screen time, avoid phone at night, sleep 8 hours."
        elif risk == "Medium":
            suggestion = "Limit social media and take regular breaks."
        else:
            suggestion = "Good balance! Maintain healthy habits."

    return render_template("index.html", result=result, suggestion=suggestion)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
