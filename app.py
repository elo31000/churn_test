from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

with open("data/churn_model.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # data = request.get_json()

        features = np.array([
                int(request.form["Age"]),
                int(request.form["Years"]),
                int(request.form["Num_Sites"]),
                int(request.form["Account_Manager"])
            ]).reshape(1, -1)

        # Faire la prédiction
        prediction = model.predict(features)[0]

        return jsonify({"churn_prediction": int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
