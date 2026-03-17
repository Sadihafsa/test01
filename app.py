from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('modele_naive_bayes.pkl')

@app.route('/')
def home():
    return "API is running 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')

        # Convert input into DataFrame (adapt if your model needs different preprocessing)
        df = pd.DataFrame([text], columns=['text'])

        prediction = model.predict(df)[0]

        return jsonify({'prediction': str(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
