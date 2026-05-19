print("App starting...")
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news']
    data = vectorizer.transform([text])
    prediction = model.predict(data)

    result = "Real News" if prediction[0] == 1 else "Fake News"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    print("Running Flask...")
    app.run(debug=True)