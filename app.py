from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')


@app.route('/', methods=['GET'])
def form():
    return render_template('form.html')


@app.route('/', methods=['POST'])
def predict():
    input_data = []
    for i in range(1, 11):
        input_name = 'input{}'.format(i)
        input_value = request.form[input_name]
        input_data.append(float(input_value))

    input_data = np.array([input_data])
    prediction = str((model.predict_proba(input_data)[0][1]*100).round(2)) + '%'
    print(prediction)
    return render_template('prediction.html', prediction=prediction)


if __name__ == "__main__":
    app.run()
