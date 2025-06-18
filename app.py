from flask import Flask, request, render_template
from flask import send_from_directory
import os
import pandas as pd
import joblib
import sklearn
print(sklearn.__version__)

app = Flask(__name__, static_folder='static')

pipeline_with_preprocessor = joblib.load('full_pipeline.joblib')
preprocessor = pipeline_with_preprocessor.named_steps['preprocessor']
model = pipeline_with_preprocessor.named_steps['classifier']



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #if request.method == 'POST':
        # Get form data
        user_input = {

            'Gender': request.form['Gender'],
            'Customer Type': request.form['Customer Type'],
            'Age': float(request.form['Age']),
            'Type of Travel': request.form['Type of Travel'],
            'Class': request.form['Class'],
            'Flight Distance': float(request.form['Flight Distance']),
            'Inflight wifi service': int(request.form['Inflight wifi service']),
            'Departure/Arrival time convenient': int(request.form['Departure/Arrival time convenient']),
            'Ease of Online booking': int(request.form['Ease of Online booking']),
            'Gate location': int(request.form['Gate location']),
            'Food and drink': int(request.form['Food and drink']),
            'Online boarding': int(request.form['Online boarding']),
            'Seat comfort': int(request.form['Seat comfort']),
            'Inflight entertainment': int(request.form['Inflight entertainment']),
            'On-board service': int(request.form['On-board service']),
            'Leg room service': int(request.form['Leg room service']),
            'Baggage handling': int(request.form['Baggage handling']),
            'Checkin service': int(request.form['Checkin service']),
            'Inflight service': int(request.form['Inflight service']),
            'Cleanliness': int(request.form['Cleanliness']),
            'Departure Delay in Minutes': float(request.form['Departure Delay in Minutes']),
            'Arrival Delay in Minutes': float(request.form['Arrival Delay in Minutes'])


    }

        # Convert to DataFrame
        input_df = pd.DataFrame([user_input])
        print(input_df.head())

        # Make prediction
        preprocessed_input = preprocessor.transform(input_df)
        prediction = model.predict(preprocessed_input)[0]

        if prediction == 0:
            result = 'Neutral or dissatisfied'
        elif prediction == 1:
            result = 'Satisfied'



        # Return result
        return render_template('output.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
