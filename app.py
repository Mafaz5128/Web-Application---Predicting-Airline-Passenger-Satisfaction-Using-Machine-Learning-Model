from flask import Flask, request, render_template
from flask_mysqldb import MySQL
import pandas as pd
import joblib
import os

app = Flask(__name__, static_folder='static')

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'  # replace with your actual MySQL user
app.config['MYSQL_PASSWORD'] = 'mfz4u_6221616'  # replace with your actual password
app.config['MYSQL_DB'] = 'happywings'

mysql = MySQL(app)

# Load ML pipeline
pipeline_with_preprocessor = joblib.load('full_pipeline.joblib')
preprocessor = pipeline_with_preprocessor.named_steps['preprocessor']
model = pipeline_with_preprocessor.named_steps['classifier']

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form = request.form
    cursor = mysql.connection.cursor()

    user_input = {
        'Gender': form['Gender'],
        'Customer Type': form['Customer Type'],
        'Age': float(form['Age']),
        'Type of Travel': form['Type of Travel'],
        'Class': form['Class'],
        'Flight Distance': float(form['Flight Distance']),
        'Inflight wifi service': int(form['Inflight wifi service']),
        'Departure/Arrival time convenient': int(form['Departure/Arrival time convenient']),
        'Ease of Online booking': int(form['Ease of Online booking']),
        'Gate location': int(form['Gate location']),
        'Food and drink': int(form['Food and drink']),
        'Online boarding': int(form['Online boarding']),
        'Seat comfort': int(form['Seat comfort']),
        'Inflight entertainment': int(form['Inflight entertainment']),
        'On-board service': int(form['On-board service']),
        'Leg room service': int(form['Leg room service']),
        'Baggage handling': int(form['Baggage handling']),
        'Checkin service': int(form['Checkin service']),
        'Inflight service': int(form['Inflight service']),
        'Cleanliness': int(form['Cleanliness']),
        'Departure Delay in Minutes': float(form['Departure Delay in Minutes']),
        'Arrival Delay in Minutes': float(form['Arrival Delay in Minutes'])
    }

    # Prediction
    input_df = pd.DataFrame([user_input])
    preprocessed_input = preprocessor.transform(input_df)
    prediction = model.predict(preprocessed_input)[0]
    result = 'Satisfied' if prediction == 1 else 'Neutral or dissatisfied'

    # Save to MySQL
    cursor.execute("""
        INSERT INTO survey_responses (
            passenger_name, flight_date, origin, destination, gender, customer_type, travel_type,
            flight_class, age, flight_distance, departure_delay, arrival_delay,
            inflight_wifi_service, departure_arrival_time, online_booking, gate_location,
            food_and_drink, online_boarding, seat_comfort, inflight_entertainment,
            onboard_service, leg_room, baggage_handling, checkin_service,
            inflight_service, cleanliness, prediction
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        form['PassengerName'], form['Date'], form['Origin'], form['Destination'],
        form['Gender'], form['Customer Type'], form['Type of Travel'], form['Class'],
        form['Age'], form['Flight Distance'], form['Departure Delay in Minutes'],
        form['Arrival Delay in Minutes'], form['Inflight wifi service'],
        form['Departure/Arrival time convenient'], form['Ease of Online booking'],
        form['Gate location'], form['Food and drink'], form['Online boarding'],
        form['Seat comfort'], form['Inflight entertainment'], form['On-board service'],
        form['Leg room service'], form['Baggage handling'], form['Checkin service'],
        form['Inflight service'], form['Cleanliness'], result
    ))

    mysql.connection.commit()
    cursor.close()

    return render_template('output.html', result=result)

# Flask App Runner
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
