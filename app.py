from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import joblib
import os

app = Flask(__name__, static_folder='static')

# PostgreSQL Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')  # e.g. 'postgresql://username:password@host:port/dbname'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database model
class SurveyResponse(db.Model):
    __tablename__ = 'survey_responses'
    id = db.Column(db.Integer, primary_key=True)  # You were missing the primary key
    passenger_name = db.Column(db.String(100))
    flight_date = db.Column(db.String(100))
    origin = db.Column(db.String(50))
    destination = db.Column(db.String(50))
    gender = db.Column(db.String(20))
    customer_type = db.Column(db.String(50))
    travel_type = db.Column(db.String(50))
    flight_class = db.Column(db.String(50))
    age = db.Column(db.Float)
    flight_distance = db.Column(db.Float)
    departure_delay = db.Column(db.Float)
    arrival_delay = db.Column(db.Float)
    inflight_wifi_service = db.Column(db.Integer)
    departure_arrival_time = db.Column(db.Integer)
    prediction = db.Column(db.String(50))

with app.app_context():
    db.create_all()

# Load the ML pipeline
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

    # Extract and format user input for model prediction
    user_input = {
        'Gender': form['Gender'],
        'Customer Type': form['CustomerType'],
        'Age': float(form['Age']),
        'Type of Travel': form['TypeofTravel'],
        'Class': form['Class'],
        'Flight Distance': float(form['FlightDistance']),
        'Inflight wifi service': int(form['Inflightwifiservice']),
        'Departure/Arrival time convenient': int(form['Departure/Arrival time convenient']),
        'Ease of Online booking': int(form['EaseofOnline booking']),
        'Gate location': int(form['Gatelocation']),
        'Food and drink': int(form['Foodanddrink']),
        'Online boarding': int(form['Online boarding']),
        'Seat comfort': int(form['Seatcomfort']),
        'Inflight entertainment': int(form['Inflightentertainment']),
        'On-board service': int(form['On-boardservice']),
        'Leg room service': int(form['Leg room service']),
        'Baggage handling': int(form['Baggage handling']),
        'Checkin service': int(form['Checkin service']),
        'Inflight service': int(form['Inflight service']),
        'Cleanliness': int(form['Cleanliness']),
        'Departure Delay in Minutes': float(form['Departure Delay in Minutes']),
        'Arrival Delay in Minutes': float(form['Arrival Delay in Minutes'])
    }

    # Perform prediction
    input_df = pd.DataFrame([user_input])
    preprocessed_input = preprocessor.transform(input_df)
    prediction = model.predict(preprocessed_input)[0]
    result = 'Satisfied' if prediction == 1 else 'Neutral or dissatisfied'

    # Store to DB
    response = SurveyResponse(
        passenger_name=form['PassengerName'],
        flight_date=form['Date'],
        origin=form['Origin'],
        destination=form['Destination'],
        gender=form['Gender'],
        customer_type=form['Customer Type'],
        travel_type=form['Type of Travel'],
        flight_class=form['Class'],
        age=float(form['Age']),
        flight_distance=float(form['Flight Distance']),
        departure_delay=float(form['Departure Delay in Minutes']),
        arrival_delay=float(form['Arrival Delay in Minutes']),
        inflight_wifi_service=int(form['Inflight wifi service']),
        departure_arrival_time=int(form['Departure/Arrival time convenient']),
        prediction=result
    )
    db.session.add(response)
    db.session.commit()

    return render_template('output.html', result=result)

# Flask App Runner
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
