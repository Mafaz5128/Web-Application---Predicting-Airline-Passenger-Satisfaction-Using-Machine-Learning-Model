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
        'Type of Travel': form['TypeOfTravel'],
        'Class': form['Class'],
        'Flight Distance': float(form['FlightDistance']),
        'Inflight wifi service': int(form['InflightWifiService']),
        'Departure/Arrival time convenient': int(form['DepartureArrivalTimeConvenient']),
        'Ease of Online booking': int(form['EaseOfOnlineBooking']),
        'Gate location': int(form['GateLocation']),
        'Food and drink': int(form['FoodAndDrink']),
        'Online boarding': int(form['OnlineBoarding']),
        'Seat comfort': int(form['SeatComfort']),
        'Inflight entertainment': int(form['InflightEntertainment']),
        'On-board service': int(form['OnBoardService']),
        'Leg room service': int(form['LegRoomService']),
        'Baggage handling': int(form['BaggageHandling']),
        'Checkin service': int(form['CheckinService']),
        'Inflight service': int(form['InflightService']),
        'Cleanliness': int(form['Cleanliness']),
        'Departure Delay in Minutes': float(form['DepartureDelay']),
        'Arrival Delay in Minutes': float(form['ArrivalDelay'])
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
        customer_type=form['CustomerType'],
        travel_type=form['TypeOfTravel'],
        flight_class=form['Class'],
        age=float(form['Age']),
        flight_distance=float(form['FlightDistance']),
        departure_delay=float(form['DepartureDelay']),
        arrival_delay=float(form['ArrivalDelay']),
        inflight_wifi_service=int(form['InflightWifiService']),
        departure_arrival_time=int(form['DepartureArrivalTimeConvenient']),
        prediction=result
    )
    db.session.add(response)
    db.session.commit()

    return render_template('output.html', result=result)

# Flask App Runner
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
