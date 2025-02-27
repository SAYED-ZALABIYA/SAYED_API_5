from flask import Flask, request, jsonify
import joblib
import numpy as np


MODEL_PATH = "best_random_forest_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Crop Recommendation API is running!"})

#@app.route("/predict", methods=["POST"])
#def predict():
    try:


       
        if not request.is_json:
            return jsonify({"error": "Invalid request format. Please send JSON data."}), 400

        print(request.data)  

        data = request.get_json(force=True)


#data = request.get_json()
        
       
        features = np.array([data["nitrogen"], data["phosphorus"], data["potassium"], data["ph"], data["rainfall"]]).reshape(1, -1)
        
        
        prediction = model.predict(features)
        recommended_crop = int(prediction[0])  
        
        return jsonify({"recommended_crop": recommended_crop})
    except Exception as e:
        return jsonify({"error": str(e)})
#@app.route("/predict", methods=["POST"])
#def predict():
    try:
        
        print("Headers:", request.headers)
        
        
        print("Raw Data:", request.data)

        
        if not request.is_json:
            return jsonify({"error": "Invalid request format. Please send JSON data."}), 400

       
        data = request.get_json()
        print("Parsed JSON Data:", data)

       
        features = np.array([data["nitrogen"], data["phosphorus"], data["potassium"], data["ph"], data["rainfall"]]).reshape(1, -1)
        
        
        prediction = model.predict(features)
        recommended_crop = int(prediction[0])  
        
        return jsonify({"recommended_crop": recommended_crop})
    except Exception as e:
        return jsonify({"error": str(e)})
#@app.route("/predict", methods=["POST"])
#def predict():
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid request format. Please send JSON data."}), 400

        data = request.get_json()
        print("Received data:", data)  

        # استخراج القيم
        features = np.array([
            data["nitrogen"], data["phosphorus"], data["potassium"],
            data["temperature"], data["humidity"], data["ph"], data["rainfall"]
        ]).reshape(1, -1)

        print("Prepared features:", features)  

        
        prediction = model.predict(features)
        recommended_crop = int(prediction[0])  

        return jsonify({"recommended_crop": recommended_crop})

    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid request format. Please send JSON data."}), 400

        data = request.get_json()
        print("Received data:", data)  
        features = np.array([
            data["Nitrogen (kg/ha )"], data["Phosphorus (kg/ha )"], data["Potassium (kg/ha )"],
            data["Temperature"], data["Humidity"], data["pH_Value"], data["Rainfall"]
        ]).reshape(1, -1)

        print("Prepared features:", features) 

        features = scaler.transform(features) 

        
        #prediction = model.predict(features)
        #recommended_crop = int(prediction[0])  

        probabilities = model.predict_proba(features)  
        predicted_class = model.predict(features)[0]  
        confidence_score = np.max(probabilities)


        recommended_crop = encoder.inverse_transform([predicted_class])[0]


        #recommended_crop = encoder.inverse_transform([prediction[0]])[0]


        return jsonify({
            "recommended_crop": recommended_crop,
            "confidence_score": round(float(confidence_score) * 100, 2)  
        })

        #return jsonify({"recommended_crop": recommended_crop})

    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
