import os

# Root directory
root = "neural_network_deployment"

# File structure
files = {
    "model_saver.py": "# model_saver.py (Already created)\n\n# This script saves the trained model as all_approaches_model.pkl\n",
    "api.py": """# api.py (FastAPI backend)
from fastapi import FastAPI, UploadFile
import pickle
import uvicorn

app = FastAPI()

# Load model
with open("all_approaches_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Neural Network Deployment API is running"}

# Example predict endpoint
@app.post("/predict")
async def predict(data: dict):
    # Dummy example: pass input dict to model
    # Replace with actual preprocessing and model inference
    prediction = model.predict([list(data.values())])
    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
""",
    "frontend.html": """<!-- frontend.html (Complete Web Interface) -->
<!DOCTYPE html>
<html>
<head>
    <title>Neural Network Deployment</title>
</head>
<body>
    <h1>Neural Network Deployment</h1>
    <form id="predict-form">
        <label>Input Data (comma-separated):</label><br>
        <input type="text" id="inputData" placeholder="e.g. 1.2, 3.4, 5.6"><br><br>
        <button type="submit">Predict</button>
    </form>
    <h2>Prediction Result:</h2>
    <p id="result"></p>

    <script>
        document.getElementById("predict-form").onsubmit = async function(event) {
            event.preventDefault();
            let rawInput = document.getElementById("inputData").value.split(",").map(Number);
            let data = {};
            rawInput.forEach((val, idx) => { data["feature" + idx] = val; });

            let response = await fetch("http://127.0.0.1:8000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });

            let result = await response.json();
            document.getElementById("result").innerText = JSON.stringify(result);
        }
    </script>
</body>
</html>
""",
    "requirements.txt": """fastapi
uvicorn
pickle-mixin
scikit-learn
"""
}

# Create directory if not exists
os.makedirs(root, exist_ok=True)

# Create files
for filename, content in files.items():
    filepath = os.path.join(root, filename)
    if not os.path.exists(filepath):  # Don't overwrite model_saver.py if it exists
        with open(filepath, "w") as f:
            f.write(content)

print(f"Project structure created under: {root}")
