from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
import pandas as pd
from .model import MachineLearningModel
from .schemas import PredictionInput

app = FastAPI(
    title="Manufacturing Predictor API",
    description="An API for predicting machine downtime based on operating parameters",
    version="1.0.0"
)

ml_model = MachineLearningModel()

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Manufacturing Predictor API</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5e4d7;
            color: #333;
            line-height: 1.5;
            animation: fadeIn 1.2s ease-in;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }

        header {
            text-align: center;
            padding: 20px 10px;
            border-bottom: 1px solid #eaeaea;
            animation: slideDown 1.5s ease;
        }

        @keyframes slideDown {
            from {
                transform: translateY(-50px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }

        header h1 {
            margin: 0;
            font-size: 2em;
            font-weight: 500;
            color: #111;
        }

        .container {
            max-width: 600px;
            margin: 30px auto;
            padding: 20px;
            background-color: #f9f9f9;
            border: 1px solid #eaeaea;
            border-radius: 8px;
            box-shadow: 2px 3px 4px #000000;
            animation: scaleUp 0.8s ease-out;
        }

        @keyframes scaleUp {
            from {
                transform: scale(0.9);
                opacity: 0.8;
            }
            to {
                transform: scale(1);
                opacity: 1;
            }
        }

        h2 {
            font-size: 1.5em;
            font-weight: 500;
            margin-bottom: 10px;
            color: #111;
        }

        a {
            text-decoration: none;
            color: #007BFF;
            transition: color 0.2s ease;
        }

        a:hover {
            color: #0056b3;
        }

        form {
            display: grid;
            gap: 15px;
            margin-top: 15px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.9em;
            font-weight: 500;
            margin-bottom: 5px;
            color: #555;
        }

        input[type="text"],
        input[type="file"] {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 1em;
            color: #333;
            transition: box-shadow 0.2s ease;
        }

        input[type="text"]:focus,
        input[type="file"]:focus {
            box-shadow: 0 0 5px #007BFF;
        }

        input[type="submit"] {
            padding: 10px;
            background-color: #d4a373;
            color: #333;
            border: 1px solid #ddd;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 500;
            transition: background-color 0.2s ease, transform 0.2s ease;
        }

        input[type="submit"]:hover {
            background-color: #d6f0f0;
            transform: translateY(-2px);
            box-shadow: 0 5px 10px rgba(0, 0, 0, 0.1);
        }

        footer {
            text-align: center;
            margin-top: 30px;
            font-size: 0.85em;
            color: #666;
            animation: fadeIn 2s ease;
        }

        footer a {
            color: #007BFF;
            text-decoration: none;
            transition: color 0.2s ease, transform 0.2s ease;
        }

        footer a:hover {
            text-decoration: underline;
            transform: scale(1.1);
        }
    </style>
</head>
<body>
    <header>
        <h1>Manufacturing Predictor API</h1>
    </header>

    <div class="container">
        <h2>Upload CSV Data</h2>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label for="file">Choose a CSV file:</label>
                <input type="file" id="file" name="file" accept=".csv" required>
            </div>
            <input type="submit" value="Upload CSV">
        </form>
    </div>

    <div class="container">
        <h2>Train Model</h2>
        <form action="/train" method="post">
            <input type="submit" value="Train Model">
        </form>
    </div>

    <div class="container">
        <h2>Predict Downtime</h2>
        <form action="/predict" method="post">
            <div class="form-group">
                <label for="temperature">Temperature:</label>
                <input type="text" id="temperature" name="temperature" required>
            </div>
            <div class="form-group">
                <label for="run_time">Run Time:</label>
                <input type="text" id="run_time" name="run_time" required>
            </div>
            <input type="submit" value="Predict">
        </form>
    </div>

    <footer>
        <p>Need help? <a href="/help">Visit our Help Center</a>.</p>
    </footer>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")
    
    try:
        df = pd.read_csv(file.file)
        ml_model.store_data(df)
        return JSONResponse(content={"message": "Data uploaded successfully"})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train")
async def train_model():
    try:
        metrics = ml_model.train()
        return JSONResponse(content=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(
    temperature: float = Form(...),
    run_time: float = Form(...)
):
    try:
        input_data = PredictionInput(Temperature=temperature, Run_Time=run_time)
        prediction = ml_model.predict(input_data)
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
