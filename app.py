from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load your trained pipeline model (preprocessing + model)
model = joblib.load("best_model.pkl")  # make sure path is correct


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    CreditScore: float = Form(...),
    Age: float = Form(...),
    Tenure: int = Form(...),
    Balance: float = Form(...),
    NumOfProducts: int = Form(...),
    HasCrCard: int = Form(...),
    IsActiveMember: int = Form(...),
    EstimatedSalary: float = Form(...),
    Geography: str = Form(...),
    Gender: str = Form(...)
):
    # Format input for your model (match training order)
    input_data = np.array([[
        CreditScore, Age, Tenure, Balance, NumOfProducts,
        HasCrCard, IsActiveMember, EstimatedSalary,
        Geography, Gender
    ]], dtype=object)

    # Make prediction
    prediction = model.predict(input_data)[0]
    result = "Customer Will Exit" if prediction == 1 else "Customer Will Stay"

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result}
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
