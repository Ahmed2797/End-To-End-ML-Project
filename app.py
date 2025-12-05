from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from project.utils import load_object
from project.entity.estimator import ProjectModel

app = FastAPI()
templates = Jinja2Templates(directory="project/templates")


# its already done in when final_model was save
# # # Load preprocessor and trained model
# # preprocessor = load_object("final_model/preprocessing.pkl")
# # # Wrap them in ProjectModel
# # predict_pipeline = ProjectModel(transform_object=preprocessor, best_model_details=model)

model = load_object("final_model/best_model.pkl")


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
    try:
        # Collect input data as a dict
        # Convert all form inputs to proper Python types
        input_data = {
            "CreditScore": float(CreditScore),
            "Age": float(Age),
            "Tenure": int(Tenure),
            "Balance": float(Balance),
            "NumOfProducts": int(NumOfProducts),
            "HasCrCard": int(HasCrCard),
            "IsActiveMember": int(IsActiveMember),
            "EstimatedSalary": float(EstimatedSalary),
            "Geography": str(Geography),
            "Gender": str(Gender)
        }

        print(type(input_data))
        print(input_data)

        # Use ProjectModel, which can accept dict directly
        prediction_df = model.predict(input_data)
        prediction = prediction_df['prediction'].iloc[0]

        result = "Customer Will Exit" if prediction == 1 else "Customer Will Stay"

    except Exception as e:
        result = f"Error: {e}"

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
