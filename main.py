import uvicorn
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import shutil
import os
import io

from model_utils_tatarnikov import load_model, load_data, preprocess_data, make_prediction

app = FastAPI()

# Шаблоны и статика
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

model = load_model()

@app.get("/", response_class=HTMLResponse, summary="Главная", tags=["Интерфейс"])
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/", summary="Предсказать", tags=["Интерфейс"])
def predict(
    file: UploadFile = File(...),
    response_format: str = Form("csv")
):

    # Сохраняем загруженный файл
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Загружаем и обрабатываем данные
    data = load_data(file_path)
    data = preprocess_data(data)

    # Предсказание и сохранение
    output_df = make_prediction(data, model)
    if response_format == "json":
        result_json = output_df.to_dict(orient="records")
        return JSONResponse(content={"result": result_json})

    else:
        # Формируем CSV в памяти
        output_csv = io.StringIO()
        output_df.to_csv(output_csv, index=False)
        output_csv.seek(0)

        # Возвращаем CSV как файл для скачивания
        return StreamingResponse(
            output_csv,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=tatarnikov_results.csv"})

@app.post("/show_json/", response_class=HTMLResponse)
async def show_json(request: Request, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    data = load_data(file_path)
    data = preprocess_data(data)
    output_df = make_prediction(data, model)
    result_json = output_df.to_dict(orient="records")

    return templates.TemplateResponse(
        "json_result.html",
        {"request": request, "result_json": result_json}
    )

if __name__ == "__main__":
    #uvicorn.run("main:app", host="127.0.0.1")
    uvicorn.run("main:app", host="0.0.0.0", port=8000)