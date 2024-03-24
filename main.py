from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np
from preprocess import lemmatize_sentence


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("countVectorizer.pkl", "rb") as f:
    cv = pickle.load(f)

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):

    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_sentiment(request: Request, text: str = Form(...)):

    token = lemmatize_sentence(text)
    sentence = " ".join(token)
    sentence = cv.transform([sentence]).toarray()
    sentiment = model.predict(sentence)

    return templates.TemplateResponse(
        "result.html", {"request": request, "sentiment": sentiment[0]}
    )


# To run the application on localhost, after executing main.py, enter this code into the terminal.

# uvicorn main:app --reload
