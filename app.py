from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse 
from pydantic import BaseModel
from inference import translate_vi2en

STATIC_FILES_DIR = './templates/'
app = FastAPI()

class InputModel(BaseModel):
    sentence: str

class OutputModel(BaseModel):
    language: str

@app.get("/", response_class=FileResponse)
async def home():
    # return FileResponse('home.html')
    return FileResponse(
        STATIC_FILES_DIR + "home.html",
        headers={"Cache-Control": "no-cache"}
    )

app.mount("/", StaticFiles(directory=STATIC_FILES_DIR), name="templates")


@app.get('/predict', response_model= OutputModel)
async def predict(input: InputModel):
    language = translate_vi2en(input.sentence)
    print("INPUT: ", input.sentence)
    print("RESULT: ", language)
    return {
        "target_text": language
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
    



