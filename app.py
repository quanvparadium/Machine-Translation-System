from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import FileResponse 
from fastapi.staticfiles import StaticFiles

from urbans import Translator

src_to_target_grammar =  {
    "NP -> JJ NN": "NP -> NN JJ" # in Vietnamese NN goes before JJ
}

en_to_vi_dict = {
    "I":"tôi",
    "love":"yêu",
    "hate":"ghét",
    "dogs":"những chú_chó",
    "good":"ngoan",
    "bad":"hư"
    }
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en", src_lang="vi_VN")
# model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en")
# device_vi2en = torch.device("cpu")
# model_vi2en.to(device_vi2en)
# def translate_vi2en(vi_texts: str) -> str:
#     input_ids = tokenizer_vi2en(vi_texts, padding=True, return_tensors="pt").to(device_vi2en)
#     output_ids = model_vi2en.generate(
#         **input_ids,
#         decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
#         num_return_sequences=1,
#         num_beams=5,
#         early_stopping=True
#     )
#     en_texts = tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
#     return en_texts



def translate_vi2en(vi_texts: str) -> str:
    return "STRING: " + vi_texts

def process_grammar(grammar: str) -> str:
    grammarLst = grammar.strip().split('\n')
    print(grammarLst)
    grammarLst_new = list(map(lambda x: x[4:], grammarLst))
    output = '\n'.join(grammarLst_new)
    print(output)
    return output

STATIC_FILES_DIR = './templates/'
app = FastAPI()

class InputModel(BaseModel):
    sentence: str
    grammar: str
    method: int

@app.get('/')
async def home():
    # return FileResponse('home.html')
    return FileResponse(
        STATIC_FILES_DIR + "home.html",
        headers={"Cache-Control": "no-cache"}
    )


@app.post('/predict')
def predict(input: InputModel):
    print("INPUT SENTENCE: ", input.sentence)
    print("INPUT METHOD:", input.method)

    if input.method == 0:
        translator = Translator(src_grammar = process_grammar(input.grammar),
                        src_to_tgt_grammar = src_to_target_grammar,
                        src_to_tgt_dictionary = en_to_vi_dict)
        language = translator.translate(input.sentence) 
    elif input.method == 1:
        language = "METHOD Statistical" + input.sentence
    elif input.method == 2:
        language = translate_vi2en(input.sentence)
    return {
        "output": language
    }

app.mount("/", StaticFiles(directory=STATIC_FILES_DIR, html=True))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
