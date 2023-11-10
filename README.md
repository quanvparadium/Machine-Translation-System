# Machine-Translation-System
---
A Simple Machine Translation System that utilizes a pre-trained model from HuggingFace, specifically the mBART-50 model.
## To do task
- [x] [User Interface](https://github.com/quanvparadium/Machine-Translation-System/blob/main/templates)
- [x] [Ruled-based Machine Translation](https://github.com/quanvparadium/Machine-Translation-System/blob/main/refs/urban.py)
- [x] [Statistical Machine Translation](https://github.com/quanvparadium/Machine-Translation-System/blob/main/smt)
- [x] [Neural Machine Translation]()
- [ ] [Deploy Model with FastAPI, Docker and Heroku](https://github.com/quanvparadium/Machine-Translation-System/blob/main/app.py)

# 1. Usage
---
## Requirements
If you do not install Anaconda for Python, our app may still work, but package installation could be more difficult.
- **Conda**: 

    ```bash
    $ conda create -n env python=3.10 anaconda
    $ conda activate env
    ```
    To leave the environment after you no longer need it: 
    ```bash
    $ conda deactivate
    ```
- **Mac/Linux Users**:
    ```bash
    $ python -m pip install --user --upgrade pip
    $ python -m pip install --user virtualenv
    $ python -m venv venv
    ```
    To activate the virtual environment:
    ```bash
    source venv/bin/activate
    ```
    To leave the environment after you no longer need it: 
    ```bash
    $ deactivate
    ```

---
## Dependencies

- All of the dependencies are listed in `requirements.txt`. To reduce the likelihood of environment errors, install the dependencies inside a virtual environment with the following steps.
    ```bash
	    $ pip install -r requirements.txt
    ```
- **requirements.txt**
    ```txt
    numpy
    pandas
    matplotlib
    torch
    tqdm==4.65.0

    # Rule-based Machine Translation
    urbans

    # Statistical Machine Translatation
    bs4
    nltk==3.8.1
    wrapt

    mBART - Neural Machine Translation
    fsspec==2023.9.2
    datasets==2.14.6
    sentencepiece==0.1.97 
    sacrebleu==2.3.1
    transformers==4.26.1
    protobuf==3.20.1

    #Deployment
    fastapi
    pydantic
    flask
    uvicorn
    ```

## Config
Run the following commands in terminal:
```bash
    python train.py --model_name 'Transformer' 
        --device 'cpu'
        --model_type 'unigram' 
        --src_lang 'vi' 
        --tgt_lang 'en' 
        --num_heads 8 
        --num_layers 6 
        --d_model 512 
        --d_ff 2048 
        --drop_out 0.1 
        --seq_len 150    
        --batch_size 16
```

Or
```bash
    bash bash/transformer.sh
```

To run our machine translation system, you can run the following commands in terminal:
```bash
    python app.py
```


# 2. Ruled-based Machine Translation
---
We use **Translator** built from the **urbans** library.
For more details about **URBANS**, you can visit the [URBANS GitHub page](https://github.com/pyurbans/urbans).

# 3. Statistical Machine Translation
---

# 4. Neural Machine Translation
---
## Dataset

The *IWSLT'15 English-Vietnamese* data is used from [Stanford NLP group](https://nlp.stanford.edu/projects/nmt/).

For all experiments the corpus was split into training, development and test set:

| Data set    | Sentences | Download
| ----------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------
| Training    | 133,317   | via [GitHub](https://github.com/stefan-it/nmt-en-vi/raw/master/data/train-en-vi.tgz) or located in `data/train.en` or `data/train.vi`
| Development |   1,553   | via [GitHub](https://github.com/stefan-it/nmt-en-vi/raw/master/data/dev-2012-en-vi.tgz) or located in `data/validation.en` or `data/validation.vi`
| Test        |   1,268   | via [GitHub](https://github.com/stefan-it/nmt-en-vi/raw/master/data/test-2013-en-vi.tgz) or located in `data/test.en` or `data/test.vi`

## Beam search


# 5. Deploy Model with FastAPI, Docker and Heroku
- We utilize **FastAPI** ([docs](https://fastapi.tiangolo.com/deployment/)) for deploying our machine translation model. **FastAPI** is a powerful web framework that has gained popularity for several compelling reasons: 
    - **High Performance**
    - **Automatic API Documentation** 
    - **Type Annotations and Validation** 
    - **Asynchronous Support** 
    - **Dependency Injection System** 
    - **Security Features** 
    - **Easy Integration with Pydantic Models**

```python
STATIC_FILES_DIR = './templates/'
app = FastAPI()

class InputModel(BaseModel):
    sentence: str
    grammar: str
    method: int

@app.get('/')
async def home():
    return FileResponse(
        STATIC_FILES_DIR + "home.html",
        headers={"Cache-Control": "no-cache"}
    )

@app.post('/predict')
def predict(input: InputModel):
    if input.method == 0: # Rule-based Machine Translation
        translator = Translator(src_grammar = process_grammar(input.grammar),
                        src_to_tgt_grammar = src_to_target_grammar,
                        src_to_tgt_dictionary = en_to_vi_dict)
        language = translator.translate(input.sentence) 
    elif input.method == 1: # Statistical Machine Translation
        language = smt_translate_vi2en(input.sentence)
    elif input.method == 2: # Neural Machine Translation
        language = translate_vi2en(input.sentence)
    return {
        "output": language
    }
app.mount("/", StaticFiles(directory=STATIC_FILES_DIR, html=True))
```


# References
---
- Official repositories:
    - VinAI Research [BARTpho](https://github.com/VinAIResearch/BARTpho): Pre-trained Sequence-to-Sequence Models for Vietnamese) ([paper](https://arxiv.org/pdf/2109.09701.pdf)) 
    - [Pytorch Beam Search](https://github.com/jarobyte91/pytorch_beam_search)
    - [Pytorch Documentation](https://pytorch.org/docs/stable/)

- Tutorial:
    - A [Github Gist](https://github.com/t4train/t4train/blob/master/readme_assets/setup-README.md) explaining how to setup README.md properly
    - [FastAPI in Containers - Docker](https://fastapi.tiangolo.com/deployment/docker/)
    - [Heroku: Deploying with Git](https://devcenter.heroku.com/articles/git)

- Colaboratory:
    - [Statistical Machine Translation](https://github.com/sayarghoshroy/Statistical-Machine-Translation/blob/master/SMT_English_to_Hindi.ipynb): English to Hindi