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

All of the dependencies are listed in `requirements.txt`. To reduce the likelihood of environment errors, install the dependencies inside a virtual environment with the following steps.
```
	$ pip install -r requirements.txt
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
We use the [iwslt2015-en-vi](https://huggingface.co/datasets/mt_eng_vietnamese/viewer/iwslt2015-en-vi) (136k rows) dataset to train the **transformer** and **mBART50** models. The dataset consists of 133k training samples, 1.27k validation samples, and 1.27k test samples.
# 5. Deploy Model with FastAPI, Docker and Heroku
- We utilize **FastAPI** ([docs](https://fastapi.tiangolo.com/deployment/)) for deploying our machine translation model. **FastAPI** is a powerful web framework that has gained popularity for several compelling reasons: 
    - **High Performance**
    - **Automatic API Documentation** 
    - **Type Annotations and Validation** 
    - **Asynchronous Support** 
    - **Dependency Injection System** 
    - **Security Features** 
    - **Easy Integration with Pydantic Models**

# References
---
- Official repositories:
    - VinAI Research [BARTpho](https://github.com/VinAIResearch/BARTpho): Pre-trained Sequence-to-Sequence Models for Vietnamese) ([paper](https://arxiv.org/pdf/2109.09701.pdf)) 
    - Pytorch Beam Search: [github](https://github.com/jarobyte91/pytorch_beam_search)

- Tutorial:
    - A [Github Gist](https://github.com/t4train/t4train/blob/master/readme_assets/setup-README.md) explaining how to setup README.md properly
    - Docs []https://fastapi.tiangolo.com/deployment/docker/