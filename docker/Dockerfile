FROM python:3.9

# python
ENV PYENV_ROOT /home/docker/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
ARG PYTHON_VERSION
RUN pyenv install ${PYTHON_VERSION} && pyenv global ${PYTHON_VERSION}

# torch
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
RUN pip install torch==${TORCH_VERSION} -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchvision==${TORCHVISION_VERSION}

# requirements

RUN pip install -U pip setuptools
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# path
ENV PYTHONPATH $PYTHONPATH:/work
ENV PYTHONPATH $PYTHONPATH:/home/docker/

WORKDIR /work

ENTRYPOINT ["/bin/bash"]