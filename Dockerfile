FROM tensorflow/tensorflow:1.14.0-gpu-py3
RUN apt update && apt install -y git nano ncdu locales && pip install --upgrade pip
RUN git clone https://github.com/EdwardBurgin/polyai-models.git 
WORKDIR /polyai-models
RUN pip install -r requirements.txt
RUN pip install -r intent_detection/requirements.txt
