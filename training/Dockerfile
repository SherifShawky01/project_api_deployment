FROM continuumio/miniconda3

RUN mkdir /training
WORKDIR /training

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV MLFLOW_TRACKING_URI=http://mlflow:5000

CMD ["./entrypoint.sh"]