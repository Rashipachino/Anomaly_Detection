FROM python:latest

WORKDIR /usr/app/src

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY conn_attack_iforest.py ./
COPY conn_attack.csv ./
COPY conn_attack_anomaly_labels.csv ./

CMD ["python", "./conn_attack_iforest.py"]

EXPOSE 8080
