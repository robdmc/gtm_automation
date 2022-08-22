FROM python:3

RUN apt-get update
RUN apt-get install -y vim

WORKDIR /app

COPY requirements.txt ./
RUN pip install -U pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

