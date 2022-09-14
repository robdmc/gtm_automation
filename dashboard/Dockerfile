FROM python:3

RUN apt-get update
# RUN apt-get install -y vim
RUN apt-get install -y locales locales-all
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

WORKDIR /app

#COPY requirements.txt ./
COPY . .
RUN pip install -U pip
RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

