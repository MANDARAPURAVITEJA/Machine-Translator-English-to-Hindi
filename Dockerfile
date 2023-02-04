FROM python:3.10-alpine

LABEL maintainer="Ravi Teja Mandarapu"
LABEL version="1.0"
LABEL description="docker image of machine learning translation engine"

COPY . /usr/app/

EXPOSE 5000

WORKDIR /usr/app/

RUN pip install -r requirements.txt

CMD python app.py