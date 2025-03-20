FROM python:3.13.2-alpine

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8090

CMD ["python3", "app.py"]