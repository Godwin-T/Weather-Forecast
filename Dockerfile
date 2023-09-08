FROM python:latest

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock","./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "weather.bin", "./"]

EXPOSE 5000

ENTRYPOINT ["waitress", "--bind = 127.0.0.1:5000", "predict:app"]