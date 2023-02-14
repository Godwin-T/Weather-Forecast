FROM python:latest

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock","./"]

RUN pipenv install --system --deploy


COPY ["api.py", "./"]

EXPOSE 5000

ENTRYPOINT ["waitress", "--bind = 127.0.0.1:5000", "Q6docker_predict:app"]