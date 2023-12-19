FROM python:latest

WORKDIR /app

COPY ["requirements.txt", "./"]

RUN pip install -r requirements.txt 

VOLUME /app/models/

COPY ["streamapp.py", "./"]

CMD ["streamlit", "run", "streamapp.py"]