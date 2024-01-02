FROM python:3.11

WORKDIR /app

COPY ./config.py .
COPY ./gui.py .
COPY ./requirements.txt .

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "gui.py", "--server.port", "80"]
