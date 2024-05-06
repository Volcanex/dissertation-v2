FROM rocm/pytorch:latest

RUN pip install -r requirements.txt

WORKDIR /app

COPY model.py /app/model.py

COPY 2.0data /app/2.0data

COPY test.wav /app/test.wav

CMD ["python", "model.py"]             