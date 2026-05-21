# Use a lightweight Python base image
FROM python:3.9


WORKDIR /app


COPY . .


RUN pip install --upgrade pip
RUN pip install -r requirements.txt


EXPOSE 8080


CMD ["python", "index.py"]
