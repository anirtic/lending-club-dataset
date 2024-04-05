FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
EXPOSE 8000
RUN useradd -m myuser
USER myuser
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]