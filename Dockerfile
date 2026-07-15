FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "bank_churn_api:app", "--host", "0.0.0.0", "--port", "8000"]