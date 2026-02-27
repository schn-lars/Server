FROM eclipse-temurin:8-jdk-jammy

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean

RUN pip3 install --no-cache-dir \
    pyspark==3.2.0 \
    spark-nlp \
    fastapi \
    numpy \
    uvicorn[standard] \
    pydantic

WORKDIR /app
COPY main.py .

EXPOSE 5050
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5050"]