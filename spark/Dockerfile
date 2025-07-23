ARG IMAGE_VARIANT=slim-buster
ARG OPENJDK_VERSION=8
ARG PYTHON_VERSION=3.9.8

FROM python:${PYTHON_VERSION}-${IMAGE_VARIANT} AS py3
FROM openjdk:${OPENJDK_VERSION}-${IMAGE_VARIANT}

COPY --from=py3 / /

ARG PYSPARK_VERSION=3.2.0

RUN pip install --no-cache-dir pyspark==${PYSPARK_VERSION} \
    spark-nlp \
    fastapi \
    numpy \
    uvicorn[standard] \
    pydantic

COPY main.py /app/main.py
WORKDIR /app

EXPOSE 5050
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5050"]