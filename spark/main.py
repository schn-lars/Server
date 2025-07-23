from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import sparknlp
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import MultiDateMatcher
from pyspark.ml import Pipeline
from starlette.responses import JSONResponse

app = FastAPI()

spark = sparknlp.start()

document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
multi_date = MultiDateMatcher() \
    .setInputCols(["document"]) \
    .setOutputCol("multi_date") \
    .setOutputFormat("dd/MM/yyyy") \
    .setSourceLanguage("de")
pipeline = Pipeline().setStages([document_assembler, multi_date])

# Input model
class TextInput(BaseModel):
    text: str

@app.post("/extractdates")
async def extract_dates(data: TextInput):
    try:
        print("Extracting dates...")
        df = spark.createDataFrame([[data.text]], ["text"])
        result = pipeline.fit(df).transform(df)
        dates = result.select("multi_date.result").rdd.map(lambda row: row[0]).collect()
        return JSONResponse(content={"dates": dates}, status_code=200)
    except Exception as e:
        print(f"Error in extracting dates: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)