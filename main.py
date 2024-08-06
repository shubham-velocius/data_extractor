from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import fitz
from openai import AsyncAzureOpenAI
import re
import os
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

GPT_4_TURBO = "gpt-4-turbo"
GPT_4_OMNI = "gpt-4o"

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_VERSION= os.getenv('AZURE_OPENAI_API_VERSION')
AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT')
#openai.api_key = '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'

class ExtractedData(BaseModel):
    cardholder_name: str
    date_range: str
    total_billing_amount: str
    transactions: list

def extract_text_from_pdf(pdf_file: UploadFile):
    pdf_document = fitz.open(stream=pdf_file.file.read(), filetype="pdf")
    print(pdf_document)
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

async def extract_fields_from_text(text: str):

    prompt = (
        "Extract the following fields from the text of a credit card statement:\n"
        "- Cardholder Name\n"
        "- Date range for transactions\n"
        "- Total billing amount\n"
        "- Transaction details (for all transactions in the statement): Transaction Date, Description, Billing Amount\n"
        f"Text: {text}\n"
        "Please return the data in JSON format."
    )
    client = AsyncAzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        max_retries=2,
    )

    response = await client.chat.completions.create(
        model=GPT_4_OMNI,
        response_format={"type": "json_object"},
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a highly accurate assistant skilled at extracting credit card transaction from provided text data."},
            {"role": "user", "content": prompt},
            # {"role": "user", "content":"If you are unable to identify mortgage document with book 430 and page 402, give reasons for doing so, guidelines taken into consideration, additional instructions/guidelines required in additional json fields in the json output"}
        ]
    )

    openai_response = response.choices[0].message.content

    return openai_response

@app.post("/hello")
async def hello_user(name):
    return f"Hello {name}"

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    text = extract_text_from_pdf(file)
    extracted_data = await extract_fields_from_text(text)
    return {"extracted_data": extracted_data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
