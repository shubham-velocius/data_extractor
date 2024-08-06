from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import fitz  # PyMuPDF
from openai import AsyncAzureOpenAI
import re
import os
import json
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_bytes

app = FastAPI()

load_dotenv()

GPT_4_TURBO = "gpt-4-turbo"
GPT_4_OMNI = "gpt-4o"

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT')

class ExtractedData(BaseModel):
    cardholder_name: str
    date_range: str
    total_billing_amount: str
    transactions: list

def extract_text_from_pdf(pdf_file: UploadFile):
    try:
        file_data = pdf_file.file.read()
        pdf_document = fitz.open(stream=file_data, filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text("text")
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")

def ocr_pdf(pdf_file: UploadFile):
    try:
        file_data = pdf_file.file.read()
        if not file_data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        images = convert_from_bytes(file_data)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image)
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during OCR: {str(e)}")

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
            {"role": "system", "content": "You are a highly accurate assistant skilled at extracting credit card transaction details from provided text data."},
            {"role": "user", "content": prompt},
        ]
    )

    openai_response = response.choices[0].message.content

    return json.loads(openai_response)

@app.post("/hello")
async def hello_user(name):
    return f"Hello {name}"

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Reset the file pointer before reading
        file.file.seek(0)
        text = extract_text_from_pdf(file)
        
        if not text.strip():  # If no text is extracted, use OCR
            # Reset the file pointer before reading for OCR
            file.file.seek(0)
            text = ocr_pdf(file)
        
        extracted_data = await extract_fields_from_text(text)
        return {"extracted_data": extracted_data}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the PDF: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
