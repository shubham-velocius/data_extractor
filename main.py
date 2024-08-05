from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import fitz  # PyMuPDF
import openai
import re
import os

app = FastAPI()

openai.api_key = os.getenv("API_KEY")

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

def extract_fields_from_text(text: str):
    prompt = (
        "Extract the following fields from the text of a credit card statement:\n"
        "- Cardholder Name\n"
        "- Date range for transactions\n"
        "- Total billing amount\n"
        "- Transaction details (for all transactions in the statement): Transaction Date, Description, Billing Amount\n"
        f"Text: {text}\n"
        "Please return the data in JSON format."
    )
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=1000
    )
    return response.choices[0].text.strip()

@app.post("/hello")
async def hello_user(name):
    return f"Hello {name}"

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    text = extract_text_from_pdf(file)
    extracted_data = extract_fields_from_text(text)
    return {"extracted_data": extracted_data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
