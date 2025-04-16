import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from jsonschema import validators
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, field_validator

load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for the UI
app.mount("/static", StaticFiles(directory="static"), name="static")


# Serve the index.html at root
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("static/index.html", "r") as f:
        return f.read()


model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
temperature = float(os.getenv("OPENAI_TEMPERATURE", 0))


llm = ChatOpenAI(model=model, temperature=temperature)

min_length_content = int(os.getenv("MIN_LENGTH_CONTENT", 10))
max_length_content = int(os.getenv("MAX_LENGTH_CONTENT", 10000))

json_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Product",
    "description": "A product in the catalog",
    "type": "object",
    "properties": {
        "id": {"description": "The unique identifier for a product", "type": "integer"},
        "name": {"description": "Name of the product", "type": "string"},
        "price": {
            "description": "The price of the product in USD",
            "type": "number",
            "minimum": 0,
            "exclusiveMinimum": True,
        },
        "tags": {
            "description": "Tags for the product",
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "uniqueItems": True,
        },
        "status": {
            "description": "Product availability status",
            "type": "string",
            "enum": ["available", "discontinued", "out_of_stock"],
        },
    },
    "required": ["id", "name", "price"],
}


class JsonSchemaRequest(BaseModel):
    input: str

    @field_validator("input")
    def validate_input(cls, v):
        if len(v) < min_length_content or len(v) > max_length_content:
            raise ValueError(
                f"Input must be at least {min_length_content} characters long and less than {max_length_content} characters"
            )
        return v


suggest_prompt = ChatPromptTemplate.from_messages(
    messages=[
        (
            "system",
            "You are a JSON Schema Generator expert tasked with creating precise, well-structured JSON schemas based on descriptions provided.\n"
            "Here is example of json schema:\n"
            f"{json_schema}\n"
            "Please generate a json schema based on the description provided by the user.",
        ),
        ("user", "{{input}}"),
    ],
    template_format="jinja2",
)


@app.post("/suggest")
async def suggest(data: JsonSchemaRequest):
    chain = suggest_prompt | llm.with_structured_output(None, method="json_mode")
    response = await chain.ainvoke({"input": data.input})
    return JSONResponse(
        {
            "status": "success",
            "message": "JSON schema generated successfully",
            "json_schema": response,
        },
        status_code=status.HTTP_200_OK,
    )


@app.post("/suggest/stream")
async def suggest_stream(data: JsonSchemaRequest):
    chain = suggest_prompt | llm.with_structured_output(None, method="json_mode")

    async def stream_response():
        async for chunk in chain.astream({"input": data.input}):
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


class ExtractJsonRequest(BaseModel):
    json_schema: dict
    text: str

    @field_validator("text")
    def validate_text(cls, v):
        if len(v) < min_length_content or len(v) > max_length_content:
            raise ValueError(
                f"Text must be at least {min_length_content} characters long and less than {max_length_content} characters"
            )
        return v

    @field_validator("json_schema")
    def validate_json_schema(cls, v):
        try:
            if v:
                validator_class = validators.validator_for(v)
                validator_class.check_schema(v)
            else:
                raise ValueError("JSON schema is required")
        except Exception as e:
            raise ValueError(str(e))
        return v


extract_prompt = ChatPromptTemplate.from_messages(
    messages=[
        ("system", "You are a helpful assistant"),
        ("user", "{input}"),
    ]
)


@app.post("/extract")
async def extract(data: ExtractJsonRequest):
    chain = extract_prompt | llm.with_structured_output(data.json_schema)
    response = await chain.ainvoke({"input": data.text})
    return JSONResponse(
        {"status": "success", "message": "JSON extracted successfully", "data": response},
        status_code=status.HTTP_200_OK,
    )


@app.post("/extract/stream")
async def extract_stream(data: ExtractJsonRequest):
    chain = extract_prompt | llm.with_structured_output(data.json_schema)

    async def stream_response():
        async for chunk in chain.astream({"input": data.text}):
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


if __name__ == "__main__":
    import uvicorn

    environment = os.getenv("ENVIRONMENT", "dev")
    reload = False
    workers = int(os.getenv("WORKERS", 1))
    if environment == "dev":
        reload = True
        workers = 1

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=reload,
        workers=workers,
    )
