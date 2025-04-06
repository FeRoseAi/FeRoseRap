from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import uvicorn

app = FastAPI()

# templatesディレクトリの設定
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# CORSの設定(Reactとの通信)
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:3000"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

# モデルの設定
model = AutoModelForCausalLM.from_pretrained("line-corporation/japanese-large-lm-1.7b-instruction-sft")
tokenizer = AutoTokenizer.from_pretrained("line-corporation/japanese-large-lm-1.7b-instruction-sft", use_fast=False, legacy=False)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, torch_dtype=torch.float16)

# チャットのリクエストデータモデル
class ChatRequest(BaseModel):
    text: str
@app.post("/chat")
async def chat(request: ChatRequest):
    # ユーザーの入力によって始まりの文章を可変とする
    request_input = request.text + "\nシステム:"
    text_output = generator(
        request_input,
        max_length=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        repetition_penalty=1.1,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1,
    )

    # 文字列を取り出して整形
    generated_text = text_output[0]["generated_text"]
    
    # 「システム: 」以降を抽出
    if "システム:" in generated_text:
        system_reply = generated_text.split("システム:")[-1].strip()
    else:
        system_reply = generated_text.strip()

    return {"generated_text": system_reply}

# @app.get('/')
# async def read_form(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request,
#                                                      "user_input": None,
#                                                      "output": None})

# @app.post("/")
# async def handle_form(request: Request, user_input: str = Form(...)):
#     # ユーザーの入力によって始まりの文章を可変とする
#     model = AutoModelForCausalLM.from_pretrained("cyberagent/open-calm-7b", device_map="auto", torch_dtype=torch.float16)
#     tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-7b")
#     inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         tokens = model.generate(
#             **inputs,
#             max_new_tokens=64,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9,
#             repetition_penalty=1.05,
#             pad_token_id=tokenizer.pad_token_id,
#         )
#     output = tokenizer.decode(tokens[0], skip_special_tokens=True)
#     return templates.TemplateResponse("index.html", {"request": request,
#                                                      "user_input": user_input,
#                                                      "output": output})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)