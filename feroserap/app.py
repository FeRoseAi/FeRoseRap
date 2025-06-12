from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
import uvicorn
import json
import os

app = FastAPI()

# templatesディレクトリの設定
BASE_DIR = Path(__file__).resolve().parent
chat_data_path = BASE_DIR / "chat_data"
chattext_path = chat_data_path / "chattext.json"
# templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# CORSの設定(Reactとの通信)
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:3000"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

# データセット(JGLUEを利用)
# train_dataset = load_dataset("llm-book/JGLUE", name="JCommonsenseQA", split="train[:20%]")
# valid_dataset = load_dataset("llm-book/JGLUE", name="JCommonsenseQA", split="validation[:20%]")

# モデルの設定
model = AutoModelForCausalLM.from_pretrained("line-corporation/japanese-large-lm-1.7b-instruction-sft")
tokenizer = AutoTokenizer.from_pretrained("line-corporation/japanese-large-lm-1.7b-instruction-sft", use_fast=False, legacy=False)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, torch_dtype=torch.float16)

# チャットのリクエストデータモデル
class ChatRequest(BaseModel):
    text: str

@app.post("/chat")
async def chat(request: ChatRequest):
    # ユーザーの会話をjson形式でデータに保存する
    user_input = request.text
    
    # ユーザーの入力によって始まりの文章を可変とする
    request_input = user_input + "\nシステム:"
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

    # 質問と回答をjson形式で保存
    input_text = user_input
    output_text = system_reply

    if "\n" in input_text:
        input_text = input_text.replace("\n", "")

    if "\n" in output_text:
        output_text = output_text.replace("\n", "")

    json_text = {"input_text": input_text, "output_text": output_text}

    if os.path.exists(chattext_path):
        try:
            with open(chattext_path, "r", encoding="utf-8") as f:
                chat_history = json.load(f)
        except json.JSONDecodeError:
            chat_history = []
    else:
        chat_history = []
    chat_history.append(json_text)
    with open(chattext_path, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)

    return {"generated_text": system_reply}

# 学習関数


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)