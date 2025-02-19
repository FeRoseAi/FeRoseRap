from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

app = FastAPI()

# templatesディレクトリの設定
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.get('/')
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request,
                                                     "user_input": None,
                                                     "output": None})

@app.post("/")
async def handle_form(request: Request, user_input: str = Form(...)):
    # ユーザーの入力によって始まりの文章を可変とする
    model = AutoModelForCausalLM.from_pretrained("cyberagent/open-calm-7b", device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-7b")
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
        )
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return templates.TemplateResponse("index.html", {"request": request,
                                                     "user_input": user_input,
                                                     "output": output})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)