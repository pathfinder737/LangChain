# web_server.py

# 필요한 라이브러리와 모듈을 불러옵니다.
import requests
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.chains import LLMChain
import uvicorn
import logging
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Pydantic 모델 정의 (기존과 동일)
class TranslateRequest(BaseModel):
    language: str
    text: str

class TranslateResponse(BaseModel):
    result: str

# 2. 커스텀 출력 파서 정의 (기존과 동일)
class StrOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        return text.strip()

# 3. Ollama LLM 클래스 정의 (기존과 동일)
class OllamaLLM(LLM):
    model_name: str = Field(default="llama3.2:3b")
    api_url: str = Field(default="http://localhost:11435")

    class Config:
        extra = "allow"

    def _call(self, prompt: str, stop: list = None) -> str:
        endpoint = f"{self.api_url}/v1/completions"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": stop
        }
        logger.info(f"Ollama API에 요청을 보냅니다: {endpoint} with payload: {payload}")
        try:
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Ollama API 응답 데이터: {data}")
            if "choices" in data and len(data["choices"]) > 0:
                result = data["choices"][0]["text"].strip()
                logger.info(f"Ollama API에서 받은 번역 결과: {result}")
                return result
            else:
                logger.error("Ollama API 응답에 'choices'가 없습니다.")
                raise HTTPException(status_code=500, detail="Ollama API로부터 유효하지 않은 응답을 받았습니다.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API와 통신 중 오류 발생: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @property
    def _identifying_params(self):
        return {"model": self.model_name}

    @property
    def _llm_type(self):
        return "OllamaLLM"

# 4. Ollama LLM 모델 초기화 (기존과 동일)
llama_llm = OllamaLLM()

# 5. 번역을 위한 프롬프트 템플릿 생성 (기존과 동일)
prompt_template = PromptTemplate(
    input_variables=["language", "text"],
    template=(
        "Translate the following text into {language}.\n"
        "Provide only the translated text without any additional explanations:\n\n{text}"
    )
)

# 6. 출력 파서 생성 (기존과 동일)
parser = StrOutputParser()

# 7. 체인 생성 - LLMChain 사용 (기존과 동일)
chain = LLMChain(llm=llama_llm, prompt=prompt_template, output_parser=parser)

# 8. FastAPI 앱 정의
app = FastAPI(
    title="LangChain Ollama Translation API",
    version="1.0",
    description="FastAPI, LangChain, 그리고 Ollama의 LLaMA 모델을 사용한 번역 API입니다."
)

# 9. 템플릿 설정
templates = Jinja2Templates(directory="templates")

# 10. 웹 페이지 라우트 추가
@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    """
    사용자 입력을 받을 수 있는 웹 페이지를 렌더링합니다.
    """
    return templates.TemplateResponse("form.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
def submit_form(request: Request, language: str = Form(...), text: str = Form(...)):
    """
    사용자가 제출한 데이터를 받아 번역 결과를 웹 페이지에 표시합니다.
    """
    logger.info(f"번역 요청 수신: 언어={language}, 텍스트='{text}'")
    try:
        result = chain.run(language=language, text=text)
        logger.info(f"번역 결과: {result}")
        return templates.TemplateResponse("form.html", {"request": request, "result": result, "language": language, "text": text})
    except Exception as e:
        logger.error(f"번역 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 11. 기존의 /translate 엔드포인트 유지 (필요에 따라 제거 가능)
@app.post("/translate", response_model=TranslateResponse)
def translate(request: TranslateRequest):
    """
    기존의 /translate 엔드포인트입니다.
    """
    logger.info(f"번역 요청 수신: 언어={request.language}, 텍스트='{request.text}'")
    try:
        result = chain.run(language=request.language, text=request.text)
        logger.info(f"번역 결과: {result}")
        return TranslateResponse(result=result)
    except Exception as e:
        logger.error(f"번역 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 12. 서버 시작
if __name__ == "__main__":
    logger.info("FastAPI 서버를 시작합니다...")
    uvicorn.run(app, host="0.0.0.0", port=7777)
