# server.py

# 필요한 라이브러리와 모듈을 불러옵니다.
import requests  # 다른 서비스에 HTTP 요청을 보내기 위해 사용
from fastapi import FastAPI, HTTPException  # FastAPI는 API를 만들기 위한 웹 프레임워크
from pydantic import BaseModel, Field  # Pydantic은 데이터 검증을 위해 사용
from langchain.llms.base import LLM  # LangChain에서 언어 모델의 기본 클래스
from langchain.prompts import PromptTemplate  # 프롬프트 템플릿을 만들기 위해 사용
from langchain.schema import BaseOutputParser  # 언어 모델의 출력을 파싱하기 위해 사용
from langchain.chains import LLMChain  # LLM과 함께 사용할 작업 체인
import uvicorn  # FastAPI 서버를 실행하기 위해 사용
import logging  # 디버깅과 모니터링을 위한 로그 기록

# 로깅 설정: 콘솔에 로그 메시지를 출력하도록 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Pydantic 모델 정의
# API가 받을 데이터와 보낼 데이터의 구조를 정의합니다.

class TranslateRequest(BaseModel):
    """
    번역 요청의 구조를 정의하는 클래스입니다.
    두 가지 정보가 필요합니다:
    - language: 번역할 언어
    - text: 번역할 텍스트
    """
    language: str
    text: str

class TranslateResponse(BaseModel):
    """
    번역 응답의 구조를 정의하는 클래스입니다.
    하나의 필드를 포함합니다:
    - result: 번역된 텍스트
    """
    result: str

# 2. 커스텀 출력 파서 정의
# 언어 모델에서 받은 텍스트를 처리하는 파서입니다.

class StrOutputParser(BaseOutputParser):
    """
    언어 모델의 출력 텍스트를 받아서 불필요한 공백을 제거합니다.
    """
    def parse(self, text: str) -> str:
        return text.strip()

# 3. Ollama LLM 클래스 정의
# Ollama API와 통신하여 번역을 수행하는 언어 모델 클래스입니다.

class OllamaLLM(LLM):
    """
    Ollama API와 통신하는 커스텀 언어 모델 클래스입니다.
    """
    # 기본 모델 이름과 API URL을 설정합니다.
    model_name: str = Field(default="llama3.2:3b")  # 사용하려는 언어 모델의 이름
    api_url: str = Field(default="http://localhost:11435")  # Ollama API가 실행 중인 URL

    class Config:
        extra = "allow"  # 설정에 추가 필드를 허용

    def _call(self, prompt: str, stop: list = None) -> str:
        """
        프롬프트를 Ollama API에 보내고 응답을 받아오는 메서드입니다.
        """
        # API 엔드포인트 URL을 구성합니다.
        endpoint = f"{self.api_url}/v1/completions"
        # 요청에 보낼 데이터를 준비합니다.
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 100,  # 생성할 최대 토큰 수
            "temperature": 0.7,  # 출력의 다양성을 조절하는 파라미터
            "top_p": 0.9,  # 또 다른 다양성 조절 파라미터
            "stop": stop  # 응답 생성을 중단할 토큰 리스트
        }
        logger.info(f"Ollama API에 요청을 보냅니다: {endpoint} with payload: {payload}")
        try:
            # Ollama API에 POST 요청을 보냅니다.
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()  # 요청이 실패하면 예외를 발생시킵니다.
            data = response.json()  # JSON 응답을 파싱합니다.
            logger.debug(f"Ollama API 응답 데이터: {data}")
            # 응답에 'choices'가 포함되어 있는지 확인합니다.
            if "choices" in data and len(data["choices"]) > 0:
                result = data["choices"][0]["text"].strip()  # 번역된 텍스트를 추출합니다.
                logger.info(f"Ollama API에서 받은 번역 결과: {result}")
                return result
            else:
                logger.error("Ollama API 응답에 'choices'가 없습니다.")
                raise HTTPException(status_code=500, detail="Ollama API로부터 유효하지 않은 응답을 받았습니다.")
        except requests.exceptions.RequestException as e:
            # 요청 중 오류가 발생하면 로그에 기록하고 예외를 발생시킵니다.
            logger.error(f"Ollama API와 통신 중 오류 발생: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @property
    def _identifying_params(self):
        """
        LLM의 식별 매개변수를 반환합니다.
        """
        return {"model": self.model_name}

    @property
    def _llm_type(self):
        """
        LLM의 타입을 반환합니다.
        """
        return "OllamaLLM"

# 4. Ollama LLM 모델 초기화
llama_llm = OllamaLLM()

# 5. 번역을 위한 프롬프트 템플릿 생성
# 이 템플릿은 모델에게 어떤 작업을 수행할지 지시합니다.

prompt_template = PromptTemplate(
    input_variables=["language", "text"],
    template=(
        "Translate the following text into {language}.\n"
        "Provide only the translated text without any additional explanations:\n\n{text}"
    )
)

# 6. 출력 파서 생성
# 모델의 응답을 정리하기 위한 파서입니다.

parser = StrOutputParser()

# 7. 체인 생성 - LLMChain 사용
# LLM, 프롬프트, 파서를 연결하여 번역 작업을 수행합니다.

chain = LLMChain(llm=llama_llm, prompt=prompt_template, output_parser=parser)

# 8. FastAPI 앱 정의
# FastAPI를 사용하여 API 애플리케이션을 설정합니다.

app = FastAPI(
    title="LangChain Ollama Translation API",
    version="1.0",
    description="FastAPI, LangChain, 그리고 Ollama의 LLaMA 모델을 사용한 번역 API입니다."
)

# 9. 엔드포인트 정의
# 사용자가 번역 요청을 보낼 수 있는 /translate 엔드포인트를 만듭니다.

@app.post("/translate", response_model=TranslateResponse)
def translate(request: TranslateRequest):
    """
    /translate 엔드포인트에 대한 POST 요청을 처리하는 함수입니다.
    번역 요청을 받아 번역된 결과를 반환합니다.
    """
    logger.info(f"번역 요청 수신: 언어={request.language}, 텍스트='{request.text}'")
    try:
        # 번역 체인을 실행하여 결과를 얻습니다.
        result = chain.run(language=request.language, text=request.text)
        logger.info(f"번역 결과: {result}")
        return TranslateResponse(result=result)
    except Exception as e:
        # 번역 중 오류가 발생하면 로그에 기록하고 예외를 발생시킵니다.
        logger.error(f"번역 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 10. 임베디드 테스트 라우트 (선택 사항)
# 서버가 제대로 작동하는지 빠르게 테스트할 수 있는 /test 엔드포인트를 만듭니다.

@app.get("/test")
def test_server():
    """
    /test 엔드포인트에 대한 GET 요청을 처리하는 함수입니다.
    간단한 번역 테스트를 수행하고 결과를 반환합니다.
    """
    logger.info("임베디드 테스트 실행: 'Hello, world!'를 스페인어로 번역 중입니다.")
    test_request = TranslateRequest(language="spanish", text="Hello, world!")
    try:
        # 테스트 번역 체인을 실행하여 결과를 얻습니다.
        result = chain.run(language=test_request.language, text=test_request.text)
        logger.info(f"테스트 번역 결과: {result}")
        return TranslateResponse(result=result)
    except Exception as e:
        # 테스트 번역 중 오류가 발생하면 로그에 기록하고 예외를 발생시킵니다.
        logger.error(f"테스트 번역 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 11. 서버 시작
# FastAPI 서버를 실행하여 외부에서 요청을 받을 수 있게 합니다.

if __name__ == "__main__":
    logger.info("FastAPI 서버를 시작합니다...")
    # uvicorn.run은 서버를 호스트의 모든 IP(0.0.0.0)와 포트 8000에서 실행시킵니다.
    uvicorn.run(app, host="0.0.0.0", port=8000)
