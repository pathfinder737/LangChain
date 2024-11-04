# client.py

# 필요한 라이브러리와 모듈을 불러옵니다.
import requests  # 서버에 HTTP 요청을 보내기 위해 사용
import json  # JSON 데이터를 다루기 위해 사용
import logging  # 디버깅과 모니터링을 위한 로그 기록

# 로깅 설정: 콘솔에 로그 메시지를 출력하도록 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def translate_text(language, text, server_url="http://localhost:8000/translate"):
    """
    이 함수는 서버에 번역 요청을 보내고 번역된 텍스트를 반환합니다.
    
    매개변수:
    - language: 번역할 언어
    - text: 번역할 텍스트
    - server_url: 번역 서버의 URL (기본값은 로컬호스트의 /translate 엔드포인트)
    
    반환값:
    - 성공 시 번역된 텍스트
    - 실패 시 오류 메시지
    """
    # 번역 요청에 필요한 데이터를 생성합니다.
    payload = {
        "language": language,
        "text": text
    }
    # 요청 헤더를 설정하여 JSON 데이터를 보낸다는 것을 알립니다.
    headers = {
        "Content-Type": "application/json"
    }
    logger.info(f"{server_url}에 POST 요청을 보냅니다: {payload}")
    try:
        # /translate 엔드포인트에 POST 요청을 보냅니다.
        response = requests.post(server_url, data=json.dumps(payload), headers=headers)
        logger.info(f"응답 상태 코드: {response.status_code}")
        if response.status_code == 200:
            # 요청이 성공하면 응답에서 번역된 텍스트를 가져옵니다.
            result = response.json().get("result")
            logger.info(f"번역 결과: {result}")
            return result
        else:
            # 요청이 실패하면 오류 메시지를 가져옵니다.
            error_detail = response.json().get("detail", response.text)
            logger.error(f"요청 실패: {error_detail}")
            return f"Error: {error_detail}"
    except requests.exceptions.RequestException as e:
        # 요청 중 예외가 발생하면 로그에 기록하고 예외 메시지를 반환합니다.
        logger.error(f"요청 중 예외 발생: {e}")
        return f"Exception: {e}"

def run_tests():
    """
    이 함수는 여러 번역 테스트 케이스를 실행하여 서버가 제대로 작동하는지 확인합니다.
    """
    # 다양한 언어와 텍스트로 구성된 테스트 케이스 목록을 정의합니다.
    test_cases = [
        {"language": "italian", "text": "Hello, how are you?"},
        {"language": "spanish", "text": "Good morning!"},
        {"language": "french", "text": "What is your name?"},
        {"language": "german", "text": "I love programming."},
    ]

    # 각 테스트 케이스를 순차적으로 실행합니다.
    for i, test in enumerate(test_cases, 1):
        logger.info(f"테스트 케이스 {i}: '{test['text']}'를 {test['language']}로 번역 중입니다.")
        translated = translate_text(test['language'], test['text'])
        print(f"테스트 케이스 {i} - {test['language'].capitalize()}: {translated}")

if __name__ == "__main__":
    logger.info("클라이언트 테스트를 시작합니다...")
    run_tests()
