import os
import logging
import asyncio

from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# from vector_store_db import check_vector_store, save_to_vector_store
# from vector_search import search_similar_code

# 환경 변수 로드
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 로그 설정
logging.basicConfig(level=logging.DEBUG)

# LangChain LLM 초기화
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)

class CodeStyle(str, Enum):
    PEP8 = "PEP8"
    Google = "Google"
    NoneStyle = "None"

class CodeStructure(str, Enum):
    Functional = "functional"
    ClassBased = "class-based"

class CodeRequest(BaseModel):
    description: str
    style: CodeStyle = CodeStyle.PEP8
    include_comments: bool = True
    structure: CodeStructure = CodeStructure.Functional

class CodeGenerator:
    """Python 코드 생성기 (RAG 미적용)"""

    @classmethod
    async def generate_code(cls, request: CodeRequest, model: str = "gemini-pro") -> str:
        """비동기 Gemini API 호출 (RAG 미적용)"""
        
        # # 기존 코드 확인 (Vector Store)
        # cached_code = check_vector_store(request.description)
        # if cached_code:
        #     return cached_code

        # # 유사한 코드 검색 (RAG)
        # similar_codes = search_similar_code(request.description, top_k=1)
        # similar_code_text = similar_codes[0][1] if similar_codes else "참고 코드 없음"
        
        similar_code_text = "참고 코드 없음"

        prompt = cls._generate_prompt(request, similar_code_text)

        # LangChain LLM 호출
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: llm.invoke(prompt)
        )

        generated_code = response.content if isinstance(response, AIMessage) else "코드 생성 실패"

        # # Vector Store에 저장 (DB 활용)
        # save_to_vector_store(request.description, generated_code)

        return generated_code

    @classmethod
    def _generate_prompt(cls, request: CodeRequest, similar_code: str) -> str:
        """LangChain PromptTemplate을 사용한 최적화된 프롬프트 생성"""

        # 변환된 문자열 변수 정의 (일관된 방식 적용)
        include_comments_text = "포함" if request.include_comments else "제외"
        structure_text = "함수형" if request.structure == "functional" else "클래스형"

        template = PromptTemplate(
            input_variables=["description", "style", "include_comments", "structure", "similar_code"],
            template="""
            너는 Python 코드 생성을 전문으로 하는 AI야.
            사용자가 요청한 코드가 **올바르게 실행될 수 있도록** 작성해야 해.

            ### 🛠️ 필수 조건
            - Python 문법 오류(SyntaxError)가 없어야 함.
            - 실행 시 오류(RuntimeError)가 발생하지 않아야 함.
            - 코드의 논리가 정확해야 하며, 예상된 출력이 나와야 함.

            ### 🎨 코드 스타일 & 구조
            - 코드 스타일: {style}
            - 주석 포함 여부: {include_comments}
            - 코드 구조: {structure}

            ### 🎯 코드 생성 요청
            "{description}"

            ### 📝 참고 코드
            ```python
            {similar_code}
            ```
            """
        )

        return template.format(
            description=request.description,
            style=request.style.value,
            include_comments=include_comments_text,  
            structure=structure_text,
            similar_code=similar_code
        )