from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel
from typing import Literal
from langchain_openrouter import ChatOpenRouter


class TestSchema(BaseModel):
    answer: str
    risk: Literal["low", "medium", "high"]


llm = ChatOpenRouter(
    model="liquid/lfm-2.5-2.6b:free",
    temperature=0,
)

structured_llm = llm.with_structured_output(
    TestSchema,
    method="json_schema",
    strict=True,
)

try:
    result = structured_llm.invoke(
        "Return a simple test result. The answer should say 'OpenRouter works'."
    )

    print("SUCCESS")
    print(result)
    print(result.model_dump())

except Exception as e:
    print("\n========== ACTUAL ERROR ==========")
    print(type(e).__name__)
    print(str(e))
    print("==================================")





