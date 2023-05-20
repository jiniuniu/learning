from dotenv import load_dotenv
from pydantic import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    api_key: str
    login: str
    seed: int

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
print(settings)
# api_key='mysecretapi_key' login='jiniuniu' seed=2023
