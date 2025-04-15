from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    CONFIDENCE: float

    model_config = SettingsConfigDict(env_file='.env')



def get_settings():
    return Settings()
