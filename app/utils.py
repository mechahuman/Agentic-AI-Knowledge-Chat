import os
from dotenv import load_dotenv

load_dotenv()

def get_env(var: str) -> str:
    value = os.getenv(var)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {var}")
    return value
