import os
from pydantic import BaseModel
from prompts import browser_prompt, base_prompt, google_prompt

SYSTEM_PROMPT = base_prompt + browser_prompt + google_prompt

class Config(BaseModel):
    api_key: str | None = os.environ.get("ANTHROPIC_API_KEY")
    system_prompt: str | None = SYSTEM_PROMPT

    def update_api_key(self, new_key: str):
        self.api_key = new_key
        os.environ["ANTHROPIC_API_KEY"] = new_key

    def update_system_prompt(self, new_prompt: str):
        self.system_prompt = new_prompt

    def reset_system_prompt(self):
        self.system_prompt = SYSTEM_PROMPT