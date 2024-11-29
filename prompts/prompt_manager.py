from pydantic import (BaseModel, Field, constr)
from typing import Any

class Prompt(BaseModel):
    def __init__(self, template, **data):
        super().__init__(**data)
        self.prompt_template = template


    def add_info(self, **kwargs):
        if kwargs:
            for key, value in kwargs.items():
                self.prompt_template = self.prompt_template.replace("{" + key.upper() + "}", value)
        return self.prompt_template