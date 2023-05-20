import re
from pydantic import BaseModel, validator, ValidationError
from typing import Optional


class Address(BaseModel):
    street: str
    number: int
    zipcode: str


class Person(BaseModel):
    first_name: str
    last_name: str
    cell_phone_number: str
    address: Optional[Address]

    @validator("cell_phone_number")
    def validate_cell_phone_number(cls, v):
        match = re.match(r"^135\d{8}$", v)
        if len(v) != 11:
            raise ValueError("cell phone number must be 11 digits")
        elif match is None:
            raise ValueError("cell phone number must start with 135")
        return v


data = {"first_name": "牛牛", "last_name": "纪", "cell_phone_number": "13588886666444"}
try:
    person = Person(**data)
except ValidationError as e:
    print(e.json())
