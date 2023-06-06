from pydantic import BaseModel, ValidationError, Field
from typing import Optional


class Address(BaseModel):
    street: str
    number: int
    zipcode: str


class Person(BaseModel):
    first_name: str = Field(min_length=2, max_length=20)
    last_name: str
    age: int = Field(le=150)
    address: Optional[Address]


data = {"first_name": "牛牛", "last_name": "纪", "age": 200}
try:
    person = Person(**data)
except ValidationError as e:
    print(e.json())

# [
#   {
#     "loc": [
#       "age"
#     ],
#     "msg": "ensure this value is less than or equal to 150",
#     "type": "value_error.number.not_le",
#     "ctx": {
#       "limit_value": 150
#     }
#   }
# ]
