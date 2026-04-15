from pydantic import BaseModel
from typing import Optional


class Address(BaseModel):
    city: str
    state: Optional[str] = None
    zip_code: int


class User(BaseModel):
    name: str
    address: Address  # pass  Address as a field in User


data = {"name": "Bob", "address": {"city": "Delhi", "zip_code": 110001}}

user = User(**data)

print(user.address.city)
