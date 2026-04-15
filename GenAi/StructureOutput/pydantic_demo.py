from pydantic import BaseModel, EmailStr
from typing import Optional


class person(BaseModel):
    name: str = "Narendra"
    age: Optional[int] = None
    email: EmailStr


info = {"name": "Narendra", "age": "30", "email": "abc@gmail.com"}

person_info = person(**info)

print(person_info)

print(person_info.name)

person_info_dict = person_info.model_dump()  # Convert to dictionary
print(person_info_dict["name"])

person_info_json = person_info.model_dump_json()
print(person_info_json)
print(type(person_info_json))
print(type(person_info))
