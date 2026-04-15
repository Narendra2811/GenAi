from typing import TypedDict


class Person(TypedDict):
    name: str
    age: int
    email: str


person1 = Person(name="Alice", age=30, email="alice@example.com")

person2: Person = {"name": "bob", "age": 25, "email": "abc@gmail.com"}
print(person1)
print(type(person1))
print(person2)
print(type(person2))
