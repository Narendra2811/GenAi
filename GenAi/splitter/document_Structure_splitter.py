from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=300, chunk_overlap=20, language=Language.PYTHON
)
text = """class Vehicle:
    A general class to represent any vehicle.

    def __init__(self, manufacturer, model, num_wheels):
        self.manufacturer = manufacturer
        self.model = model
        self.num_wheels = num_wheels

    def display_info(self):
        ""Prints the basic information of the vehicle.""
        print(f"Vehicle: {self.manufacturer} {self.model}")
        print(f"Wheels: {self.num_wheels}")

    def drive(self):
        ""A generic drive method.""
        print(f"The {self.model} is driving.")


class Car(Vehicle):
    ""A specific class for cars, inheriting from Vehicle.""

    def __init__(self, manufacturer, model, num_doors):
        # Call the parent class constructor using super()
        super().__init__(manufacturer, model, num_wheels=4)
        self.num_doors = num_doors

    def display_info(self):
        super().display_info()
        print(f"Doors: {self.num_doors}")

    def honk(self):
        print(f"The {self.model} says 'Beep beep'!")


class Motorcycle(Vehicle):

    def __init__(self, manufacturer, model):
        # Motorcycles typically have 2 wheels
        super().__init__(manufacturer, model, num_wheels=2)

    def drive(self):
        print(f"The {self.model} is cruising on two wheels!")


# --- Usage Example ---
if __name__ == "__main__":
    # Create objects (instances) of the classes
    my_car = Car("Honda", "Civic", num_doors=4)
    my_motorcycle = Motorcycle("Harley-Davidson", "Street 750")

    # Use the methods
    print("--- Car Details ---")
    my_car.display_info()
    my_car.drive()
    my_car.honk()
    print("\n--- Motorcycle Details ---")
    my_motorcycle.display_info()
    my_motorcycle.drive()
"""


result = splitter.split_text(text)

for r in result:
    print(r)
    print("-------------------------------------------")
