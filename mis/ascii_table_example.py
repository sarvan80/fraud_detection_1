from tabulate import tabulate

def demonstrate_ascii_table():
    # Sample data: a list of lists
    data = [
        ["Alice", 24, "Engineer"],
        ["Bob", 19, "Designer"],
        ["Charlie", 30, "Manager"],
        ["Diana", 22, "Analyst"]
    ]

    # Define headers
    headers = ["Name", "Age", "Occupation"]

    print("--- Basic ASCII Table ---")
    print(tabulate(data, headers=headers, tablefmt="grid"))

    print("\n--- Fancy ASCII Table (orgtbl) ---")
    print(tabulate(data, headers=headers, tablefmt="orgtbl"))

    print("\n--- Simple ASCII Table (plain) ---")
    print(tabulate(data, headers=headers, tablefmt="plain"))

    # Sample data: a dictionary
    data_dict = {
        "Product": ["Laptop", "Mouse", "Keyboard", "Monitor"],
        "Price": [1200, 25, 75, 300],
        "In Stock": [True, True, False, True]
    }

    print("\n--- Table from Dictionary ---")
    print(tabulate(data_dict, headers="keys", tablefmt="grid"))

if __name__ == "__main__":
    demonstrate_ascii_table()

