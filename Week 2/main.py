# TASKS:
# Pick whichever task you want, then add the changes, then commit and push, and clarify which one you did
# - Add functions
# - Add Type hints
# - Add docstrings

import csv

file_path = input("Enter the path to the CSV file: ")
records = []

with open(file_path, 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        records.append(row)

total = sum(float(record['Grade']) for record in records)
average = total / len(records)

print(f"Average Grade: {average}")
print("--------------------")

filtered_records = [record for record in records if float(record['Grade']) >= 80.0]

print("Student Report")
print("--------------")

for record in filtered_records:
    print(f"Name: {record['Name']}")
    print(f"Grade: {record['Grade']}")
    print("--------------------")