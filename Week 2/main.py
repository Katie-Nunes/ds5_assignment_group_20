# TASKS:
# Pick whichever function you want (add your name), then add all three changes, then commit and push
# - Add functions, type hints and doctstring
    # - 0 (Katie)
    # - 1 (Katie)
    # - 2 (Katie)
    # - 3
    # - 4

import csv

file_path = input("Enter the path to the CSV file: ")
records = []

#main function 0

#turn to function 1
with open(file_path, 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        records.append(row)

#turn to function 2
total = sum(float(record['Grade']) for record in records)
average = total / len(records)

#turn to function 3
print(f"Average Grade: {average}")
print("--------------------")

filtered_records = [record for record in records if float(record['Grade']) >= 80.0]

print("Student Report")
print("--------------")

#turn to function 4
for record in filtered_records:
    print(f"Name: {record['Name']}")
    print(f"Grade: {record['Grade']}")
    print("--------------------")