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

#function 0
def main (): # No typehints since it doesn't take any input or give any output beside running the function
    """Run this python script which looks at records and gets the average grade"""
    records = add_records(file_path)
    average = get_average(records)
    return

#function 1
def add_records (pathtocsv) -> list:
    """Reads records from CSV and returns them as a list"""
    records = []
    with open(pathtocsv, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            records.append(row)
    return records

#function 2
def get_average(records: list) -> float:
    """Calculates the total of the grades and from there returns the average"""
    total = sum(float(record['Grade']) for record in records)
    average = total / len(records)
    return average

#someone turn this into a function 3
print(f"Average Grade: {average}")
print("--------------------")

filtered_records = [record for record in records if float(record['Grade']) >= 80.0]

print("Student Report")
print("--------------")

#someone turn this into a function 4
for record in filtered_records:
    print(f"Name: {record['Name']}")
    print(f"Grade: {record['Grade']}")
    print("--------------------")

# needed for main
if __name__ == "__main__":
    main()