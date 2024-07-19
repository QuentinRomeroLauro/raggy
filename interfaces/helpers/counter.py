import os

COUNTER = 'counter.txt' 

# Function to read the counter from the file
def read_counter():
    if os.path.exists(COUNTER):
        with open(COUNTER, 'r') as f:
            return int(f.read())
    else:
        return 0

# Function to write the counter to the file
def write_counter(counter):
    with open(COUNTER, 'w') as f:
        f.write(str(counter))


def increment_counter():
    counter = read_counter()
    counter += 1
    write_counter(counter)
    return counter