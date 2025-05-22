"""
Counter module for managing answer counts
"""
import os
from pathlib import Path

def read_counter():
    """Read the current counter value from file"""
    counter_file = Path(__file__).parent.parent / "counter.txt"
    if not counter_file.exists():
        return 0
    with open(counter_file, "r") as f:
        return int(f.read())

def increment_counter():
    """Increment the counter and return the new value"""
    count = read_counter() + 1
    write_counter(count)
    return count

def write_counter(count):
    """Write the counter value to file"""
    counter_file = Path(__file__).parent.parent / "counter.txt"
    with open(counter_file, "w") as f:
        f.write(str(count)) 