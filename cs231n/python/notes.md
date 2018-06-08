# Python

Python is a high-level, dynamically typed multiparadigm programming language.
* High-level: lots of abstraction
* Dynamically-typed: type-check at runtime, as opposed to static languages that type check at compile time.
  * "Compiling" is just turning what you write into machine-readable code.
* Multiparadigm: object-oriented, imperative, functional, procedural.

## Basic Data Types
Integers, floats, booleans, strings all act like you would expect from Ruby.

## Containers

### Lists
Basically a resizable array that can contain different data types.

```Python
xs = [3, 1, 2]    # Create a list
print(xs, xs[2])  # Prints "[3, 1, 2] 2"
print(xs[-1])     # Negative indices count from the end of the list; prints "2"
xs[2] = 'foo'     # Lists can contain elements of different types
print(xs)         # Prints "[3, 1, 'foo']"
xs.append('bar')  # Add a new element to the end of the list
print(xs)         # Prints "[3, 1, 'foo', 'bar']"
x = xs.pop()      # Remove and return the last element of the list
print(x, xs)      # Prints "bar [3, 1, 'foo']"
```

**Slicing**
```Python
nums = list(range(5))     # range is a built-in function that creates a list of integers
print(nums)               # Prints "[0, 1, 2, 3, 4]"
print(nums[2:4])          # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(nums[2:])           # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(nums[:2])           # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(nums[:])            # Get a slice of the whole list; prints "[0, 1, 2, 3, 4]"
print(nums[:-1])          # Slice indices can be negative; prints "[0, 1, 2, 3]"
nums[2:4] = [8, 9]        # Assign a new sublist to a slice
print(nums)               # Prints "[0, 1, 8, 9, 4]"
```

**Looping**
```Python
things = ['glass', 'phone', 'almond']
for thing in things:
  print(thing)

for idx, thing in enumerate(things):
  print('#%d: %s' % (idx + 1, thing))

nums = [1, 2, 3]
squared = []
for num in nums:
  squared.append(num ** 2)
  print squared

squared = [num ** 2 for num in nums if num % 2 == 0]
```

### Dictionaries
```python
d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(d['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in d)     # Check if a dictionary has a given key; prints "True"
d['fish'] = 'wet'     # Set an entry in a dictionary
print(d['fish'])      # Prints "wet"
# print(d['monkey'])  # KeyError: 'monkey' not a key of d
print(d.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(d.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
del d['fish']         # Remove an element from a dictionary
print(d.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"
```

**Loops**
```python
d = {'person': 2, 'cat': 4, 'spider': 8}

for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))

for animal, legs in d.items():
    print('A %s has %d legs' % (animal, legs))
```

### Sets
Unordered (don't make assumptions about ordering!) collection of unique elements.

```python
animals = {'cat', 'dog'}
print('cat' in animals)   # Check if an element is in a set; prints "True"
print('fish' in animals)  # prints "False"
animals.add('fish')       # Add an element to a set
print('fish' in animals)  # Prints "True"
print(len(animals))       # Number of elements in a set; prints "3"
animals.add('cat')        # Adding an element that is already in the set does nothing
print(len(animals))       # Prints "3"
animals.remove('cat')     # Remove an element from a set
print(len(animals))       # Prints "2"
```

### Tuples
Immutable, ordered. Can be used as keys in dictionaries, elements of sets (lists can't).

```python
d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)        # Create a tuple
print(type(t))    # Prints "<class 'tuple'>"
print(d[t])       # Prints "5"
print(d[(1, 2)])  # Prints "1"
```

## Functions
