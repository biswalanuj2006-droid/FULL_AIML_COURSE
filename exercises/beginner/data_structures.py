# BEGINNER DATA STRUCTURES EXERCISES
# Complete these exercises to practice basic data structures

# ============================================================
# EXERCISE 1: Stack Implementation
# Implement a stack using a Python list
# ============================================================

class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        """Add item to top of stack"""
        # Your code here
        pass
    
    def pop(self):
        """Remove and return top item"""
        # Your code here
        pass
    
    def peek(self):
        """Return top item without removing"""
        # Your code here
        pass
    
    def is_empty(self):
        """Check if stack is empty"""
        # Your code here
        pass
    
    def size(self):
        """Return number of items"""
        # Your code here
        pass


# ============================================================
# EXERCISE 2: Queue Implementation
# Implement a queue using a Python list
# ============================================================

class Queue:
    def __init__(self):
        self.items = []
    
    def enqueue(self, item):
        """Add item to back of queue"""
        # Your code here
        pass
    
    def dequeue(self):
        """Remove and return front item"""
        # Your code here
        pass
    
    def front(self):
        """Return front item without removing"""
        # Your code here
        pass
    
    def is_empty(self):
        """Check if queue is empty"""
        # Your code here
        pass
    
    def size(self):
        """Return number of items"""
        # Your code here
        pass


# ============================================================
# EXERCISE 3: Linked List Implementation
# Implement a singly linked list
# ============================================================

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        """Add node to end"""
        # Your code here
        pass
    
    def prepend(self, data):
        """Add node to beginning"""
        # Your code here
        pass
    
    def delete(self, data):
        """Delete first occurrence of data"""
        # Your code here
        pass
    
    def search(self, data):
        """Return True if data exists"""
        # Your code here
        pass
    
    def display(self):
        """Print all elements"""
        # Your code here
        pass
    
    def length(self):
        """Return length of list"""
        # Your code here
        pass


# ============================================================
# EXERCISE 4: Hash Map Implementation
# Implement a simple hash map
# ============================================================

class HashMap:
    def __init__(self, size=10):
        self.size = size
        self.map = [[] for _ in range(size)]
    
    def _hash(self, key):
        """Hash function"""
        # Your code here
        pass
    
    def put(self, key, value):
        """Insert key-value pair"""
        # Your code here
        pass
    
    def get(self, key):
        """Get value by key"""
        # Your code here
        pass
    
    def remove(self, key):
        """Remove key-value pair"""
        # Your code here
        pass
    
    def contains(self, key):
        """Check if key exists"""
        # Your code here
        pass


# ============================================================
# EXERCISE 5: Practice Problems
# ============================================================

# 1. Check if a string has balanced parentheses
def is_balanced(s):
    # Your code here
    pass

# 2. Reverse a string using a stack
def reverse_string(s):
    # Your code here
    pass

# 3. Implement a queue using two stacks
class QueueWithStacks:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
    
    def enqueue(self, item):
        # Your code here
        pass
    
    def dequeue(self):
        # Your code here
        pass

# 4. Find the middle element of a linked list
def find_middle(linked_list):
    # Your code here
    pass

# 5. Check if a linked list has a cycle
def has_cycle(linked_list):
    # Your code here
    pass


# ============================================================
# SOLUTIONS (Unfold to check your answers)
# ============================================================

# EXERCISE 1 SOLUTION:
# class Stack:
#     def __init__(self):
#         self.items = []
#     def push(self, item):
#         self.items.append(item)
#     def pop(self):
#         if not self.is_empty():
#             return self.items.pop()
#         return None
#     def peek(self):
#         if not self.is_empty():
#             return self.items[-1]
#         return None
#     def is_empty(self):
#         return len(self.items) == 0
#     def size(self):
#         return len(self.items)

# EXERCISE 5 SOLUTION (is_balanced):
# def is_balanced(s):
#     stack = []
#     mapping = {')': '(', ']': '[', '}': '{'}
#     for char in s:
#         if char in mapping:
#             if not stack or stack[-1] != mapping[char]:
#                 return False
#             stack.pop()
#         else:
#             stack.append(char)
#     return len(stack) == 0
