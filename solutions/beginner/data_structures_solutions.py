# ============================================================
# SOLUTIONS — exercises/beginner/data_structures.py
# Run: python data_structures_solutions.py
# ============================================================

# ------------------------------------------------------------
# EXERCISE 1: Stack Implementation
# ------------------------------------------------------------
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        """Add item to top of stack (O(1) amortized)."""
        self.items.append(item)

    def pop(self):
        """Remove and return top item."""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self.items.pop()

    def peek(self):
        """Return top item without removing."""
        if self.is_empty():
            return None
        return self.items[-1]

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)


# ------------------------------------------------------------
# EXERCISE 2: Queue Implementation
# NOTE: a list-based queue with pop(0) is O(n). For a
# production queue use collections.deque (O(1) both ends).
# ------------------------------------------------------------
from collections import deque


class Queue:
    def __init__(self):
        self.items = deque()

    def enqueue(self, item):
        """Add item to back of queue."""
        self.items.append(item)

    def dequeue(self):
        """Remove and return front item."""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self.items.popleft()

    def front(self):
        """Return front item without removing."""
        if self.is_empty():
            return None
        return self.items[0]

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)


# ------------------------------------------------------------
# EXERCISE 3: Singly Linked List
# ------------------------------------------------------------
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        """Add node to end (O(n) without a tail pointer)."""
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        curr = self.head
        while curr.next is not None:
            curr = curr.next
        curr.next = new_node

    def prepend(self, data):
        """Add node to beginning (O(1))."""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def delete(self, data):
        """Delete first occurrence of data."""
        if self.head is None:
            return
        if self.head.data == data:
            self.head = self.head.next
            return
        curr = self.head
        while curr.next is not None and curr.next.data != data:
            curr = curr.next
        if curr.next is not None:
            curr.next = curr.next.next

    def search(self, data):
        """Return True if data exists."""
        curr = self.head
        while curr is not None:
            if curr.data == data:
                return True
            curr = curr.next
        return False

    def display(self):
        """Print all elements."""
        curr = self.head
        while curr is not None:
            print(curr.data, end=" -> ")
            curr = curr.next
        print("None")

    def length(self):
        """Return length of list."""
        count = 0
        curr = self.head
        while curr is not None:
            count += 1
            curr = curr.next
        return count

    def to_list(self):
        """Helper for tests: convert to a Python list."""
        result = []
        curr = self.head
        while curr is not None:
            result.append(curr.data)
            curr = curr.next
        return result


# ------------------------------------------------------------
# EXERCISE 4: Hash Map (chaining for collisions)
# ------------------------------------------------------------
class HashMap:
    def __init__(self, size=10):
        self.size = size
        self.map = [[] for _ in range(size)]

    def _hash(self, key):
        """Simple hash function (built-in hash is salted per process,
        so use a deterministic one for reproducible behavior)."""
        total = 0
        for char in str(key):
            total += ord(char)
        return total % self.size

    def put(self, key, value):
        """Insert key-value pair (overwrite existing key)."""
        bucket = self.map[self._hash(key)]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)   # overwrite
                return
        bucket.append((key, value))

    def get(self, key):
        """Get value by key; raise KeyError if absent."""
        bucket = self.map[self._hash(key)]
        for k, v in bucket:
            if k == key:
                return v
        raise KeyError(key)

    def remove(self, key):
        """Remove key-value pair."""
        bucket = self.map[self._hash(key)]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                return

    def contains(self, key):
        """Check if key exists."""
        bucket = self.map[self._hash(key)]
        return any(k == key for k, v in bucket)


# ------------------------------------------------------------
# EXERCISE 5: Practice Problems
# ------------------------------------------------------------
# 1. Balanced parentheses
def is_balanced(s):
    stack = []
    mapping = {")": "(", "]": "[", "}": "{"}
    for char in s:
        if char in mapping:  # closing bracket
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:                # opening bracket (ignore other chars)
            if char in mapping.values():
                stack.append(char)
    return len(stack) == 0


# 2. Reverse a string using a stack
def reverse_string(s):
    stack = list(s)
    return "".join(stack.pop() for _ in range(len(stack)))


# 3. Queue implemented with two stacks (amortized O(1))
class QueueWithStacks:
    def __init__(self):
        self.stack_in = []   # for enqueue
        self.stack_out = []  # for dequeue

    def enqueue(self, item):
        self.stack_in.append(item)

    def dequeue(self):
        if not self.stack_out:
            while self.stack_in:
                self.stack_out.append(self.stack_in.pop())
        if not self.stack_out:
            raise IndexError("dequeue from empty queue")
        return self.stack_out.pop()


# 4. Middle element of a linked list (tortoise & hare)
def find_middle(linked_list):
    slow = fast = linked_list.head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
    return slow.data if slow else None


# 5. Detect a cycle in a linked list (Floyd's algorithm)
def has_cycle(linked_list):
    slow = fast = linked_list.head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False


# ============================================================
# TESTS — verify everything works
# ============================================================
if __name__ == "__main__":
    # Stack
    s = Stack()
    for x in [1, 2, 3]:
        s.push(x)
    assert s.size() == 3
    assert s.pop() == 3
    assert s.peek() == 2
    assert not s.is_empty()
    print("Stack: OK")

    # Queue
    q = Queue()
    for x in [1, 2, 3]:
        q.enqueue(x)
    assert q.dequeue() == 1
    assert q.front() == 2
    assert q.size() == 2
    print("Queue: OK")

    # Linked list
    ll = LinkedList()
    for x in [1, 2, 3, 4]:
        ll.append(x)
    ll.prepend(0)
    assert ll.to_list() == [0, 1, 2, 3, 4]
    ll.delete(3)
    assert ll.to_list() == [0, 1, 2, 4]
    assert ll.search(2) and not ll.search(99)
    assert ll.length() == 4
    assert find_middle(ll) == 2        # even length -> second middle
    assert not has_cycle(ll)
    print("LinkedList + find_middle + has_cycle: OK")

    # Cycle detection on a cyclic list
    cyclic = LinkedList()
    cyclic.append(1)
    cyclic.append(2)
    cyclic.append(3)
    cyclic.head.next.next.next = cyclic.head  # 3 -> back to 1
    assert has_cycle(cyclic)
    print("Cycle detection: OK")

    # Hash map
    hm = HashMap()
    hm.put("name", "Alice")
    hm.put("age", 30)
    hm.put("name", "Bob")            # overwrite
    assert hm.get("name") == "Bob"
    assert hm.get("age") == 30
    assert hm.contains("age")
    hm.remove("age")
    assert not hm.contains("age")
    print("HashMap: OK")

    # Problems
    assert is_balanced("{[()]}")
    assert not is_balanced("{[(])}")
    assert is_balanced("(a + b) * [c - {d}]")
    assert reverse_string("hello") == "olleh"

    qws = QueueWithStacks()
    for x in [1, 2, 3]:
        qws.enqueue(x)
    assert qws.dequeue() == 1
    qws.enqueue(4)
    assert qws.dequeue() == 2
    assert qws.dequeue() == 3
    assert qws.dequeue() == 4
    print("is_balanced / reverse_string / QueueWithStacks: OK")

    print("\nALL DATA STRUCTURE SOLUTIONS PASSED [OK]")
