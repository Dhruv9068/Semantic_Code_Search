To set up and test the **Semantic Code Search Engine**, you need to install the required dependencies and follow the steps to create the test repository and run the application. Below is a complete guide:

---

### **Step 1: Install Required Dependencies**
Run the following command to install all the required Python packages:

```bash
pip install flask==3.1.0 numpy==2.2.3 faiss-cpu==1.10.0 sentence-transformers==3.4.1 pygments==2.19.1 chardet==5.2.0 torch==2.6.0
```

---

### **Step 2: Create the Test Repository**
1. Create a new directory for the test files:
   ```bash
   mkdir test_repository
   cd test_repository
   ```

2. Create the following test files in the `test_repository` directory:

#### **1. `sorting_algorithms.py`**
```python
# Sorting Algorithms in Python

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
```

#### **2. `data_structures.py`**
```python
# Data Structures in Python

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, value):
        new_node = Node(value)
        if not self.head:
            self.head = new_node
            return
        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        last_node.next = new_node
```

#### **3. `file_operations.py`**
```python
# File Operations in Python

import json

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
```

#### **4. `calculator.js`**
```javascript
// Calculator in JavaScript

function add(a, b) {
    return a + b;
}

function subtract(a, b) {
    return a - b;
}

function multiply(a, b) {
    return a * b;
}

function divide(a, b) {
    if (b === 0) throw new Error("Cannot divide by zero");
    return a / b;
}
```

#### **5. `user_management.java`**
```java
// User Management in Java

public class User {
    private String username;
    private String password;

    public User(String username, String password) {
        this.username = username;
        this.password = password;
    }

    public boolean authenticate(String inputPassword) {
        return this.password.equals(inputPassword);
    }
}
```

---

### **Step 3: Run the Semantic Code Search Engine**
1. Make sure your Semantic Code Search Engine is running. If you have an `app.py` file, run:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

### **Step 4: Index the Test Repository**
1. In the "Index Repository" section, enter the full path to your `test_repository` folder. For example:
   - On Windows: `C:\Users\YourName\test_repository`
   - On macOS/Linux: `/home/username/test_repository`

2. Click "Index Repository" and wait for the indexing process to complete.

---

### **Step 5: Test Searching**
Try the following search queries to test the semantic search capabilities:

1. **Algorithm Searches**:
   - "sorting algorithm that uses divide and conquer"
   - "bubble sort implementation"
   - "algorithm with O(n log n) complexity"

2. **Data Structure Searches**:
   - "linked list implementation"
   - "binary search tree"
   - "stack data structure"

3. **File Operation Searches**:
   - "read json file"
   - "function to copy files"
   - "csv file handling"

4. **Language-Specific Searches**:
   - "javascript calculator"
   - "java user management"
   - "python data structures"

5. **Concept Searches**:
   - "authentication system"
   - "memory management"
   - "factorial calculation"

---

### **Step 6: Evaluate Results**
For each search:
- Check the relevance of the results.
- Verify if the code snippets match the query.
- Note the similarity scores.
- Ensure the system correctly identifies the programming language.

---

### **Advanced Testing**
1. **Try Complex Queries**:
   - "function that handles errors when reading files"
   - "data structure with O(1) insertion time"
   - "authentication with password hashing"

2. **Test Language Understanding**:
   - "find me code that sorts arrays"
   - "show me how to handle JSON data"
   - "I need to implement a user login system"

---

### **Requirements**
Here are the dependencies you need to install:

```plaintext
flask==3.1.0
numpy==2.2.3
faiss-cpu==1.10.0
sentence-transformers==3.4.1
pygments==2.19.1
chardet==5.2.0
torch==2.6.0
```

Run the following command to install them:
```bash
pip install flask==3.1.0 numpy==2.2.3 faiss-cpu==1.10.0 sentence-transformers==3.4.1 pygments==2.19.1 chardet==5.2.0 torch==2.6.0
```

---

Let me know if you need further assistance! 🚀
