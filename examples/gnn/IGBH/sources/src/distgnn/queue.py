class gqueue():
    def __init__(self):
        self.queue = []

    def push(self, val):
        self.queue.append(val)

    def pop(self):
        if self.empty():
            return -1
        return self.queue.pop(0)

    def size(self):
        return len(self.queue)

    def printq(self):
        print(self.queue)

    def empty(self):
        if (len(self.queue)) == 0:
            return True
        else:
            return False

    def reset(self):
        self.queue.clear()
        assert len(self.queue) == 0

class gstack():
    def __init__(self):
        self.stack = []

    def push(self, val):
        self.stack.append(val)

    def pop(self):
        if self.empty():
            return -1
        return self.stack.pop()

    def size(self):
        return len(self.stack)

    def printq(self):
        print(self.stack)

    def empty(self):
        if (len(self.stack)) == 0:
            return True
        else:
            return False

    def purge(self):
        self.stack = []
