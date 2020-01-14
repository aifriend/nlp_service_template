class SparseVector:

    def __init__(self):
        self.data = []
        self.size = 0

    def fromList(self, lst):
        self.size = len(lst)
        pairs = [(i, lst[i]) for i in range(len(lst))]
        print(self.size)
        self.data = list(filter(lambda x: x[1] != 0, pairs))

    def toList(self):
        pass

    def __str__(self):
        return str(self.data)
