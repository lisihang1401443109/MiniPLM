class A:
    def __init__(self):
        self.f()

    def f(self):
        print("A.f")

class B(A):
    def __init__(self):
        super().__init__()
    
    def f(self):
        print("B.f")
        

b = B()
    