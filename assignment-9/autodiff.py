#!/usr/bin/env python3

import math
import random

class Expression(object):

    def __init__(self, parents):
        self.value = None
        self.gradient_value = 0
        self.parents_done = 0
        self.parents_active = 0
        self.children = []
        self.parents = list(expr(p) for p in parents)
        for parent in self.parents:
            parent.children.append(self)

    ##########################################################################
    # debugging
    
    def debug_print(self, indent=0):
        istr = ' ' * indent
        print("%s%s[0x%x]:" % (istr, self.__class__.__name__, id(self)))
        print("%s  value: %s" % (istr, self.value))
        print("%s  gradient_value: %s" % (istr, self.gradient_value))
        print("%s  parents_done: %s" % (istr, self.parents_done))
        print("%s  parents_active: %s" % (istr, self.parents_active))
        if len(self.parents) > 0:
            print("%s  parents:" % istr)
            for parent in self.parents:
                parent.debug_print(indent + 4)

    ##########################################################################
    # interface for subclasses
    
    def compute_value(self, *args):
        pass
    
    def push_derivative(self, *args):
        pass

    def reset(self):
        self.parents_active = 0
        self.parents_done = 0
        self.gradient_value = 0
        self.value = None

    ##########################################################################
    # forward pass

    def evaluate(self):
        if self.value is not None:
            return self.value
        vals = []
        for parent in self.parents:
            parent.parents_active += 1
            vals.append(parent.evaluate())
        self.value = self.compute_value(*vals)
        return self.value

    ##########################################################################
    # backward pass

    def backward(self):
        # mark parents_done here so that clear_eval() works as intended
        self.parents_done = 1
        self.parents_active = 1
        self.gradient_value = 1
        for parent in self.parents:
            parent.parents_done += 1
        self.push_derivative(*self.parents)
        for parent in self.parents:
            parent.backward_rec()

    def backward_rec(self):
        if self.parents_done < self.parents_active:
            # not all parents have pushed yet
            return
        for parent in self.parents:
            parent.parents_done += 1
        self.push_derivative(*self.parents)
        for parent in self.parents:
            parent.backward_rec()

    ##########################################################################
    # cleanup of evaluation objects for reuse
            
    def reverse_dependencies(self, d=None):
        if d is None:
            d = {}
        if id(self) in d:
            return
        d[id(self)] = self
        for parent in self.parents:
            parent.reverse_dependencies(d)
        return d
            
    def clear_eval(self):
        deps = self.reverse_dependencies()
        for v in deps.values():
            v.reset()

    ##########################################################################
    # operator overloading, for notational convenience
    
    def __add__(self, other):      return Add(self, other)
    def __radd__(self, other):     return Add(other, self)
    def __sub__(self, other):      return Sub(self, other)
    def __rsub__(self, other):     return Sub(other, self)
    def __neg__(self):             return Neg(self)
    def __mul__(self, other):      return Mul(self, other)
    def __rmul__(self, other):     return Mul(other, self)
    def __truediv__(self, other):  return Div(self, other)
    def __rtruediv__(self, other): return Div(other, self)
    def __pow__(self, other):      return pow(self, other)

def expr(obj):
    if isinstance(obj, Expression):
        return obj
    elif isinstance(obj, float) or isinstance(obj, int):
        return Var(float(obj))
    else:
        raise Exception("Don't know how to turn %s into an Expression" % obj)
    
##############################################################################

class Var(Expression):
    def __init__(self, value, name=None):
        Expression.__init__(self, [])
        self.name = name
        self.value = value
        self.name = name
    def reset(self):
        self.parents_active = 0
        self.parents_done = 0
        self.gradient_value = 0
        
class Add(Expression):
    def __init__(self, *parents):
        Expression.__init__(self, parents)
    def compute_value(self, *values):
        return sum(values)
    def push_derivative(self, *parents):
        v = self.gradient_value
        for p in parents:
            p.gradient_value += v


class Sub(Expression):
    def __init__(self, left, right):
        Expression.__init__(self, [left, right])
    def compute_value(self, v1, v2):
        return v1 - v2
    def push_derivative(self, p1, p2):
        v = self.gradient_value
        p1.gradient_value += v
        p2.gradient_value -= v

################################################################################
# for the classes below, implement the methods compute_value and push_derivative
        
class Neg(Expression):
    def __init__(self, v):
        Expression.__init__(self, [v])
    def compute_value(self,v1):
        return -v1
    def push_derivative(self,p1):
        v = self.gradient_value
        p1.gradient_value -= v

class Mul(Expression):
    def __init__(self, left, right):
        Expression.__init__(self, [left, right])
    def compute_value(self, v1 , v2):
        return v1 * v2
    def push_derivative(self, p1 , p2):
        v = self.gradient_value
        p1.gradient_value += (v * p2.value)
        p2.gradient_value += (v * p1.value)

class Div(Expression):
    def __init__(self, left, right):
        Expression.__init__(self, [left, right])
    def compute_value(self, v1 , v2):
        return v1 / v2
    def push_derivative(self, p1, p2):
        v = self.gradient_value
        p1.gradient_value += (v * (1/p2.value))
        p2.gradient_value += (v * (-(p1.value) / ((p2.value)**2)))

# exponential
class exp(Expression):
    def __init__(self, v):
        Expression.__init__(self, [v])
    def compute_value(self, v1):
        return math.exp(v1)
    def push_derivative(self, p1):
        v = self.gradient_value
        p1.gradient_value += (v * math.exp(p1.value))

# logarithm
class log(Expression):
    def __init__(self, v):
        Expression.__init__(self, [v])
    def compute_value(self, v1):
        return math.log(v1)
    def push_derivative(self, p1):
        v = self.gradient_value
        p1.gradient_value += (v / p1.value)

class Max(Expression):
    def __init__(self, *parents):
        Expression.__init__(self, parents)
    def compute_value(self, *values):
        return max(values)
    def push_derivative(self, *parents):
        v = self.gradient_value
        
        m = parents[0].value
        idx = 0
        for i in range(len(parents)):
            p = parents[i]
            if p.value > m:
                m = p.value
                idx = i
        
        for i in range(len(parents)):
            if (i == idx):
                parents[i].gradient_value += v

# you won't need these but you might want to play around with them anyway
class sin(Expression):
    def __init__(self, v):
        Expression.__init__(self, [v])
    def compute_value(self, v1):
        return math.sin(v1)
    def push_derivative(self, p1):
        v = self.gradient_value
        p1.gradient_value += (v * math.cos(p1.value))

class cos(Expression):
    def __init__(self, v):
        Expression.__init__(self, [v])
    def compute_value(self , v1):
        return math.cos(v1)
    def push_derivative(self, p1):
        v = self.gradient_value
        p1.gradient_value += (v * (-(math.sin(p1.value))))

# This assumes a constant power!!
class pow(Expression):
    def __init__(self, v, k):
        self.k = float(k)
        Expression.__init__(self, [v])
    def compute_value(self, v1):
        return v1**(self.k)
    def push_derivative(self, p1):
        v = self.gradient_value
        p1.gradient_value += (v * (self.k) * ((p1.value)**(self.k - 1)))


##############################################################################
# linear algebra API, for convenience

# in real implementations, proper initialization of the weights
# matters. This is a very simplistic strategy.
def random_vector(dim):
    return Vector(list(Var(random.random() - 0.5) for d in range(dim)))
    
def zero_vector(dim):
    return Vector(list(Var(0.0) for d in range(dim)))

def dot(vec1, vec2):
    return Add(*(v1 * v2 for (v1, v2) in zip(vec1.vals, vec2.vals)))

class Vector(object):
    def __init__(self, vals):    
        self.vals = vals
    def __add__(self, other):
        return Vector(list(a + b) for (a,b) in zip(self.vals, other.vals))
    def __sub__(self, other):
        return Vector(list(a - b) for (a,b) in zip(self.vals, other.vals))
    def __neg__(self):
        return Vector(list(-a) for a in self.vals)
    def __mul__(self, val):
        return Vector(list(v * val for v in self.vals))
    def __rmul__(self, val):
        return Vector(list(val * v for v in self.vals))
    def __truediv__(self, val):
        return Vector(list(v / val for v in self.vals))
    def __getitem__(self, k):
        return self.vals[k]
    def map_over(self, f):
        return Vector(list(f(v) for v in self.vals))
    def evaluate(self):
        return list(v.evaluate() for v in self.vals)
    def reverse_dependencies(self, d=None):
        if d is None:
            d = {}
        if id(self) in d:
            return
        d[id(self)] = self
        for parent in self.parents:
            parent.reverse_dependencies(d)
        return d
    def clear_eval(self):
        d = {}
        for v in self.vals:
            v.reverse_dependencies(d)
        for v in d.values():
            v.reset()

##############################################################################
# neural-net helper functions

class LinearLayer(object):
    def __init__(self, n, m):
        self.rows = list(random_vector(n) for i in range(m))
    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector(list(dot(row, other) for row in self.rows))
    def evaluate(self):
        return list(v.evaluate() for v in self.rows)

# ReLu over a Vector (aka an entire layer of neurons)
def ReLU(vec):
    return vec.map_over(relu)

def softmax(vec):
    d = len(vec.vals)
    
    if (d == 1):
        sum = exp(vec.vals[0])
    else:
        sum = exp(vec.vals[0]) + exp(vec.vals[1])
        for i in range(2,d):
            sum += exp(vec.vals[i])

    results = []
    for i in range(d):
        results.append((exp(vec.vals[i]) / sum))
    
    return Vector(results)
    
def relu(v):
    return Max(Var(0.0) , v)	

def cross_entropy_loss(yhat_vec, y_vec):
    M = len(y_vec.vals)
    
    sum = y_vec[0] * log(yhat_vec[0])
    for i in range(1,M):
        sum = sum + (y_vec[i] * log(yhat_vec[i]))
    sum = (-sum)
    
    return sum
    

def l2_loss(fx , y):
    return (fx - y)*(fx - y)

##############################################################################

def test_1():
    x1 = Var(1, 'x1')
    x2 = Var(1, 'x2')
    y1 = Var(0, 'y1')
    y2 = Var(0, 'y2')
    le = log(1 + exp(-(x1 * y1 + x2 * y2)))
    le.clear_eval()
    print(le.evaluate())
    le.backward()
    # this should replicate the example we did in class
    print(x1.gradient_value,
          x2.gradient_value,
          y1.gradient_value,
          y2.gradient_value)

def test_2():
    a = Var(1)
    b = Var(1)
    c = Var(1)
    d = Vector([a, b, c])
    d = softmax(d)

    # the gradient values should be "symmetric", that is, the set of values
    # in the calls below should be the same, but in a different order

    d[1].clear_eval()
    print(d[1].evaluate())
    d[1].backward()
    print(a.gradient_value, b.gradient_value, c.gradient_value)

    d[0].clear_eval()
    print(d[0].evaluate())
    d[0].backward()
    print(a.gradient_value, b.gradient_value, c.gradient_value)

    d[2].clear_eval()
    print(d[2].evaluate())
    d[2].backward()
    print(a.gradient_value, b.gradient_value, c.gradient_value)
    
def test_3():
    a = Var(2)
    b = 1 / a
    b.evaluate()
    b.backward()
    
    print (a.gradient_value , b.value)

    # a.gradient_value should be 0.25
    # b.value should be 0.5

def test_4():
    # gradient descent to find minimum of x^2 - 5x - 6 = 0
    x = Var(0)
    a = Var(1)
    b = Var(-5)
    c = Var(-6)
    y = Var(0)
    fx = a * x ** 2 + b * x + c
    l = l2_loss(fx, y)
    alpha = 0.001
    for i in range(1000):
        l.clear_eval()
        v = l.evaluate()
        l.backward()
        print("\r                         \r %.2f %.2f %.2f" % (x.value, x.gradient_value, v))
        x.value -= x.gradient_value * alpha
    
if __name__ == '__main__':
    test_1()
    test_2()
    test_3()
    test_4()
