import torch
from micrograd.engine import Value

def test_sanity_check():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    c.backward()
    a_mg, c_mg = a, c

    a = torch.tensor([-4.0]).double()
    b = torch.tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    c.backward()
    a_pt, c_pt = a, c 
    
    # feedfoward went well
    assert c_mg.data == c_pt.data.item()

    # backward pass went well
    assert a_mg.grad == a_pt.grad.item()

def test_more_ops():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f

    g.backward()
    a_mg, b_mg, g_mg = a, b, g # micrograd values

    
    a = torch.tensor([-4.0]).double()
    b = torch.tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f

    g.backward()
    a_pt, b_pt, g_pt = a, b, g # pytorch values

    tol = 1e-6
    # forward pass went well
    assert abs(g_mg.data - g_pt.data.item()) < tol

    # backward pass went well
    assert abs(a_mg.grad - a_pt.grad.item()) < tol
    assert abs(b_mg.grad - b_pt.grad.item()) < tol