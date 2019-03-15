from __future__ import print_function
import torch 
import numpy as np 

x = torch.empty(5,3)
#print(x)
x = x.new_ones(5,3,dtype = torch.double)
# print(x)
x = torch.randn_like(x,dtype = torch.float)
print(x)

# print(x.size())

y = torch.rand(5,3)
print(y)

# z = torch.zeros(5,3,dtype = torch.long)
# print(z)

# tensor = torch.tensor([5.5,3])
# print(tensor)

print(x+y)
print(torch.add(x,y))

y.add_(x)
print(y)

# any mutations of the original array would need to be called on function_ 
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1,8)
print(x.size(),y.size(),z.size())
print(x)
print(y)
print(z)


if torch.cuda.is_available():
	device = torch.device("cuda")
	y = torch.ones_like(x,device = device)
	x = x.to(device)
	print("is_available")


