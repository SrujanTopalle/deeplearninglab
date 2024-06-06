import numpy as np
import matplotlib.pyplot as plt

def graph(inp,op,label):
  plt.plot(inp,op)
  plt.xlabel('x')
  plt.ylabel(label+'(x)')
  plt.title(label)
  plt.show()

def sigmoid(inp):
  op = 1/(1+np.exp(-inp))
  print(f"Input:\n{inp}\n\n\nOutput:\n{op}")
  graph(inp,op,'Sigmoid')

def hyperbolic(inp):
  op = np.tanh(inp)
  print(f"Input:\n{inp}\n\n\nOutput:\n{op}")
  graph(inp,op,'Tanh')

def relu(inp):
  op = np.maximum(0,inp)
  print(f"Input:\n{inp}\n\n\nOutput:\n{op}")
  graph(inp,op,'ReLU')

def leaky_relu(inp):
  op = np.maximum(0.1*inp,inp)
  print(f"Input:\n{inp}\n\n\nOutput:\n{op}")
  graph(inp,op,'Leaky ReLU')

def softmax(inp):
  op = np.exp(inp)/np.sum(np.exp(inp))
  print(f"Input:\n{inp}\n\n\nOutput:\n{op}")
  graph(inp,op,'Softmax')

#input
inp = np.linspace(-10,10,100)

#menu
while(True):
  print("1.Sigmoid")
  print("2.Tanh")
  print("3.ReLU")
  print("4.Leaky ReLU")
  print("5.Softmax")
  print("6.Exit")
  choice = int(input("Enter your choice: "))
  if choice == 1:
    sigmoid(inp)
  elif choice == 2:
    hyperbolic(inp)
  elif choice == 3:
    relu(inp)
  elif choice == 4:
    leaky_relu(inp)
  elif choice == 5:
    softmax(inp)
  elif choice == 6:
    break
  else:
    print("Invalid choice")
