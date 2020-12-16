import numpy as np

balls = ["1", "1", "1", "2", "2", "3-noir"]
while True:
    b = np.random.randint(len(balls))
    print("B", balls[b])
    del balls[b]
    input("")