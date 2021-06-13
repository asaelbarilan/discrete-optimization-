import threading
def call(x):
    return x**2
threads = []
for n in range(1, 4):
    t = threading.Thread(target=call, args=n)
    threads.append(t)
    t.start()

# wait for the threads to complete
for t in threads:
    t.join()
if __name__ == '__main__':
    print(call())