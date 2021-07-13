def find_perfects() -> kernel:
    thread_count = blockDim.x * gridDim.x
    index = threadIdx.x + blockDim.x * blockIdx.x

    num = index + 2
    if index == 1000000:
        print("Found!\n")

    while True:
        sum = 1
        sqaure_root = sqrt(num)
        for div in range(2, ceil(sqaure_root)):
            if div == sqaure_root:
                sum += sqaure_root
            elif num % div == 0:
                sum += div + num / div
        
        if sum == num:
            print("%d\n", num)

        num += thread_count


def f(x : double) -> (device, double):
    return sqrt(1-x**2)


def integral(num : double[1]) -> kernel:
    a = -1
    b = 1

    thread_count = blockDim.x * gridDim.x
    index = threadIdx.x + blockDim.x * blockIdx.x

    atomicAdd(num, 2 * f((b-a)*index/thread_count + a)*(b-a) / thread_count)


def main():
    num = [0]
    # num = [0, 1]

    start_device_timer()
    integral[65535, 1024](num)
    elapsed = stop_device_timer()
    deviceSync()

    print(elapsed)

    print(num[0])

    return 0