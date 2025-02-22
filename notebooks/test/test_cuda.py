import pycuda.driver as cuda
import pycuda.autoinit


def test_cuda():
    try:
        cuda.init()
        print("CUDA Successfully Installed")
        print("CUDA Device Count:", cuda.Device.count())
        for i in range(cuda.Device.count()):
            dev = cuda.Device(i)
            print("Device Name:", dev.name())
            print("Device Memory :", dev.total_memory() // (1024 * 1024), "MB")
        return True
    except Exception as e:
        print("CUDA Installation Error", e)
        return False


test_cuda()
