import numpy as np
import argparse
from timeit import default_timer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="assets/model/model.h5")
    parser.add_argument("--total-samples", type=int, default=10)

    args = parser.parse_args()

    from src.app import D3DScreenshot, MSSScreenshot
    bounding_box = (0, 0, 400, 500)
    d3d_screenshotter = D3DScreenshot()
    mss_screenshotter = MSSScreenshot()

    print("Taking {:d} samples".format(args.total_samples))

    dt = profile_time(d3d_screenshotter.get_screen_shot, args.total_samples)(bounding_box)
    print("D3D took {:.02f}ms".format(dt*1000))

    dt = profile_time(mss_screenshotter.get_screen_shot, args.total_samples)(bounding_box)
    print("MSS took {:.02f}ms".format(dt*1000))

    from src.model import Model
    model = Model.load(args.model)
    test_image = np.full((1, 160, 227, 3), 125)/255
    dt = profile_time(model.predict, args.total_samples)(test_image)
    print("Model took {:.02f}ms".format(dt*1000))



def profile_time(func, n):
    def wrapper(*args, **kwargs):
        times = []
        for _ in range(n):
            start = default_timer()
            func(*args, **kwargs)
            end = default_timer()
            dt = end-start
            times.append(dt)
        average_dt = sum(times)/n
        return average_dt
    return wrapper

if __name__ == '__main__':
    main()
    


