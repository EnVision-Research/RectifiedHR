
def next_multiple_of_8(number):
    if number % 8 == 0:
        return int(number)
    else:
        return int((number // 8 + 1) * 8)

def quantic_HW(
    HW_original=(1024, 1024),
    HW_target=(1024, 1024),
    expansion_num=3,
    poly_num=1,
):  
    W, H = HW_target
    W_base, H_base = HW_original
    W, H = next_multiple_of_8(W), next_multiple_of_8(H)
    resolutions = []
    def func(max_number, min_number, x, poly_num):
        return (max_number - min_number) * (x ** poly_num) + min_number
    if H > W:
        r = H / W
        W_diff_W_base = W - W_base
        incre = W_diff_W_base // (expansion_num - 1)
        for i in range(expansion_num):
            # W_ = W_base + i * incre
            W_ = func(W, W_base, i / (expansion_num - 1), poly_num)
            resolutions.append((next_multiple_of_8(W_), next_multiple_of_8(W_ * r)))
    else:
        r = W / H
        H_diff_H_base = H - H_base
        incre = H_diff_H_base // (expansion_num - 1)
        for i in range(expansion_num):
            # H_ = H_base + i * incre
            H_ = func(H, H_base, i / (expansion_num - 1), poly_num)
            resolutions.append((next_multiple_of_8(H_ * r), next_multiple_of_8(H_)))
    return resolutions


def quantic_cfg(min_number, max_number, resize_num, poly_num):
    x = [i / (resize_num - 1) for i in range(resize_num)]
    def func(x):
        return (max_number - min_number) * (x ** poly_num) + min_number
    y = [func(x_) for x_ in x]
    return y

def quantic_step(min_number, max_number, resize_num, poly_num):
    x = [i / (resize_num - 1) for i in range(resize_num)]
    def func(x):
        return int((max_number - min_number) * (x ** poly_num) + min_number)
    y = [func(x_) for x_ in x]
    if y[0] != 0: y.insert(0, 0)
    return y