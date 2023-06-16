class Config():
    data_path = 'data/'
    filename = 'tang.npz'
    use_gpu = False
    batch_size = 128
    lr = 1e-3
    save_path = 'models/'
    max_gen_len = 200
    model_path = None
    epoch = 10