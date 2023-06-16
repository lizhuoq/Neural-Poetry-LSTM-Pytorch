import torch
from config import Config
from utils import get_data
from torch.utils.data import DataLoader
from model import Model, MaskedSoftmaxCELoss
import torch.nn as nn
from tqdm import tqdm
import time
from loguru import logger


opt = Config()


def train(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v)

    opt.device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')
    device = opt.device
    data, word2ix, ix2word = get_data(opt)
    data = torch.from_numpy(data)
    dataloader = DataLoader(data, batch_size=opt.batch_size, shuffle=True)
    model = Model(len(word2ix), 64, 128)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = MaskedSoftmaxCELoss()
    if opt.model_path:
        model.load_state_dict(torch.load(opt.model_path))
    model.to(device)

    avg_loss = 0
    step = 0
    for epoch in range(opt.epoch):
        for ii, data_ in tqdm(enumerate(dataloader)):
            data_ = data_.long().transpose(1, 0).contiguous()
            data_ = data_.to(device)
            optimizer.zero_grad()
            input_, target = data_[:-1, :], data_[1:, :]
            output, _ = model(input_)
            loss = criterion.forward(output.permute(1, 2, 0), target.transpose(0, 1), \
                                     word2ix['</s>'])
            loss.backward()
            optimizer.step()
            step += 1
            avg_loss += loss
            if step % 100 == 0:
                logger.info("{}\tEpoch: {}\tStep: {}\tLastLoss: {}\tAvgLoss: {}\t".format(
                    time.strftime("[%Y-%m-%d-%H_%M_%S]", time.localtime(time.time())), epoch, step, str(loss.item()), \
                    str(avg_loss/100)
                ))
                avg_loss = 0
            
        torch.save(model.state_dict(), '%s%s_%s.pth' % (opt.save_path, \
            time.strftime("[%Y-%m-%d-%H_%M_%S]", time.localtime(time.time())), epoch))


def generate(model, start_words, prefix_word=None):
    _, word2ix, ix2word = get_data(opt)
    results = list(start_words)
    start_word_len = len(start_words)
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    if opt.use_gpu:
        input = input.cuda()
    hidden = None

    if prefix_word:
        for word in prefix_word:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)

        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == "<EOP>":
            del results[-1]
            break
    return results


if __name__ == "__main__":
    import fire 
    fire.Fire()