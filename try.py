import torch
from config.harper_config import config
from network.AINet import AINet

if __name__ == '__main__':
    b, n, d1, d2 = 2, 60, 63, 69  #
    x = torch.rand(b, n, d1).cuda()
    y = torch.rand(b, n, d2).cuda()


    model = AINet(config)
    model.cuda()
    out = model(x, y)
    print(out[0].shape, out[1].shape
          )