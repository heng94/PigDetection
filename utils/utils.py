class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
        self.avg_reduce = 0.

    def update(self, val, n=1):
        self.val = val
        if n==-1:
            self.sum=val
            self.count=1
        else:
            self.sum += val * n
            self.count += n
        self.avg = self.sum / self.count

    def update_reduce(self, val):
        self.avg_reduce = val

    def __str__(self):
        fmtstr = '{name} {avg_reduce' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)