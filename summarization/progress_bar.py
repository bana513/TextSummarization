import time


class ProgressBar:
    def __init__(self, total_items, start=True, ema=.95):
        self.total_items = total_items
        self.count = 0
        self.start_time, self.end_time = None, None
        self.ema = ema
        self.loss = None

        if start:
            self.start_time = time.time()

    def start(self):
        self.start_time = time.time()
        self.count = 0

    def update(self, item_num=1, loss=None, print_progress=True):
        self.count += item_num
        if loss is not None:
            self.loss = (self.ema * self.loss + (1-self.ema) * loss) if self.loss is not None else loss
        if print_progress:
            self.progress()

    def progress(self):
        elapsed_sec = time.time() - self.start_time
        predicted_sec = (self.total_items - self.count) / self.count * elapsed_sec
        print(f'\r{self.count}\t/{self.total_items}\t({format_sec(elapsed_sec)} - {format_sec(predicted_sec)})' +
              f'\tLoss:{self.loss:2.4f}' if self.loss is not None else "",
              end='')

    def stop(self):
        self.end_time = time.time()
        elapsed_sec = self.end_time - self.start_time
        print(f'Total time: {format_sec(elapsed_sec)}, total items: {self.count}')


def format_sec(seconds):
    s = round(seconds)
    sec = s % 60
    s //= 60
    minute = s % 60
    hour = s // 60
    res = '{0:02.0f}'.format(sec)
    # if minute > 0 or hour > 0:
    res = '{0:02.0f}:'.format(minute) + res
    if hour > 0:
        res = '{0:02.0f}:'.format(hour) + res
    return res
