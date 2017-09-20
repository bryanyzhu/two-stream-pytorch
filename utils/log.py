import logging
import time,os
import ConfigParser


class log(object):
    # root logger setting

    save_path = time.strftime("%m_%d_%H_%M") + '.log'
    l = logging.getLogger()
    l.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # clear handler streams
    for it in l.handlers:
        l.removeHandler(it)

    # file handler setting
    config = ConfigParser.RawConfigParser()
    config.read('utils/util.config')
    save_dir = config.get('general', 'log_path')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_path)

    f_handler = logging.FileHandler(save_path)
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(formatter)

    # console handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_handler.setFormatter(formatter)

    l.addHandler(f_handler)
    l.addHandler(c_handler)


    # print(l.handlers[0].__dict__)
