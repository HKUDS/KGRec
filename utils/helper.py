'''
Created on Aug 19, 2016
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
__author__ = "xiangwang"
import os
import re
import logging

def txt2list(file_src):
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines


def ensureDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def uni2str(unicode_str):
    return str(unicode_str.encode('ascii', 'ignore')).replace('\n', '').strip()


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def delMultiChar(inputString, chars):
    for ch in chars:
        inputString = inputString.replace(ch, '')
    return inputString

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        # print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

def ensure_file(dir_name, filename):
    log_files = os.listdir(dir_name)
    if filename in log_files:
        filename = filename.split('.')
        if len(filename) == 2:
            filename = filename[0] + '.1.' + filename[1]
        else:
            filename = filename[0] + '.' + str(int(filename[1]) + 1) + '.' + filename[2]
        return ensure_file(dir_name, filename)
    return filename

def init_logger(args, subdir=None):
    enable_fh = args.log
    logfilename = args.log_fn
    dataset = args.dataset

    LOGROOT = './logs/'+dataset+'/'
    dir_name = os.path.dirname(LOGROOT)
    if subdir is not None:
        dir_name = os.path.join(dir_name, subdir)

    ensureDir(dir_name)

    fileformatter = logging.Formatter("%(message)s")

    sformatter = logging.Formatter("%(message)s")

    level = logging.NOTSET

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    if enable_fh and logfilename is not None:
        logfilename = logfilename+f'_{dataset}'+'.log'
        logfilename = ensure_file(dir_name, logfilename)
        logfilepath = os.path.join(dir_name, logfilename)
        fh = logging.FileHandler(logfilepath)
        fh.setLevel(level)
        fh.setFormatter(fileformatter)
        logging.basicConfig(level=level, handlers=[sh, fh])
        return logfilename.strip('.log')
    else:
        logging.basicConfig(level=level, handlers=[sh])
        return None
