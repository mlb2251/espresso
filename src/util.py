
def die(s):
    red("ERROR:"+s)
    exit(1)


class cols:
    MAGENTA = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    YELLOW = '\033[33m'
    CYAN = '\033[36m'
    GRAY = '\033[90m'


    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def mk_green(s):
    return cols.OKGREEN+str(s)+cols.ENDC
def mk_red(s):
    return cols.FAIL+str(s)+cols.ENDC
def mk_purple(s):
    return cols.MAGENTA+str(s)+cols.ENDC
def mk_blue(s):
    return cols.OKBLUE+str(s)+cols.ENDC
def mk_cyan(s):
    return cols.CYAN+str(s)+cols.ENDC
def mk_yellow(s):
    return cols.YELLOW+str(s)+cols.ENDC
def mk_gray(s):
    return cols.GRAY+str(s)+cols.ENDC
def mk_bold(s):
    return cols.BOLD+str(s)+cols.ENDC


def pc(color,msg):
    print(color+str(msg)+cols.ENDC)
def red(msg):
    pc(cols.BOLD+cols.FAIL,msg)
def red_thin(msg):
    pc(cols.FAIL,msg)
def magenta(msg):
    pc(cols.MAGENTA,msg)
def blue(msg):
    pc(cols.OKBLUE,msg)
def green(msg):
    pc(cols.OKGREEN+cols.BOLD,msg)
