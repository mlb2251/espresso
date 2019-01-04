
from importlib import reload

import os
homedir = os.environ['HOME']
src_path = homedir+'/espresso/src'

def die(s):
    red("ERROR:"+s)
    exit(1)

def warn(s):
    print(mk_yellow("WARN:"+s))

def pretty_path(p):
    return p.replace(homedir,'~')

class PrettifyErr(Exception): pass
class SafeImportErr(Exception): pass

import traceback as tb
def exception_str(e):
    return ''.join(tb.format_exception(e.__class__,e,e.__traceback__))

# relevant_path_piece is just used to remove a lot of irrelevant parts of exceptions.
# for example it might be '/espresso/src/' to ignore all lines of traceback before
# the first one mentioning '/espresso/src/' at any point
# "verbose" option will print out the whole original exception in blue followed by
# the formatted one
def format_exception(e,relevant_path_piece,tmpfile=None,verbose=False):
    if verbose:
        blue(exception_str(e))
    try:
        raw_tb = tb.format_exception(e.__class__,e,e.__traceback__)
        assert(raw_tb[0] == 'Traceback (most recent call last):\n')
        raw_tb = raw_tb[1:-1]   #rm first+last. last is a copy of str(e)
        #blue(''.join(raw_tb))
        #magenta(raw_tb)
        formatted = raw_tb


        # any lines not starting with 'File' are appended to the last seen
        # line that started with 'File'
        # (this is for standardization bc the exceptions arent that standardized)
        # all this should really be done with whatever python under the hood
        # is generating exceptions haha
        for (i,s) in enumerate(formatted):
            if s.strip()[:4] == 'File':
                lastfile = i
            else:
                formatted[lastfile] += s
        # now remove those leftover copies that dont start with File
        formatted = list(filter(lambda s: s.strip()[:4] == 'File',formatted))

        # delete everything before the first line that relates to the tmpfile
        #for (i,s) in enumerate(raw_tb):
        #    if relevant_path_piece in s:
        #        formatted = raw_tb[i:]
        #        break
        formatted = list(filter(lambda s: relevant_path_piece in s, formatted))


        # for a traceback segment that looks like (second line after \n is optional, only shows up sometimes):
        #File "/Users/matthewbowers/espresso/repl-tmpfiles/113/a_out.py", line 8, in <module>\n      test = 191923j2k9E # syntax error
        def try_pretty_tb(s):
            try:
                return pretty_tb(s)
            except Exception as e:
                warn("(ignorable) Error during prettifying traceback component.\ncomponent={}\nreason={}".format(s,exception_str(e)))
                return [s,'']
        def pretty_tb(s):
            if s[-1] == '\n': s = s[:-1]
            if s.count('\n') == 0:
                includes_code = includes_arrow = False
                msg = s
            elif s.count('\n') == 1:
                includes_code=True
                includes_arrow=False
                [msg,code_line] = s.split('\n')
            elif s.count('\n') == 2:
                includes_code = includes_arrow = True
                [msg,code_line,arrow] = s.split('\n')
            elif s.count('\n') > 2:
                raise PrettifyErr('more than 3 lines in a single traceback component:{}')

            if msg.count(',') == 1:
                includes_mod = False
                [fpath,lineno] = msg.split(',')
            elif msg.count(',') == 2:
                includes_mod = True
                [fpath,lineno,modname] = msg.split(',')
            else:
                raise PrettifyErr('unexpected number of commas in traceback component line:{}'.format(s))


            # prettify file name (see example text above function)
            assert(len(fpath.split('"'))==3)
            fpath = fpath.split('"')[1] # extract actual file name
            if fpath == tmpfile:
                fname = "tmpfile"
            else:
                fname = os.path.basename(fpath)
            fpath_abs = fpath
            fpath = fpath.replace(homedir,'~')

            # prettify line number (see example text above function)
            lineno = lineno.strip()
            assert(len(lineno.split(' '))==2)
            lineno_white = lineno.split(' ')[1]
            lineno = lineno.split(' ')[1]

            # prettify module name
            if includes_mod:
                modname = modname.strip()
                assert(len(modname.split(' '))==2)
                modname = modname.split(' ')[1]
                if modname == '<module>':
                    modname = ''
                else:
                    modname = '('+modname+')'
            else:
                modname = ''

            # build final result
            result = "{} {} {}".format(
                    mk_underline(mk_red(fname+" @ "+lineno)),
                    mk_gray('('+fpath+')'),
                    mk_green(modname)
                    )
            command = "nvim +{} {}".format(lineno,fpath_abs)
            if includes_code:
                code_num_spaces = code_line.index(code_line.strip()[0])
                code_line = code_line.strip()
                lineno_fmtd = '{:>6}: '.format(lineno)
                lineno_width = len(lineno_fmtd)
                result += '\n{}{}'.format(
                        mk_bold(mk_green(lineno_fmtd)),
                        mk_yellow(code_line)
                        )
            if includes_arrow:
                arrow_num_spaces = arrow.index(arrow.strip()[0])
                offset = lineno_width + arrow_num_spaces - code_num_spaces
                result += '\n{}{}'.format(' '*offset,mk_bold(mk_green('^')))
            return (result,command)

        res = [try_pretty_tb(s) for s in formatted]
        (formatted,commands) = zip(*res)
        formatted = ['',mk_red(e)] + list(formatted) + ['']
        return ('\n'.join(formatted), commands)
    except Exception as e2:
        warn("(ignorable) Failed to Prettify exception, using default format. Note that the prettifying failure was due to: {}".format(exception_str(e2)))
        return (mk_red(exception_str(e)),[])




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

def mk_underline(s):
    return cols.UNDERLINE+str(s)+cols.ENDC
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
def purple(msg):
    pc(cols.MAGENTA,msg)
def blue(msg):
    pc(cols.OKBLUE,msg)
def yellow(msg):
    pc(cols.YELLOW,msg)
def green(msg):
    pc(cols.OKGREEN+cols.BOLD,msg)
