import sys
import os
import util as u
import subprocess as sp


def disablePrint(): sys.stdout = open(os.devnull, 'w')
def enablePrint(): sys.stdout = sys.__stdout__

def sh(s):
    res = sp.run(s,cwd='.',shell=True,stdout=sp.PIPE,stderr=sp.PIPE)
    return res.stdout.decode("utf-8")


# takes an input string (could be the result of an sh{} or py expr or paren block or whatever --
# at this point it just looks like an 'str' to python in this final generated code
# and takes a list of (idx,cast) tuples
def m_parse(oneline_s,idx_cast_list,delim=' ',try_parselines=True):
    # promote to m_parselines if multiple \n or just 1 \n but it's not the last character of the input
    # can be forced off with try_parselines=False
    if try_parselines:
        if oneline_s.count('\n') > 1 or (oneline_s.count('\n') == 1 and oneline_s[-1] != '\n'):
            return m_parselines(oneline_s,idx_cast_list,delim)

    if oneline_s[-1] == '\n':
        oneline_s = oneline_s[:-1]

    tokens = oneline_s.split(delim)
    tokens = list(filter(None,tokens))
    result = []
    for (idx,cast) in idx_cast_list:
        idx = int(idx)
        cast = eval(cast)
        try:
            result.append(cast(tokens[idx-1]))
        except IndexError:
            print(u.mk_yellow("Warning[line parse failed: out of bounds].\n rule:(idx={},cast={})\n line:{}\n note that (idx-1) is what's actually used".format(idx,cast.__name__,oneline_s)))
            return None
        except ValueError:
            print(u.mk_yellow("Warning[line parse failed: cast failure].\n rule:(idx={},cast={}) => {}({}) failed".format(idx,cast.__name__,cast.__name__,tokens[idx-1])))
            return None

    return tuple(result)

# figures out whether you want to undo parselines or just parse
def m_unparse(input,delim=' '):
    pass
    # input can be a:
        # list of list of tuples from parselines1
        # tuple of list of tuples from parselines
        # list of 


def m_parselines1(multiline_s,idx_cast_list,delim=' '):
    lines = list(filter(None,multiline_s.split('\n')))  # throw out empty lines
    result = list(filter(None,[m_parse(line,idx_cast_list,delim=delim) for line in lines]))
    return result

# trick to turn single list of many tuples into single tuple of many lists
def m_parselines(multiline_s,idx_cast_list,delim=' '):
    result = tuple(zip(*(m_parselines1(multiline_s,idx_cast_list,delim=delim))))
    return tuple([list(x) for x in result]) #convert inner tuples into lists

def m_argv():
    return sys.argv

def m_argc():
    return len(sys.argv)

def m_cat(filename):
    with open(filename,'r') as f:
        text = f.read()
    return text

def m_exists_qmark(the_path):
    return os.path.isfile(the_path)

def m_direxists_qmark(the_path):
    return os.path.isdir(the_path)

# in bytes
def m_fsize(the_path):
    return os.path.getsize(the_path)

def m_fullpath(the_path):
    return os.path.abspath(the_path)

def m_basename(the_path):
    return os.path.basename(the_path)

def m_die(s):
    u.red("Error:"+str(s))
    exit(1)

def m_head(filename,nlines=10):
    with open(filename,'r') as f:
        hd = ''.join([next(f) for x in range(nlines)])
    return hd

def m_p(s):
    print(str(s))

def m_blue(s):
    return u.mk_blue(s)
def m_red(s):
    return u.mk_red(s)
def m_green(s):
    return u.mk_green(s)
def m_yellow(s):
    return u.mk_yellow(s)
def m_purple(s):
    return u.mk_purple(s)

def m_ls(dir='.',regex=None):
    if regex is None:
        return os.listdir(dir)
    else:
        return list(filter(lambda s: re.match(regex,s) is not None, os.listdir(dir)))

def m_mkdir(the_path):
    os.mkdir(the_path)
def m_mkdirp(the_path):
    os.makedirs(the_path)



#m_p(m_parselines('1 hi here\n3 5 here\n6 7 here\n1 2 a',[(1,int),(3,str),(2,int)]))





