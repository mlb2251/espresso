# REPL
import sys
import os
from importlib import reload
import traceback as tb
import ast

import codegen
import util as u
from util import *

###############################
###############################
# TODO solution to many issues:
# enable CustomCompleterDisplayHook
# make ' ' '\t' no longer delims so that 'text' is the full line
# modify CustomCompleterDisplayHook hook to only print the part of the match at the end of the line
# This enables us to do repls on the entire contents of the line! And '\ ' will not print funny or
# anything like that anymore.
###############################
###############################


# This segment sets up the interpreter to have history with arrows and tab based autocomplete
import readline
import rlcompleter
import shlex

# When attempting tab completion, completer() is called repeatedly until it returns
# None. Each time it is called it should return the next item in the list of possible
# completions. Its first call is done with state=0, then state=1, etc.
# Therefore the actual os.listdir() only happens in state==0, and in all other states
# we simply return the approrpiate file by indexing our file list using the 'state' int
## Note one quirk: if there's a '\ ' in the filename the tab completion will only
## show you the bit after the '\ ', tho in terms of behavior it'll act fine. This
## is just bc we need to specify ' ' as a delimiter with set_completer_delim()
readline.set_completer_delims(' \t\n/')


class CustomCompleterDisplayHook:
    def __init__(self,repl):
        self.repl=repl
    def __call__(self,searchtext,matches,longest_match_len):
        term_width = os.popen('stty size', 'r').read().split()[1]
        # TODO use term_width to pretty print the matches, also add color
        print()
        print('  '.join(matches))
        print(self.repl.state.banner,readline.get_line_buffer(),sep='', end='')
        sys.stdout.flush()

def completer(text, state):
    if state == 0:
        # completer only gets the text AFTER the last ' '
        # readline.get_line_buffer to get full line
        # then grab the last item on the linewith shlex.split 
        # *Note this only really is the deal with the '\ ' case
        line = readline.get_line_buffer()
        lastitem = shlex.split(line)[-1]
        if lastitem[:2] == '~/':
            lastitem = os.path.expanduser('~') + lastitem[1:]

        # the folder to ls and the partial filename to use
        folder = os.path.dirname(lastitem)
        partial_fname = os.path.basename(lastitem)
        if folder == '':
            folder = '.'

        # TODO can expand on this however you want, enabling '*' for example or whatever else
        # (unfortunately the '*' may not work if it isn't successfully contained within the 'text'
        # variable because thats what ends up getting replaced, so if * happens to act as a separator
        # for 'text' like '-' is for example, it wouldn't work. Also you certainly couldn't have
        # a '*' earlier in the line for the same substitution reason. 
        # Tho if * expanded to exactly 1 thing maybe you could do it actually, tho the expansion
        # wouldn't happen right here

        # TODO readline.set_completion_display_matches_hook (custom displaying e.g. could color!!)

        # could also be useful:
        # readline.set_startup_hook readline.set_pre_input_hook
        # readline.insert_text

        files = os.listdir(folder)
        for i in range(len(files)): # append '/' if it's a folder
            if os.path.isdir(folder+'/'+files[i]):
                files[i] += '/'
        completer.options = [i for i in files if i.startswith(partial_fname)]
        completer.partial_fname = partial_fname
        completer.text = text
    if state < len(completer.options):
        # This simply returns the next tab completion result: completer.options[state].
        # Unfortunately it looks a little more complex than that because of this situation:
        # imagine you're completing 'some\ te' to get the file 'some\ test.txt'
        # thus partial_fname = 'some\ te'
        # and text = 'te' (due to readline stupidity)
        # however the completer() is supposed to return whatever is replacing _text_ not
        # whatever is replacing partial_fname, since _text_ is the official thing it gave us.
        # So we actually want to return 'st.txt' rather than 'some-test.txt'
        # hence this substringing / fancy indexing. 
        return completer.options[state][completer.partial_fname.rindex(completer.text):].replace(' ',r'\ ')
    else:
        return None

readline.parse_and_bind("tab: complete")
readline.set_completer(completer)
histfile = os.path.join(u.error_path+'eshist')
try:
    readline.read_history_file(histfile)
    # default history len is -1 (infinite), which may grow unruly
    readline.set_history_length(1000)
except IOError:
    pass
import atexit
atexit.register(readline.write_history_file, histfile)


del histfile, rlcompleter





# this is the Repl called from main.py
# Repl.next() is the fundamental function that gets input, parses it, and runs it
class Repl:
    def __init__(self,state):
        self.state=state #self.state is a ReplState (defined in main.py, intentionally not here for reload() reasons)
        #readline.set_completion_display_matches_hook(CustomCompleterDisplayHook(self)) # TODO uncomment

    def get_state(self):
        return self.state

    def next(self):
        if self.state.banner_cwd != os.getcwd(): self.update_banner()
        line = self.get_input()
        if line is None: return
        if len(line.strip()) == 0: return
        # for convenience, 'ls' or 'cd [anything]' will change the mode to speedy
        if (line.strip() == 'ls' or line.strip()[:3] == 'cd ') and self.state.mode == 'normal':
            print(mk_gray('implicit switch to speedy mode'))
            self.state.mode = 'speedy'
            self.update_banner()

        # deal with ! commands
        if self.try_metacommands(line):
            return

        new_code = self.gen_code(line)
        self.run_code(new_code)

    # handles ! commands and returns 'False' if none were detected
    # (indicating the line should be parsed normally)
    def try_metacommands(self,line):
        if len(line) == 0: return False
        if line.strip() == '%':
            if self.state.mode != 'speedy':
                self.state.mode = 'speedy'
            else:
                self.state.mode = 'normal'
            self.update_banner()
            print(mk_gray("metacommand registered"))
            return True
        # handle metacommands
        if line[0] == '!':
            if line.strip() == '!print':
                print(mk_yellow('\n'.join(self.state.code)))
            if line.strip() == '!debug':
                self.state.debug = not self.state.debug
            if line.strip() == '!verbose_exc':
                self.state.verbose_exceptions = not self.state.verbose_exceptions
            if line.strip() == '!reset':
                self.state.communicate += ['reset state']
            #if line.strip() == '!cleanup':  #clears the /repl-tmpfiles directory
                #u.clear_repl_tmpfiles()
                #os.makedirs(os.path.dirname(self.state.tmpfile))
            #if line.strip() == '!which':
                #print(self.state.tmpfile)
            if line.strip() == '!help':
                blue('Currently implemented macros listing:')
                print(mk_purple('\n'.join(codegen.macro_argc.keys())))
            print(mk_gray("metacommand registered"))
            return True
        return False

    # the banner is like the '>>> ' in the python repl for example
    def update_banner(self):
        self.state.banner_cwd = os.getcwd()
        prettycwd = pretty_path(self.state.banner_cwd)
        if self.state.mode == 'normal':
            banner_txt = "es:"+prettycwd+" > "
            self.state.banner_uncoloredlen = len(banner_txt)
            self.state.banner = mk_green(banner_txt)
        if self.state.mode == 'speedy':
            banner_txt = "es:"+prettycwd+" $ "
            self.state.banner_uncoloredlen = len(banner_txt)
            self.state.banner = mk_yellow(banner_txt)

    # prompts user for input and returns the line they enter.
    def get_input(self):
        try:
            # MUST reload everything bc this is the first step after taking input
            # so may have been sitting at the input() screen for a while
            # so everything should be reloaded in case of syntax errors
            # Note that this even sucessfully reloads itself... it reloads
            # sys.modules['repl'] across all these files so that when main()
            # calls repl.Repl() next it will use the new one
            # reload all source files (branches out to EVERYTHING including backend and other things not directly imported here)
            line = input(self.state.banner)
            while True:
                failed_mods = u.reload_modules(sys.modules,verbose=self.state.verbose_exceptions)
                if not failed_mods: # empty list is untruthy
                    break
                if 'repl' in failed_mods:
                    blue('error in repl.py, breaking to main.py to recompile') # fuck ya, clever boiiii
                    self.state.communicate += ['drop out of repl to reload from main']
                    return
                line = input(mk_green("[Reload with ENTER]"))
                if self.try_metacommands(line): #eg '!verbose_exc'
                    return
        except KeyboardInterrupt: # ctrl-c lets you swap modes quickly TODO instead have this erase the current like like in the normal python repl
            if self.state.mode == 'speedy':
                self.state.mode = 'normal'
            elif self.state.mode == 'normal':
                self.state.mode = 'speedy'
            self.update_banner()
            print('')
            return
        except EOFError: # exit with ctrl-d
            print('')
            exit(0)
        return line

# takes a line of input and generates a list of lines of final python code using codegen.parse()
    def gen_code(self,line):
        new_code = [] #this is what we'll be adding our code to
        # MULTILINE STATEMENTS
        if line.strip()[-1] == ':': # start of an indent block
            if self.state.mode == 'speedy': print(mk_gray('dropping into normal mode for multiline'))
            lines = [line]
            while True:
                line = input(mk_green('.'*self.state.banner_uncoloredlen))
                if line.strip() == '': break    # ultra simple logic! No need to keep track of dedents/indents
                lines.append(line)
            new_code += [codegen.parse(line,debug=self.state.debug) for line in lines]
        else:
            if self.state.mode == 'speedy':
                line = line.strip()
                toks = line.split(' ')
                if line[:3] == 'cd ':
                    line = '%cd "'+' '.join(toks[1:])+'"'
                else:
                    line = 'sh{'+line+'}'
            # SPEEDY/NORMAL MODE
            new_code.append(codegen.parse(line,debug=self.state.debug))
            #to_undo = 1
        return new_code

    # takes the output of gen_code and runs it
    def run_code(self,new_code):
        # write to tmpfile
        #with open(self.tmpfile,'w') as f:
            #f.write('\n'.join(self.code))
        codestring = '\n'.join(new_code)
        try:
            # Note that this can only take one Interactive line at a time (which may
            # actually be a multiline for loop etc).
            # A version that compiled with 'exec' or 'eval' mode and thus could
            # handle multiple Interactive lines at once was used in an
            # earlier version, look back thru the git repository if you need it. However
            # really you should be able to just divide up your input and call run_code multiple
            # times with the pieces.
            as_ast = ast.parse(codestring,mode='single') # parse into a python ast object
            as_ast = ast.fix_missing_locations(as_ast)
            code = compile(as_ast,'<ast>','single')
            exec(code,self.state.globs,self.state.locs)
            #print(ast.dump(as_ast))
            self.state.code += new_code # keep track of successfully executed code
        except u.VerbatimExc as e:
            print(e)
        except Exception as e:
            # This is where exceptions for the code go.
            # TODO make em look nicer by telling format_exception this is the special case of a repl error thrown by exec() or eval()
            # (for this you may wanna have ast.parse() in a separate try-except to differentiate. It'll catch syntax errors specifically.
            print(u.format_exception(e,['<string>',u.src_path],verbose=self.state.verbose_exceptions))



