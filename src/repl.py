# REPL
import sys
import os
from importlib import reload
import traceback as tb
import ast
import main # just so it gets reloaded by reload_modes, someone has to import it!

import codegen
import util as u

###############################
###############################
# TODO solution to many issues:
# enable CustomCompleterDisplayHook
# make ' ' '\t' no longer delims so that 'text' is the full line
# modify CustomCompleterDisplayHook hook to only print the part of the match at the end of the line
# This enables us to do repls on the entire contents of the line! And '\ ' will not print funny or
# anything like that anymore.
# Another cool idea: have an '!' on the end of a line autocomplete to having 'temp=' at the start of the line
# for rapid assignment stuff
###############################
###############################
# given all this cool stuff you can probably also totally keylog and update the display manually with
# custom highlighting. Note that writing your own custom highlighter for python syntax in python regex
# w/ escape code coloring wd prob not be too hard.
# *** note that if you do this you must make it OPTIONAL bc not using the native display-when-you-type
# will be less lightweight and thus should be optional

# oh also you should really do A) let you launch stuff like 'vim' and B) let you drop into vim
# to edit multiline statements / past multiline statements
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


def set_tabcomplete(setting):
    if setting:
        readline.parse_and_bind("tab: complete")
    else:
        readline.parse_and_bind('tab: self-insert')

set_tabcomplete(True)
readline.set_completer(completer)





# this is the Repl called from main.py
# Repl.next() is the fundamental function that gets input, parses it, and runs it
class Repl:
    def __init__(self,repl=None):
        self.globs= dict() # globals dict used by exec(). TODO Should actually be initialized to whatever pythons globals() is initialized to
        self.code=[] # a list containing all the generated python code so far. Each successful line the REPL runs is added to this
        self.mode= 'speedy'    # 'normal' or 'speedy'
        self.banner='[banner uninitialized]>>>' #the banner is that thing that looks like '>>> ' in the python interpreter for example
        self.banner_uncoloredlen = len(self.banner) # the length of the banner, ignoring the meta chars used to color it
        self.banner_cwd= '' # the length of the banner, ignoring the meta chars used to color it
        self.debug=False  #if this is True then parser output is generated. You can toggle it with '!debug'
        self.communicate=[] # for passing messages between main.py and repl.py, for things like hard resets and stuff
        self.verbose_exceptions=False #if this is true then full raw exceptions are printed in addition to the formatted ones
        self._magic = "this is proof that I'm an espresso object"

        if repl is not None:
            self._load(repl)
        #readline.set_completion_display_matches_hook(CustomCompleterDisplayHook(self)) # TODO uncomment

    def _load(self, repl):
        if hasattr(repl,'_magic') and repl._magic == self._magic: # necessary due to module reloading
            d = repl.__dict__
        elif isinstance(repl,dict):
            d = repl
        else:
            print("[err] unable to load from {} as it is not a dict or Repl object".format(repl))
            breakpoint()
            return
        for key in self.__dict__.keys(): # only import keys that are in the newest version
            if key in d.keys():
                setattr(self,key,d[key])

    # updates the state based on communications sent through the list ReplState.communication

    def next(self,line):
        if line is None: return
        if len(line.strip()) == 0: return

        # deal with '!' commands
        if self.try_metacommands(line):
            return

        # generate a list of lines of valid python code
        new_code = self.gen_code(line)
        # run the code, updating the self.globals/locals
        self.run_code(new_code)

    # handles ! commands and returns 'False' if none were detected
    # (indicating the line should be parsed normally)
    def try_metacommands(self,line):
        if len(line) == 0: return False
        if line.strip() == '%':
            if self.mode != 'speedy':
                self.mode = 'speedy'
            else:
                self.mode = 'normal'
            self.update_banner()
            u.gray("metacommand registered")
            return True
        # handle metacommands
        if line[0] == '!':
            if line.strip() == '!print':
                u.y('\n'.join(self.code))
            if line.strip() == '!debug':
                self.debug = not self.debug
            if line.strip() == '!verbose_exc':
                self.verbose_exceptions = not self.verbose_exceptions
            if line.strip() == '!reset':
                self.__init__(None)
            #if line.strip() == '!cleanup':  #clears the /repl-tmpfiles directory
                #u.clear_repl_tmpfiles()
                #os.makedirs(os.path.dirname(self.tmpfile))
            #if line.strip() == '!which':
                #print(self.tmpfile)
            if line.strip() == '!help':
                u.b('Currently implemented macros listing:')
                u.p('\n'.join(codegen.macro_argc.keys()))
            u.gray("metacommand registered")
            return True
        return False

    # the banner is like the '>>> ' in the python repl for example
    def update_banner(self):
        self.banner_cwd = os.getcwd()
        prettycwd = u.pretty_path(self.banner_cwd)
        if self.mode == 'normal':
            banner_txt = "es:"+prettycwd+" > "
            self.banner_uncoloredlen = len(banner_txt)
            self.banner = u.mk_g(banner_txt)
        if self.mode == 'speedy':
            banner_txt = "es:"+prettycwd+" $ "
            self.banner_uncoloredlen = len(banner_txt)
            self.banner = u.mk_y(banner_txt)

    # prompts user for input and returns the line they enter.
    def get_input(self):
        if self.banner_cwd != os.getcwd(): self.update_banner()
        try:
            set_tabcomplete(True) # just put this here in case it gets set to False by multiline then somehow the interpreter drops out intoj
            line = input(self.banner)
        except KeyboardInterrupt: # ctrl-c lets you swap modes quickly TODO instead have this erase the current like like in the normal python repl
            if self.mode == 'speedy':
                self.mode = 'normal'
            elif self.mode == 'normal':
                self.mode = 'speedy'
            self.update_banner()
            print('')
            return
        except EOFError: # exit with ctrl-d
            print('\n[wrote history]')
            import readline
            readline.write_history_file(u.histfile)
            sys.exit(0)
        return line

# takes a line of input and generates a list of lines of final python code using codegen.parse()
    def gen_code(self,line):
        new_code = [] #this is what we'll be adding our code to
        # MULTILINE STATEMENTS
        if line.strip()[-1] == ':': # start of an indent block
            set_tabcomplete(False)
            if self.mode == 'speedy': u.gray('dropping into normal mode for multiline')
            lines = [line]
            while True:
                line = input(u.mk_g(' '*(self.banner_uncoloredlen-1)+'|'))
                if line.strip() == '': break    # ultra simple logic! No need to keep track of dedents/indents
                lines.append(line)
            new_code += [codegen.parse(line,self.globs,debug=self.debug) for line in lines]
        else:
            if self.mode == 'speedy':
                line = ':'+line.strip()
            # SPEEDY/NORMAL MODE
            new_code.append(codegen.parse(line,self.globs,debug=self.debug))
            #to_undo = 1
        return new_code

    # takes the output of gen_code and runs it
    def run_code(self,new_code):
        # write to tmpfile
        #with open(self.tmpfile,'w') as f:
            #f.write('\n'.join(self.code))
        codestring = '\n'.join(new_code)
        u.b(codestring)
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
            exec(code,self.globs) #passing locals in causes it to only update locals and not globals, when really we just want globals to be updated
            #print(ast.dump(as_ast))
            self.code += new_code # keep track of successfully executed code
        except u.VerbatimExc as e:
            print(e)
        except Exception as e:
            # This is where exceptions for the code go.
            # TODO make em look nicer by telling format_exception this is the special case of a repl error thrown by exec() or eval()
            # (for this you may wanna have ast.parse() in a separate try-except to differentiate. It'll catch syntax errors specifically.
            print(u.format_exception(e,['<string>',u.src_path],verbose=self.verbose_exceptions))



