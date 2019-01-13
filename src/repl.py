# REPL
import sys
import os
from importlib import reload
import traceback as tb
import ast

import codegen
import util as u
from util import die,warn,mk_blue,mk_red,mk_yellow,mk_cyan,mk_bold,mk_gray,mk_green,mk_purple,mk_underline,red,blue,green,yellow,purple,pretty_path


# this is the Repl called from main.py
# Repl.next() is the fundamental function that gets input, parses it, and runs it
class Repl:
    def __init__(self,state):
        self.state=state #self.state is a ReplState (defined in main.py, intentionally not here for reload() reasons)

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



