# REPL
import sys
import os
from importlib import reload
import traceback as tb

import ast

sys.path.append(os.environ['HOME']+'/espresso/src')
sys.path.append(os.getcwd())
import codegen
import util as u
from util import die,warn,mk_blue,mk_red,mk_yellow,mk_cyan,mk_bold,mk_gray,mk_green,mk_purple,mk_underline,red,blue,green,yellow,purple,pretty_path


class Repl:
    def __init__(self,state):
        self.state=state

    def get_state(self):
        return self.state

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
            if line.strip() == '!cleanup':  #clears the /repl-tmpfiles directory
                u.clear_repl_tmpfiles()
                os.makedirs(os.path.dirname(self.state.tmpfile))
            if line.strip() == '!which':
                print(self.state.tmpfile)
            if line.strip() == '!help':
                blue('Currently implemented macros listing:')
                print(mk_purple('\n'.join(codegen.macro_argc.keys())))
            print(mk_gray("metacommand registered"))
            return True
        return False

    def next(self):
        line = self.get_input()
        if line is None: return
        if len(line.strip()) == 0: return
        # for convenience, 'ls' or 'cd [anything]' will change the mode to speedy
        if line.strip() == 'ls' or line.strip()[:3] == 'cd ':
            self.state.mode = 'speedy'
            self.update_banner()

        # deal with ! commands
        if self.try_metacommands(line):
            return

        new_code = self.gen_code(line)
        self.run_code(new_code)

    def update_banner(self):
        prettycwd = pretty_path(os.getcwd())
        if self.state.mode == 'normal':
            banner_txt = "es:"+prettycwd+" $ "
            self.state.banner_uncoloredlen = len(banner_txt)
            self.state.banner = mk_green(banner_txt)
        if self.state.mode == 'speedy':
            banner_txt = "es:"+prettycwd+" $ %"
            self.state.banner_uncoloredlen = len(banner_txt)
            self.state.banner = mk_purple(banner_txt)

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
                    blue('error in repl.py, breaking to main.py to recompile')
                    self.state.communicate += ['drop out of repl to reload from main']
                    return
                line = input(mk_green("[Reload with ENTER]"))
                if self.try_metacommands(line): #eg '!verbose_exc'
                    return
        except KeyboardInterrupt: # ctrl-c lets you swap modes quickly
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

    def gen_code(self,line):
        new_code = []
        # update codeblock
        #self.code.append("backend.enablePrint()")
        if line.strip()[-1] == ':': # start of an indent block
            if self.state.mode == 'speedy': print(mk_gray('dropping into normal mode for multiline'))
            lines = [line]
            while True:
                line = input(mk_green('.'*self.state.banner_uncoloredlen))
                if line.strip() == '': break    # ultra simple logic! No need to keep track of dedents/indents
                lines.append(line)
            new_code += [codegen.parse(line,debug=self.state.debug) for line in lines]
            #to_undo = len(lines)
        else:
            if self.state.mode == 'speedy': #prepend '%' to every line
                line = line.strip() #strips line + prepends '%'
                line = '%' + line
                toks = line.split(' ')
                # deal with special case transformations
                if toks[0][1:] not in codegen.macro_argc:
                    line = 'sh{'+line[1:]+'}' # the 1: just kills '%'
                    warn('macro {} not recognized. Trying sh:\n{}'.format(toks[0][1:],line))
                elif toks[0] in ['%cd','%cat']: # speedy cd autoquotes the $* it's given
                    line = toks[0]+' "'+' '.join(toks[1:])+'"'

                # finally, print the result
                #line = 'backend.b_p_ignoreNone('+line+')'


            new_code.append(codegen.parse(line,debug=self.state.debug))
            #to_undo = 1
        return new_code

    def run_code(self,new_code):
        # write to tmpfile
        #with open(self.tmpfile,'w') as f:
            #f.write('\n'.join(self.code))
        codestring = '\n'.join(new_code)
        # execute tmpfile
        try:
            as_ast = ast.parse(codestring)
            if len(as_ast.body) ==1 and isinstance(as_ast.body[0],ast.Expr):
                #code = compile(as_ast,'<ast>','eval')
                code = compile(codestring,'<string>','eval') #should be able to compile faster from as_ast but gave some error when i tried -- maybe worth looking into more
                res = eval(code,self.state.globs,self.state.locs)
                if res is not None:
                    print(res) # TODO do special formatted print for speedy mode
            else:
                #code = compile(as_ast,'<ast>','exec')
                code = compile(codestring,'<string>','exec')
                exec(code,self.state.globs,self.state.locs)
            self.state.code += new_code
        except Exception as e:
            print(u.format_exception(e,['<string>',u.src_path],verbose=self.state.verbose_exceptions))



