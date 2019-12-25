# Espresso
Espresso is a shell combining the familiarity and speed of Bash with the flexibility and expressiveness of Python. The goal of the shell is to behave just like Bash for everyday usage (no learning curve required!) with the option to drop into a superset of Python at any time. Espresso uses a flexible parser that is able to parser all of Python plus the language additions required by Espresso, which are designed to not conflict with the Python language.

# Bash mode
Espresso can do anything that Bash can do. As a simple example:
```bash
$ mkdir dir
$ cd dir
$ echo hello world > test
$ cat test
hello world
```
But any complex example will work as well. Espresso has a strong guarantee that any existing Bash commands and scripts will work exactly as they would in Bash, which is achieved by executing pure Bash commands in a Bash shell the background. This guarantees an extremely smooth transition from Bash to Espresso – Espresso will never get in your way!

# Python mode
Drop into python mode with `ctrl-c` (normally clears current line / does nothing in Bash). Now you're in a full Python environment as if you'd started up a python interpreter, but with a lot of extra features!

Normal Python code works as you'd expect:
```python
>>> x = "hey" + " there"
>>> x
hey there
>>> def foo(a,b):
...     print("hello")
...     return a+b
...
>>> foo(10,20)
hello
30
```

However, Espresso can do more than a normal Python interpeter. At any time you can used backticks ``` `` ``` to surround a snippet of Bash code, and the resulting Python expression will evaluate to the `stdout` output of the Bash snippet:

```python
  $ echo hello > test
>>> # In Python mode the previous line would be written with backticks: `echo hello > test`
>>> x = `cat test` + " world"
>>> x
hello world
```
Bash code evaluated in backticks will have its usual side effects (creating files, printing text, etc). After the first line of the previous example `ctrl-c` is pressed to switch from Bash to Python mode.

# Space-Separated Call Syntax
One of the features of Bash that makes it so fast is the simplicity of function application. The lack of commas and parentheses make syntax that is much cleaner and quicker to type when doing simple function application. Espresso extends this space-separated call syntax to Python:
```python
>def foo(x,y):
>	print(x+y)
>
>foo 10 (1+2)
13
```
This is primarily meant for very simple expressions, so we don't worry too much about cases like how to write `a(b(c()),d(e))` in this syntax, as such a complicated situation would just look more confusing without the clarification provided by parentheses and commas.

Note that Espresso is smart about recognizing this space separated call syntax. It's only available in the interpreter mode of Espresso (where speed of typing single use code is more important than readability) which means we actually have access to all locals/globals when we're parsing this line and can identify that `callable(foo) == True`, meaning `foo` is a function and so space-separated calls would make sense.

Arguments are expressions separated by spaces, and the parser is smart enough to figure out many edge cases, however for simplicity just put parentheses around anything that you aren't sure about (like `1+2` in the above example). There can be spaces or anything else within parentheses, for example `foo 10 (1 + 2)`.




Bash opts for a linear approach where commands are linked by pipes so one can write `a | b | c` as opposed to `c(b(a))`, where the latter would require backtracking (which can be clunky in terminals) if one had `b(a)` and decided to apply `c` to the result.




For convenience Espresso stores the value the previous line evaluated to, which can be stored in a variable using the `>` symbol at the start of a line. Here we store "hello world" in the variable `foo`:
```python bash
>>> x = `echo hello` + " world"
hello world
>>> >foo
>>> foo + "!"
hello world!
```




You can also swap between Bash and Python modes at any time with ctrl-c, and variables will be shared between the two.


There are many, many more features which will be added to this description at some point :)



