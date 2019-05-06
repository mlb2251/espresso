#!/usr/local/bin/python3

while True:
    try:
        import util as u
    except Exception as e:
        print("util.py failed to compile so can't prettify")
        print(e)
        input("[Reload with ENTER]")
        continue
    try:
        import main
        main.main()
        break # the end!
    except Exception as e:
        print(u.format_exception(e,u.src_path,verbose=True))
        input(u.mk_g("[Reload with ENTER]"))
        continue


