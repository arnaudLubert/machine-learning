##
## EPITECH PROJECT, 2018
## Makefile
## File description:
## Build the program.
##

EXEC_NAME = brain
#COMPILER = x86_64-w64-mingw32-gcc # i686-w64-mingw32-gcc
COMPILER = gcc
C_FLAGS = -W -Wall

all: build

build:
	cd src ; $(COMPILER) *.c -c $(C_FLAGS)
	$(COMPILER) src/*.o -o $(EXEC_NAME)

clean:
	find . -name "*~" -delete
	find . -name "#*#" -delete
	rm -f build

fclean: clean
	cd src ; rm -f *.o
	rm -f $(EXEC_NAME)

re: clean src/*.o
	$(COMPILER) src/*.o -o $(EXEC_NAME) $(C_FLAGS)

valgrind: clean
	$(COMPILER) src/*.c -o $(EXEC_NAME) $(C_FLAGS) -g3

%.o: %.c
	$(COMPILER) -c $^ $(CFLAGS)
	mv -f *.o src
