CC=g++
CFLAGS=-std=c++11 -lsqlite3
OPTARGS=-Wall -Werror 
O_FILES=action.o state.o dictionary.o banjin.o

generator: $(O_FILES) generator.cpp
	$(CC) $(OPTARGS) -o generator generator.cpp $(O_FILES) $(CFLAGS) 

action.o: action.cpp action.h
	$(CC) -std=c++11 $(OPTARGS) -c action.cpp

state.o: state.cpp state.h
	$(CC) -std=c++11 $(OPTARGS) -c state.cpp

dictionary.o: dictionary.cpp dictionary.h
	$(CC) -std=c++11 $(OPTARGS) -c dictionary.cpp

banjin.o: banjin.cpp banjin.h action.h state.h dictionary.h
	$(CC) -std=c++11 $(OPTARGS) -c banjin.cpp

clean:
	rm -f generator *~ *.o
