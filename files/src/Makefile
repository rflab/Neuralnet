.PHONY:build rebuild clean help prof
CC = g++
#CFLAGS = -std=c++1y -MMD -MP -Wall -Wextra 
CFLAGS = -std=c++1y


ai.out: main.cpp neuralnet.h neuralnet.cpp
	$(CC) $(CFLAGS) main.cpp neuralnet.cpp -o ai.out
	

clean:
	-rm ai.out

build: ai.out

rebuild: clean build ;

help:
	@echo '  build'
	@echo '  rebuild'
	@echo '  clean'


