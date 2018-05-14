CC = mpicc
CFLAGS = -O2
LDFLAGS = -lm

PROJ = matrix_main.c
OBJS = matrix_main.o

main : $(OBJS)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

obj-clean:
	$(RM) *.o

exec-clean:
	$(RM) $(PROJ)

autosave-clean:
	$(RM) *~

clean:
	$(MAKE) obj-clean
	$(MAKE) exec-clean
	$(MAKE) autosave-clean
