#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>


int main(){

  printf("we are good!\n PID = %d\n", getpid());

  fork();
  fork();
  fork();
  printf("Hello, world!\n PID = %d\n", getpid());
    return 0;
}
