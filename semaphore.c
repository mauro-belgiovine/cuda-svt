
/*
  - based on http://beej.us/guide/bgipc/output/html/multipage/semaphores.html
  - reason this code is a little complicated: "If two processes are trying to create, initialize, and use a semaphore at the same time, a race condition might develop." <- this code solves this
*/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>


#define MAX_RETRIES 10

union semun {
    int val;
    struct semid_ds *buf;
    ushort *array;
};


/*
** initsem() -- more-than-inspired by W. Richard Stevens' UNIX Network
** Programming 2nd edition, volume 2, lockvsem.c, page 295.
*/
int initsem(key_t key, int nsems)  /* key from ftok() */
{
    int i;
    union semun arg;
    struct semid_ds buf;
    struct sembuf sb;
    int semid;

    semid = semget(key, nsems, IPC_CREAT | IPC_EXCL | 0666);

    if (semid >= 0) { /* we got it first */
	sb.sem_op = 1; sb.sem_flg = 0;
	arg.val = 1;

	//printf("press return\n"); getchar();

	for(sb.sem_num = 0; sb.sem_num < nsems; sb.sem_num++) { 
	    /* do a semop() to "free" the semaphores. */
	    /* this sets the sem_otime field, as needed below. */
	    if (semop(semid, &sb, 1) == -1) {
		int e = errno;
		semctl(semid, 0, IPC_RMID); /* clean up */
		errno = e;
		return -1; /* error, check errno */
	    }
	}

    } else if (errno == EEXIST) { /* someone else got it first */
	int ready = 0;

	semid = semget(key, nsems, 0); /* get the id */
	if (semid < 0) return semid; /* error, check errno */

	/* wait for other process to initialize the semaphore: */
	arg.buf = &buf;
	for(i = 0; i < MAX_RETRIES && !ready; i++) {
	    semctl(semid, nsems-1, IPC_STAT, arg);
	    if (arg.buf->sem_otime != 0) {
		ready = 1;
	    } else {
		sleep(1);
	    }
	}
	if (!ready) {
	    errno = ETIME;
	    return -1;
	}
    } else {
	return semid; /* error, check errno */
    }

    return semid;
}


void lock(int semid) {

    struct sembuf sb;
    
    sb.sem_num = 0;
    sb.sem_op = -1;  /* set to allocate resource */
    sb.sem_flg = SEM_UNDO;

    if (semop(semid, &sb, 1) == -1) {
	perror("semop: lock");
	exit(1);
    }

    printf("Locked.\n");

}

void unlock(int semid) {

    struct sembuf sb;

    sb.sem_num = 0;
    sb.sem_op = 1;  /* free resources */
    sb.sem_flg = SEM_UNDO;

    if (semop(semid, &sb, 1) == -1) {
	perror("semop: unlock");
	exit(1);
    }

    printf("Unlocked.\n");

}
