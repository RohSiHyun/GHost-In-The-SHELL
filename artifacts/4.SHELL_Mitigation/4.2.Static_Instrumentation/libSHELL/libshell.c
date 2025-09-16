#include <jemalloc/jemalloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>

#define MAP_HMM (0x1000000000000000)


// Separate Arena for Custom Allocation
unsigned arena0 = 0;
static size_t sz = sizeof(arena0);
static pthread_once_t mutex = PTHREAD_ONCE_INIT;
static pthread_once_t mutex2 = PTHREAD_ONCE_INIT;

// Hooks for Custom Resize Functions
static extent_hooks_t *default_hooks = NULL;
static size_t arena0_msb = 0;


extern char __start_hmm_section[] __attribute__((weak));
extern char __stop_hmm_section[] __attribute__((weak));

// Runtime Access Control
static int __cuda_device_stub__uvm_fd = -1;
typedef struct {
    void *begin;
    void *end;
} shell_table_entry_t;


// Custom allocation hook
static void *custom_alloc_hook(extent_hooks_t *extent_hooks, void *new_addr, size_t size,
                            size_t alignment, bool *zero, bool *commit, unsigned arena_ind) {
    void *p = default_hooks->alloc(default_hooks, new_addr, size, alignment, zero, commit, arena_ind);

    // enforce access control, updating shell_table_entry_t
    shell_table_entry_t st_entry = {
        .begin = p,
        .end = (void *)((char*)p + size)
    };
    ioctl(__cuda_device_stub__uvm_fd, 2050, &st_entry);
    return p;
}

// Arena Initialization
static void init_arena() {
    // initialize the runtime accesscontroller setup
    __cuda_device_stub__uvm_fd = open("/dev/nvidia-uvm", O_RDWR);
    if (__cuda_device_stub__uvm_fd < 0) {
        perror("Failed to open /dev/nvidia-uvm");
        exit(EXIT_FAILURE);
    }
    mallctl("arenas.create", &arena0, &sz, NULL, 0);
    if (arena0 == 0) {
        fprintf(stderr, "Error creating arena 0\n");
        exit(EXIT_FAILURE);
    }
    
    char hook_name[64];
    size_t hooks_sz = sizeof(extent_hooks_t *);
    snprintf(hook_name, sizeof(hook_name), "arena.%u.extent_hooks", arena0);

    default_hooks = (extent_hooks_t *)malloc(sizeof(extent_hooks_t));
    if (default_hooks == NULL) {
        fprintf(stderr, "Error allocating memory for default hooks\n");
        exit(EXIT_FAILURE);
    }

    if (mallctl(hook_name, (void **)&default_hooks, &hooks_sz, NULL, 0) != 0) {
        fprintf(stderr, "Error setting default hooks for arena 0\n");
        free(default_hooks);
        exit(EXIT_FAILURE);
    }

    extent_hooks_t *hooks = (extent_hooks_t *)malloc(sizeof(extent_hooks_t));
    if (hooks == NULL) {
        fprintf(stderr, "Error allocating memory for hooks\n");
        free(default_hooks);
        exit(EXIT_FAILURE);
    }
    *hooks = *default_hooks;
    hooks->alloc = custom_alloc_hook;
    if (mallctl(hook_name, NULL, NULL, &hooks, sizeof(extent_hooks_t *)) != 0) {
        fprintf(stderr, "Error setting custom hooks for arena 0\n");
        free(hooks);
        free(default_hooks);
        exit(EXIT_FAILURE);
    }

    void *p = mallocx(0x10, MALLOCX_ARENA(arena0));
    dallocx(p, MALLOCX_ARENA(arena0));
    
    arena0_msb = (size_t)p & (size_t)(0xffffffff00000000ULL);
    
    shell_table_entry_t st_entry = {
        .begin = (void *)p,
        .end = (void *)(p + 0x400000)
    };
    if(ioctl(__cuda_device_stub__uvm_fd, 2050, &st_entry) < 0) {
        perror("Failed to register section with UVM");
    }
}
static void init_dev() {
    // initialize the runtime accesscontroller setup
    if(__cuda_device_stub__uvm_fd >= 0) {
        return;
    }
    __cuda_device_stub__uvm_fd = open("/dev/nvidia-uvm", O_RDWR);
    if (__cuda_device_stub__uvm_fd < 0) {
        perror("Failed to open /dev/nvidia-uvm");
        exit(EXIT_FAILURE);
    }
    shell_table_entry_t st_entry = {
        .begin = (void *)__start_hmm_section,
        .end = (void *)__stop_hmm_section
    };

    if(ioctl(__cuda_device_stub__uvm_fd, 2050, &st_entry) < 0) {
        perror("Failed to register section with UVM");
    }

}

void *instru_malloc(size_t size) {
    pthread_once(&mutex, init_arena);
    void *ptr = mallocx(size, MALLOCX_ARENA(arena0));

    return ptr;
}

void instru_free(void *ptr) {
    if (((size_t)ptr & (size_t)(0xffffffff00000000ULL)) == arena0_msb) {
        dallocx(ptr, MALLOCX_ARENA(arena0));
    } else {
        free(ptr);
    }
}

void *mmap(void *addr, size_t length, int prot, uint64_t flags, int fd, off_t offset) {
    void *ret = (void*) syscall(SYS_mmap, addr, length, prot, flags, fd, offset);
    
    if(flags & MAP_HMM) {
        pthread_once(&mutex2, init_dev);

        addr = (void*)((size_t)ret & 0xFFFFFFFFFFFF0000); // Align to page size
        length = (length + 0xFFFF) & 0xFFFFFFFFFFFF0000; // Align to page size

        shell_table_entry_t st_entry = {
            .begin = addr,
            .end = (void*)((size_t)addr + length)
        };
    
        if(ret < 0) {
            perror("mmap failed");
            return ret;
        }
        ioctl(__cuda_device_stub__uvm_fd, 2050,  &st_entry);
    } 
    return ret;
}

typedef struct {
    uint64_t base;
    uint64_t fini;
} hmm_mmap_reset_t;

int munmap(void *addr, size_t length) {
    int r = syscall(SYS_munmap, addr, length);
   
    addr = (void*)((size_t)addr & 0xFFFFFFFFFFFF0000); // Align to page size
    length = (length + 0xFFFF) & 0xFFFFFFFFFFFF0000; // Align to page size

    hmm_mmap_reset_t mmap_reset;
    mmap_reset.base = (uint64_t)addr;
    mmap_reset.fini = (uint64_t)addr + length;    

    ioctl(__cuda_device_stub__uvm_fd, 2051, &mmap_reset); // UVM_RESET_HMM ioctl
     
    return r;

}
