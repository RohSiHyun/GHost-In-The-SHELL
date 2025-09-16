#ifndef LIBSHELL_H
#define LIBSHELL_H

#include <stddef.h>  // for size_t

#ifdef __cplusplus
extern "C" {
#endif

// Arena-backed memory allocation functions
void* instru_malloc(size_t size);
void  instru_free(void* ptr);

#ifdef __cplusplus
}
#endif

#endif // LIBSHELL_H