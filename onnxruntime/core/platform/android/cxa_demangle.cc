// #if USE_DUMMY_EXA_DEMANGLE

extern "C" {
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define DEMANGLE_SUCCESS 0
#define DEMANGLE_MEMORY_ALLOCATION_FAILURE -1
#define DEMANGLE_INVALID_MANGLED_NAME -2
#define DEMANGLE_INVALID_ARGUMENTS -3

/**
 * An dummy implementation of __cxa_demangle https://itanium-cxx-abi.github.io/cxx-abi/abi.html#demangler It is
 * referred in https://github.com/llvm/llvm-project/blob/d09d297c5d/libcxxabi/src/cxa_default_handlers.cpp#L53 However
 * it contribute a large percentage of binary size in minimal build. To avoid it, the LLVM user can compile libc++abi.a
 * with LIBCXXABI_NON_DEMANGLING_TERMINATE defined, see https://reviews.llvm.org/D88189 But this make the compilation
 * process extremely complex. Here we provide a dummy __cxa_demangle. It always copy the mangled name to output buffer.
 */
char* __cxa_demangle(const char* mangled_name, char* buf, size_t* n, int* status) {
  if (!status) return NULL;
  if (!mangled_name ||                             // Input must be valid
      (buf && !n) ||                               // If it is non-null, then n must also be nonnull
      (buf && n && ((*n == 0) || (*n > INT_MAX)))  // if buf and n is supplied, *n must be a 'valid' value
  ) {
    *status = DEMANGLE_INVALID_ARGUMENTS;
    return NULL;
  }

  // size is the buffer size, including the null char
  size_t output_size = n ? *n : 0;
  // It is caller's responsibility to ensure mangled_name is null-terminated as it is required in the API doc.
  size_t input_size = strlen(mangled_name) + 1;
  if (input_size > output_size) {
    free(buf);
    buf = NULL;
  }
  if (buf == NULL) {
    output_size = input_size;
    buf = static_cast<char*>(malloc(output_size));
    if (buf == NULL) {
      *status = DEMANGLE_MEMORY_ALLOCATION_FAILURE;
      return NULL;
    }
  }
  strncpy(buf, mangled_name, output_size);
  // Blindly guard the output buffer, this might cause a truncated mangled name being returned without error status,
  // but this should be fine for debugging purpose. We are return the mangle name directly, if the caller is doing
  // crazy stuff with the name, it should both fail with the return mangled name and a truncated mangled name.
  buf[output_size - 1] = '\0';
  if (n) {
    *n = output_size;
  }
  *status = DEMANGLE_SUCCESS;
  return buf;
}
}

#undef DEMANGLE_INVALID_ARGUMENTS
#undef DEMANGLE_INVALID_MANGLED_NAME
#undef DEMANGLE_MEMORY_ALLOCATION_FAILURE
#undef DEMANGLE_SUCCESS

// #endif
