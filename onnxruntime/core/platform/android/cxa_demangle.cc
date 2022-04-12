// #if USE_DUMMY_EXA_DEMANGLE

extern "C" {
#include <string.h>
#include <stddef.h>
#include <limits.h>

#define DEMANGLE_SUCCESS 0
#define DEMANGLE_MEMORY_ALLOCATION_FAILURE -1
#define DEMANGLE_INVALID_MANGLED_NAME -2
#define DEMANGLE_INVALID_ARGUMENTS -3

/**
 * An dummy implementation of __cxa_demangle https://itanium-cxx-abi.github.io/cxx-abi/abi.html#demangler It is
 * referred in https://github.com/llvm/llvm-project/blob/d09d297c5d/libcxxabi/src/cxa_default_handlers.cpp#L53 However
 * it contribute a large percentage of binary size in minial build. To avoid it, the LLVM user can compile libc++abi.a
 * with LIBCXXABI_NON_DEMANGLING_TERMINATE defined, see https://reviews.llvm.org/D88189 But this make the compilation
 * process extremely complex. Here we provide a dummy __cxa_demangle. It always copy the mangled name to output buffer.
 * The required allocation behavior in Itanium C++ ABI will result a failure status, a.k.a, not implemented.
 */
char* __cxa_demangle(const char* mangled_name, char* output_buffer, size_t* length, int* status) {
  if (!status) return NULL;
  if (!output_buffer) {
    // In the API doc, the output buffer can be null. But we are allocation free, so we just return a memory allocation
    // failure to indicate it is not the caller's fault.
    *status = DEMANGLE_MEMORY_ALLOCATION_FAILURE;
    return NULL;
  }
  if (!mangled_name || !length || (*length == 0) || (*length > INT_MAX)) {
    *status = DEMANGLE_INVALID_ARGUMENTS;
    return NULL;
  }

  // It is caller's responsibility to ensure mangled_name is null-terminated as it is required in the API doc.
  strncpy(output_buffer, mangled_name, *length);
  // Blindly guard the output buffer, this might cause a truncated mangled name being returned without error status,
  // but this should be fine for debugging purpose. We are return the mangle name directly, if the caller is doing
  // crazy stuff with the name, it should both fail with the return mangled name and a truncated mangled name.
  output_buffer[*length - 1] = '\0';
  *status = DEMANGLE_SUCCESS;
  return output_buffer;
}
}

#undef DEMANGLE_INVALID_ARGUMENTS
#undef DEMANGLE_INVALID_MANGLED_NAME
#undef DEMANGLE_MEMORY_ALLOCATION_FAILURE
#undef DEMANGLE_SUCCESS

// #endif
