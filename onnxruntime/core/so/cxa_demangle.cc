#if USE_DUMMY_EXA_DEMANGLE

extern "C" {
#include <stddef.h>
#include <limits.h>

/**
 * An dummy implementation of __cxa_demangle https://itanium-cxx-abi.github.io/cxx-abi/abi.html#demangler
 * It is referred in https://github.com/llvm/llvm-project/blob/d09d297c5d/libcxxabi/src/cxa_default_handlers.cpp#L53
 * However it contribute a large percentage of binary size in minial build. To avoid it, the LLVM user can compile
 * libc++abi.a with LIBCXXABI_NON_DEMANGLING_TERMINATE defined, see https://reviews.llvm.org/D88189
 * But this make the compilation process extremely complex. Here we provide a dummy __cxa_demangle. It always copy
 * the mangled name to output buffer. The required allocation behavior in Itanium C++ ABI will result a failure status,
 * a.k.a, not implemented.
 */
char* __cxa_demangle(const char* mangled_name, char* output_buffer, size_t* length, int* status) {
  if (!status) return NULL;
  if (!output_buffer) {
    *status = -1;
    return NULL;
  }
  if (!mangled_name || !length || (*length == 0) || (*length > INT_MAX)) {
    *status = -3;
    return NULL;
  }

  int i = 0;
  int limit = (int)(*length) - 1;
  for (; i < limit; i++) {
    output_buffer[i] = mangled_name[i];
    if (mangled_name[i] == '\0') break;
  }
  *length = i;
  *status = 0;
  return output_buffer;
}
}

#endif
