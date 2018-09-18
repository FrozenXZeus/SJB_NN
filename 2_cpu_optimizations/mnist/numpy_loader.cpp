#include "include/numpy_loader.h"

#include <stddef.h>

// copied from https://docs.scipy.org/doc/numpy/neps/npy-format.html
const int MAGIC_STRING_SIZE = 6;
const char MAGIC_STRING[] = "\x93NUMPY";
// Minimum header size
int MIN_HEADER_SIZE = 10;

#pragma pack(push, 1)
typedef struct NumpyHeader {
    int8_t magic_string[MAGIC_STRING_SIZE];
    uint8_t major_version;
    uint8_t minor_version;
    uint16_t header_data_len;
} NumpyHeader;
#pragma pack(pop)

bool verify_numpy_header(void *file_map,
        int64_t filesize,
        long min_data_size,
        long *data_offset)
{
    if (filesize < MIN_HEADER_SIZE) {
        return false;
    }

    NumpyHeader *header = (NumpyHeader *) file_map;
    for (int i = 0; i < MAGIC_STRING_SIZE; i++) {
        if (header->magic_string[i] != MAGIC_STRING[i]) {
            return false;
        }
    }
    uint16_t header_len = header->header_data_len;

    if (filesize < (MIN_HEADER_SIZE + header_len + min_data_size)) {
        return false;
    }

    *data_offset = MIN_HEADER_SIZE + (int)header_len;

    return true;
}


