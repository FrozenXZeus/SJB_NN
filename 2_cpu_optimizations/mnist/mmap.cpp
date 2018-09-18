#include "include/mmap.h"

// Linux libs
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

LinuxMMap::LinuxMMap(const char *pathname) {
    _fd = open(pathname, O_RDONLY);
}

void* LinuxMMap::open_map() {
    if (_fd == -1) {
        return nullptr;
    }

    struct stat file_stat;
    int res = fstat(_fd, &file_stat);
    if (res != 0) {
        return nullptr;
    }
    long filesize = file_stat.st_size;

    void *result = mmap(NULL, (size_t)filesize, PROT_READ, MAP_SHARED, _fd, 0);
    if (result == MAP_FAILED) {
        return nullptr;
    }
    _filesize = filesize;
    _map = result;
    return result;
}

bool LinuxMMap::valid() {
    return _fd != -1;
}

void* LinuxMMap::get_map() const {
    return _map;
}

long LinuxMMap::get_file_size() const {
    return _filesize;
}

LinuxMMap::~LinuxMMap() {
    if (_fd != -1) {
        close(_fd);
    }
    if (_map != nullptr) {
        munmap(_map, _filesize);
    }
}

DumbMMap::DumbMMap(const char *pathname) {
    _fd = open(pathname, O_RDONLY);
}

void* DumbMMap::open_map() {
    if (_fd == -1) {
        return nullptr;
    }

    struct stat file_stat;
    int res = fstat(_fd, &file_stat);
    if (res != 0) {
        return nullptr;
    }
    long filesize = file_stat.st_size;

    unsigned char *result = new unsigned char[filesize];

    res = read(_fd, result, filesize);

    if (res != filesize) {
        delete[] result;
        return nullptr;
    }

    _filesize = filesize;
    _map = result;
    return result;
}

bool DumbMMap::valid() {
    return _fd != -1;
}

void* DumbMMap::get_map() const {
    return _map;
}

long DumbMMap::get_file_size() const {
    return _filesize;
}

DumbMMap::~DumbMMap() {
    if (_fd != -1) {
        close(_fd);
    }
    if (_map != nullptr) {
        delete[] _map;
    }
}