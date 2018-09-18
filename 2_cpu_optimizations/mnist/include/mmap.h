#pragma once

#ifndef SJB_NN_MMAP_H
#define SJB_NN_MMAP_H

class LinuxMMap {
public:
    LinuxMMap(const char *pathname);
    void *open_map();

    bool valid();
    void *get_map() const;
    long get_file_size() const;

    virtual ~LinuxMMap();

private:
    long _filesize = -1;
    void *_map = nullptr;
    int _fd = -1;
};

// I wonder if mmaping is screwing with things
class DumbMMap {
public:
    DumbMMap(const char *pathname);
    void *open_map();

    bool valid();
    void *get_map() const;
    long get_file_size() const;

    virtual ~DumbMMap();

private:
    long _filesize = -1;
    unsigned char *_map = nullptr;
    int _fd = -1;
};

#endif //SJB_NN_MMAP_H
