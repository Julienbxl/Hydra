#pragma once

#include <cstdint>
#include <cstdio>
#include <csignal>
#include <string>

namespace hydra_platform {

bool install_interrupt_handler(volatile sig_atomic_t* flag);

bool write_atomic_file(const std::string& path, const std::string& payload, std::string* error);
bool remove_file_if_exists(const std::string& path, std::string* error);

bool file_seek(FILE* file, uint64_t offset);
int64_t file_tell(FILE* file);

}  // namespace hydra_platform
