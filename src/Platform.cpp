#include "Platform.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace hydra_platform {

namespace {

volatile sig_atomic_t* g_interrupt_flag = nullptr;

#ifdef _WIN32
BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT || ctrl_type == CTRL_CLOSE_EVENT) {
        if (g_interrupt_flag) *g_interrupt_flag = 1;
        return TRUE;
    }
    return FALSE;
}
#else
void signal_handler(int) {
    if (g_interrupt_flag) *g_interrupt_flag = 1;
}
#endif

bool flush_file_descriptor(FILE* file, std::string* error) {
    if (fflush(file) != 0) {
        if (error) *error = std::strerror(errno);
        return false;
    }

#ifdef _WIN32
    if (_commit(_fileno(file)) != 0) {
        if (error) *error = std::strerror(errno);
        return false;
    }
#else
    if (fsync(fileno(file)) != 0) {
        if (error) *error = std::strerror(errno);
        return false;
    }
#endif
    return true;
}

}  // namespace

bool install_interrupt_handler(volatile sig_atomic_t* flag) {
    g_interrupt_flag = flag;
#ifdef _WIN32
    return SetConsoleCtrlHandler(console_ctrl_handler, TRUE) == TRUE;
#else
    return std::signal(SIGINT, signal_handler) != SIG_ERR;
#endif
}

bool write_atomic_file(const std::string& path, const std::string& payload, std::string* error) {
    const std::string tmp_path = path + ".tmp";
    FILE* file = std::fopen(tmp_path.c_str(), "wb");
    if (!file) {
        if (error) *error = std::strerror(errno);
        return false;
    }

    const size_t written = std::fwrite(payload.data(), 1, payload.size(), file);
    if (written != payload.size()) {
        if (error) *error = std::strerror(errno);
        std::fclose(file);
        std::remove(tmp_path.c_str());
        return false;
    }

    if (!flush_file_descriptor(file, error)) {
        std::fclose(file);
        std::remove(tmp_path.c_str());
        return false;
    }

    if (std::fclose(file) != 0) {
        if (error) *error = std::strerror(errno);
        std::remove(tmp_path.c_str());
        return false;
    }

#ifdef _WIN32
    std::remove(path.c_str());
#endif

    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        if (error) *error = std::strerror(errno);
        std::remove(tmp_path.c_str());
        return false;
    }
    return true;
}

bool remove_file_if_exists(const std::string& path, std::string* error) {
    if (std::remove(path.c_str()) == 0) return true;
    if (errno == ENOENT) return true;
    if (error) *error = std::strerror(errno);
    return false;
}

bool file_seek(FILE* file, uint64_t offset) {
#ifdef _WIN32
    return _fseeki64(file, static_cast<__int64>(offset), SEEK_SET) == 0;
#else
    return fseeko(file, static_cast<off_t>(offset), SEEK_SET) == 0;
#endif
}

int64_t file_tell(FILE* file) {
#ifdef _WIN32
    return static_cast<int64_t>(_ftelli64(file));
#else
    return static_cast<int64_t>(ftello(file));
#endif
}

}  // namespace hydra_platform
