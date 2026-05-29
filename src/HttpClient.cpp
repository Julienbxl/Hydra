#include "HttpClient.h"

#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/ssl.h>

namespace hydra_http {

namespace {

void ensure_ssl_init() {
    static const bool initialized = []() {
        SSL_library_init();
        SSL_load_error_strings();
        OpenSSL_add_all_algorithms();
        return true;
    }();
    (void)initialized;
}

#ifdef _WIN32
void ensure_winsock_init() {
    static const bool initialized = []() {
        WSADATA wsa_data;
        return WSAStartup(MAKEWORD(2, 2), &wsa_data) == 0;
    }();
    if (!initialized) {
        throw std::runtime_error("WSAStartup failed");
    }
}
#endif

}  // namespace

std::string https_request(const std::string& host,
                          const std::string& request) {
    ensure_ssl_init();
#ifdef _WIN32
    ensure_winsock_init();
#endif

    SSL_CTX* ctx = nullptr;
    BIO* bio = nullptr;
    std::string response;

    try {
        ctx = SSL_CTX_new(TLS_client_method());
        if (!ctx) throw std::runtime_error("SSL context failed");

        bio = BIO_new_ssl_connect(ctx);
        if (!bio) throw std::runtime_error("BIO_new_ssl_connect failed");

        SSL* ssl = nullptr;
        BIO_get_ssl(bio, &ssl);
        if (!ssl) throw std::runtime_error("BIO_get_ssl failed");

        SSL_set_tlsext_host_name(ssl, host.c_str());
        BIO_set_conn_hostname(bio, host.c_str());
        BIO_set_conn_port(bio, "443");

        if (BIO_do_connect(bio) <= 0) throw std::runtime_error("connect failed: " + host);
        if (BIO_do_handshake(bio) <= 0) throw std::runtime_error("TLS handshake failed: " + host);

        if (BIO_write(bio, request.data(), static_cast<int>(request.size())) <= 0) {
            throw std::runtime_error("BIO_write failed");
        }

        char buffer[4096];
        for (;;) {
            const int n = BIO_read(bio, buffer, sizeof(buffer));
            if (n > 0) {
                response.append(buffer, buffer + n);
                continue;
            }
            if (n == 0) break;
            if (!BIO_should_retry(bio)) break;
        }
    } catch (const std::exception&) {
        if (bio) BIO_free_all(bio);
        if (ctx) SSL_CTX_free(ctx);
        throw;
    }

    if (bio) BIO_free_all(bio);
    if (ctx) SSL_CTX_free(ctx);

    const size_t header_end = response.find("\r\n\r\n");
    return (header_end != std::string::npos) ? response.substr(header_end + 4) : std::string();
}

std::string https_get(const std::string& host, const std::string& path) {
    std::string request = "GET " + path + " HTTP/1.0\r\nHost: " + host +
                          "\r\nConnection: close\r\nUser-Agent: Hydra\r\n\r\n";
    return https_request(host, request);
}

std::string https_post(const std::string& host,
                       const std::string& path,
                       const std::string& body,
                       const std::string& content_type) {
    std::string request = "POST " + path + " HTTP/1.0\r\nHost: " + host +
                          "\r\nConnection: close\r\nUser-Agent: Hydra" +
                          "\r\nContent-Type: " + content_type +
                          "\r\nContent-Length: " + std::to_string(body.size()) +
                          "\r\n\r\n" + body;
    return https_request(host, request);
}

}  // namespace hydra_http
