#pragma once

#include <string>

namespace hydra_http {

std::string https_get(const std::string& host, const std::string& path);
std::string https_post(const std::string& host,
                       const std::string& path,
                       const std::string& body,
                       const std::string& content_type = "application/json");

}  // namespace hydra_http
