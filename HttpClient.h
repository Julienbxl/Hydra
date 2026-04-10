#pragma once

#include <string>

namespace hydra_http {

std::string https_get(const std::string& host, const std::string& path);

}  // namespace hydra_http
