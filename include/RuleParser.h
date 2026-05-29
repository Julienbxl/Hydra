#ifndef HYDRA_RULE_PARSER_H
#define HYDRA_RULE_PARSER_H

#include "Rules.cuh"
#include <vector>
#include <string>

class RuleParser {
public:
    static bool parse_rule_file(const std::string& filename, std::vector<GpuRule>& rules, std::vector<std::string>* rule_strings = nullptr);
    static bool parse_rule_string(const std::string& line, GpuRule& rule);

private:
    static uint8_t parse_pos(char c);
    static bool parse_char(const std::string& line, size_t& i, uint8_t& out);
};

#endif // HYDRA_RULE_PARSER_H
