#include "RuleParser.h"
#include <fstream>
#include <iostream>
#include <sstream>

namespace {
int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}
}

uint8_t RuleParser::parse_pos(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'Z') return 10 + (c - 'A');
    if (c >= 'a' && c <= 'z') return 10 + (c - 'a');
    return 0; // Default or error
}

bool RuleParser::parse_char(const std::string& line, size_t& i, uint8_t& out) {
    if (i >= line.size()) return false;
    if (line[i] == '\\' && i + 1 < line.size()) {
        char next = line[i + 1];
        if (next == 'x' && i + 3 < line.size()) {
            const int h1 = hex_value(line[i + 2]);
            const int h2 = hex_value(line[i + 3]);
            if (h1 < 0 || h2 < 0) return false;
            out = (uint8_t)((h1 << 4) | h2);
            i += 3;
            return true;
        } else if (next == 't') { i++; out = '\t'; return true; }
        else if (next == 'r') { i++; out = '\r'; return true; }
        else if (next == 'n') { i++; out = '\n'; return true; }
        else if (next == '0') { i++; out = '\0'; return true; }
        else {
            i++;
            out = (uint8_t)next;
            return true;
        }
    }
    out = (uint8_t)line[i];
    return true;
}

bool RuleParser::parse_rule_string(const std::string& line, GpuRule& rule) {
    rule.num_cmds = 0;
    for (size_t i = 0; i < line.size(); ++i) {
        if (rule.num_cmds >= MAX_RULE_CMDS) {
            std::cerr << "Warning: Rule exceeds max commands (" << MAX_RULE_CMDS << "): " << line << "\n";
            return false;
        }

        char c = line[i];
        if (c == ' ' || c == '\t') continue; // Skip whitespace
        if (c == '#') break; // Comment

        RuleCmd cmd = {RULE_NOP, 0, 0};
        bool valid = true;

        switch (c) {
            case ':': cmd.op = RULE_NOP; break;
            case 'l': cmd.op = RULE_LOWER; break;
            case 'u': cmd.op = RULE_UPPER; break;
            case 'c': cmd.op = RULE_CAPITALIZE; break;
            case 'r': cmd.op = RULE_REVERSE; break;
            case 'd': cmd.op = RULE_DUPLICATE; break;
            case '[': cmd.op = RULE_DELETE_FIRST; break;
            case ']': cmd.op = RULE_DELETE_LAST; break;
            case 'T': // Toggle at N
                if (i + 1 < line.size()) {
                    cmd.op = RULE_TOGGLE_AT;
                    cmd.p1 = parse_pos(line[++i]);
                } else valid = false;
                break;
            case 't': // Toggle all
                cmd.op = RULE_TOGGLE_ALL; break;
            case '$': 
                if (i + 1 < line.size()) {
                    cmd.op = RULE_APPEND;
                    i++;
                    valid = parse_char(line, i, cmd.p1);
                } else valid = false;
                break;
            case '^':
                if (i + 1 < line.size()) {
                    cmd.op = RULE_PREPEND;
                    i++;
                    valid = parse_char(line, i, cmd.p1);
                } else valid = false;
                break;
            case 'D':
                if (i + 1 < line.size()) {
                    cmd.op = RULE_DELETE_AT;
                    cmd.p1 = parse_pos(line[++i]);
                } else valid = false;
                break;
            case '+':
                if (i + 1 < line.size()) {
                    cmd.op = RULE_INC_AT;
                    cmd.p1 = parse_pos(line[++i]);
                } else valid = false;
                break;
            case '-':
                if (i + 1 < line.size()) {
                    cmd.op = RULE_DEC_AT;
                    cmd.p1 = parse_pos(line[++i]);
                } else valid = false;
                break;
            case '.':
                if (i + 1 < line.size()) {
                    cmd.op = RULE_REPLACE_N_NEXT;
                    cmd.p1 = parse_pos(line[++i]);
                } else valid = false;
                break;
            case ',':
                if (i + 1 < line.size()) {
                    cmd.op = RULE_REPLACE_N_PREV;
                    cmd.p1 = parse_pos(line[++i]);
                } else valid = false;
                break;
            case '*':
                if (i + 2 < line.size()) {
                    cmd.op = RULE_SWAP;
                    cmd.p1 = parse_pos(line[++i]);
                    cmd.p2 = parse_pos(line[++i]);
                } else valid = false;
                break;
            case 'L': 
                if (i + 1 < line.size()) {
                    cmd.op = RULE_BIT_SHIFT_LEFT;
                    cmd.p1 = parse_pos(line[++i]);
                } else valid = false;
                break;
            case 'R': 
                if (i + 1 < line.size()) {
                    cmd.op = RULE_BIT_SHIFT_RIGHT;
                    cmd.p1 = parse_pos(line[++i]);
                } else valid = false;
                break;
            case '{': cmd.op = RULE_SHIFT_LEFT; break;
            case '}': cmd.op = RULE_SHIFT_RIGHT; break;
            case 'k': cmd.op = RULE_SWAP_FRONT; break;
            case 'K': cmd.op = RULE_SWAP_BACK; break;
            case 'q': cmd.op = RULE_DUP_ALL; break;
            case 's':
                if (i + 2 < line.size()) {
                    cmd.op = RULE_REPLACE;
                    i++; valid = parse_char(line, i, cmd.p1);
                    if (valid) { i++; valid = parse_char(line, i, cmd.p2); }
                } else valid = false;
                break;
            case 'E': cmd.op = RULE_TITLE_CASE; break;
            case 'f': cmd.op = RULE_REFLECT; break;
            case 'o':
                if (i + 2 < line.size()) {
                    cmd.op = RULE_OVERWRITE;
                    cmd.p1 = parse_pos(line[++i]);
                    i++; valid = parse_char(line, i, cmd.p2);
                } else valid = false;
                break;
            case 'i':
                if (i + 2 < line.size()) {
                    cmd.op = RULE_INSERT;
                    cmd.p1 = parse_pos(line[++i]);
                    i++; valid = parse_char(line, i, cmd.p2);
                } else valid = false;
                break;
            case 'O':
                if (i + 2 < line.size()) {
                    cmd.op = RULE_OMIT;
                    cmd.p1 = parse_pos(line[++i]);
                    cmd.p2 = parse_pos(line[++i]);
                } else valid = false;
                break;
            case 'I':
                if (i + 2 < line.size()) {
                    cmd.op = RULE_INSERT_MISSING;
                    cmd.p1 = parse_pos(line[++i]);
                    i++; valid = parse_char(line, i, cmd.p2);
                } else valid = false;
                break;
            case 'C': cmd.op = RULE_INVERT_CASE; break;
            case 'p':
                if (i + 1 < line.size()) {
                    cmd.op = RULE_APPEND_DUP;
                    cmd.p1 = parse_pos(line[++i]);
                } else valid = false;
                break;
            case '\'':
                if (i + 1 < line.size()) {
                    cmd.op = RULE_TRUNCATE;
                    cmd.p1 = parse_pos(line[++i]);
                } else valid = false;
                break;
            case 'x':
                if (i + 2 < line.size()) {
                    cmd.op = RULE_EXTRACT;
                    cmd.p1 = parse_pos(line[++i]);
                    cmd.p2 = parse_pos(line[++i]);
                } else valid = false;
                break;
            case 'y':
                if (i + 1 < line.size()) {
                    cmd.op = RULE_DUP_FIRST;
                    cmd.p1 = parse_pos(line[++i]);
                } else valid = false;
                break;
            case 'Y':
                if (i + 1 < line.size()) {
                    cmd.op = RULE_DUP_LAST;
                    cmd.p1 = parse_pos(line[++i]);
                } else valid = false;
                break;
            case '@':
                if (i + 1 < line.size()) {
                    cmd.op = RULE_PURGE_CHAR;
                    i++; valid = parse_char(line, i, cmd.p1);
                } else valid = false;
                break;
            case 'z':
                if (i + 1 < line.size()) {
                    cmd.op = RULE_DUP_FIRST_N_TIMES;
                    cmd.p1 = parse_pos(line[++i]);
                } else valid = false;
                break;
            case 'Z':
                if (i + 1 < line.size()) {
                    cmd.op = RULE_DUP_LAST_N_TIMES;
                    cmd.p1 = parse_pos(line[++i]);
                } else valid = false;
                break;
            case '<':
                if (i + 1 < line.size()) {
                    cmd.op = RULE_REJECT_LESS;
                    cmd.p1 = parse_pos(line[++i]);
                } else valid = false;
                break;
            case '>':
                if (i + 1 < line.size()) {
                    cmd.op = RULE_REJECT_GREATER;
                    cmd.p1 = parse_pos(line[++i]);
                } else valid = false;
                break;
            case '_':
                if (i + 1 < line.size()) {
                    cmd.op = RULE_REJECT_EQUAL;
                    cmd.p1 = parse_pos(line[++i]);
                } else valid = false;
                break;
            case '!':
                if (i + 1 < line.size()) {
                    cmd.op = RULE_REJECT_CONTAINS;
                    i++; valid = parse_char(line, i, cmd.p1);
                } else valid = false;
                break;
            case '/':
                if (i + 1 < line.size()) {
                    cmd.op = RULE_REJECT_NOT_CONTAINS;
                    i++; valid = parse_char(line, i, cmd.p1);
                } else valid = false;
                break;
            case '(':
                if (i + 1 < line.size()) {
                    cmd.op = RULE_REJECT_STARTS;
                    i++; valid = parse_char(line, i, cmd.p1);
                } else valid = false;
                break;
            case ')':
                if (i + 1 < line.size()) {
                    cmd.op = RULE_REJECT_ENDS;
                    i++; valid = parse_char(line, i, cmd.p1);
                } else valid = false;
                break;
            case '=':
                if (i + 2 < line.size()) {
                    cmd.op = RULE_REJECT_EQUAL_AT;
                    cmd.p1 = parse_pos(line[++i]);
                    i++; valid = parse_char(line, i, cmd.p2);
                } else valid = false;
                break;
            default:
                // Unsupported Hashcat rule op
                std::cerr << "Warning: Unsupported rule op '" << c << "' in: " << line << "\n";
                return false;
        }

        if (!valid) {
            std::cerr << "Warning: Malformed rule command in: " << line << "\n";
            return false;
        }

        // Only add non-NOP commands
        if (cmd.op != RULE_NOP) {
            rule.cmds[rule.num_cmds++] = cmd;
        }
    }
    
    // A rule might be just ':' which compiles to 0 cmds, which is perfectly valid (does nothing).
    return true; 
}

bool RuleParser::parse_rule_file(const std::string& filename, std::vector<GpuRule>& rules, std::vector<std::string>* rule_strings) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open rule file " << filename << "\n";
        return false;
    }

    std::string line;
    int line_num = 0;
    int skipped = 0;
    while (std::getline(file, line)) {
        line_num++;
        // Trim trailing whitespace/carriage return
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        
        // Skip empty or comment-only lines
        bool empty = true;
        for (char c : line) {
            if (c == '#') break;
            if (c != ' ' && c != '\t') { empty = false; break; }
        }
        if (empty) continue;

        GpuRule rule;
        if (parse_rule_string(line, rule)) {
            rules.push_back(rule);
            if (rule_strings) {
                rule_strings->push_back(line);
            }
        } else {
            skipped++;
        }
    }

    if (rules.empty()) {
        std::cerr << "Warning: No valid rules found in " << filename << " (file may contain only comments)\n";
        return true;
    }
    std::cout << "Loaded " << rules.size() << " rules from " << filename;
    if (skipped > 0) std::cout << " (skipped " << skipped << " unsupported/malformed rules)";
    std::cout << "\n";
    return true;
}
