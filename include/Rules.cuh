#ifndef HYDRA_RULES_CUH
#define HYDRA_RULES_CUH

#include <stdint.h>

#define MAX_RULE_CMDS 30

enum RuleOp : uint8_t {
    RULE_NOP = 0,
    RULE_LOWER,       // l
    RULE_UPPER,       // u
    RULE_CAPITALIZE,  // c
    RULE_TITLE_CASE,  // E
    RULE_REVERSE,     // r
    RULE_REFLECT,     // f
    RULE_DUPLICATE,   // d
    RULE_APPEND,      // $x
    RULE_PREPEND,     // ^x
    RULE_DELETE_FIRST,// [
    RULE_DELETE_LAST, // ]
    RULE_DELETE_AT,   // Dx
    RULE_REPLACE,     // sxy
    RULE_TOGGLE_ALL,  // t
    RULE_TOGGLE_AT,   // TN
    RULE_OVERWRITE,   // oNx
    RULE_INSERT,      // iNx
    RULE_INSERT_MISSING,    // INx
    RULE_INC_AT,            // +N
    RULE_DEC_AT,            // -N
    RULE_REPLACE_N_NEXT,    // .N
    RULE_REPLACE_N_PREV,    // ,N
    RULE_SWAP,              // *NM
    RULE_SHIFT_LEFT,        // {
    RULE_SHIFT_RIGHT,       // }
    RULE_BIT_SHIFT_LEFT,    // L
    RULE_BIT_SHIFT_RIGHT,   // R
    RULE_SWAP_FRONT,        // k
    RULE_SWAP_BACK,         // K
    RULE_INVERT_CASE,       // C
    RULE_APPEND_DUP,        // p
    RULE_TRUNCATE,          // '
    RULE_EXTRACT,           // x
    RULE_DUP_FIRST,         // y
    RULE_DUP_LAST,          // Y
    RULE_PURGE_CHAR,        // @x
    RULE_DUP_FIRST_N_TIMES, // zN
    RULE_DUP_LAST_N_TIMES,  // ZN
    RULE_DUP_ALL,           // q
    RULE_OMIT,              // ONM
    RULE_REJECT_LESS,       // <N
    RULE_REJECT_GREATER,    // >N
    RULE_REJECT_EQUAL,      // _N
    RULE_REJECT_CONTAINS,   // !X
    RULE_REJECT_NOT_CONTAINS, // /X
    RULE_REJECT_STARTS,     // (X
    RULE_REJECT_ENDS,       // )X
    RULE_REJECT_EQUAL_AT    // =NX
};

struct RuleCmd {
    RuleOp op;
    uint8_t p1;
    uint8_t p2;
};

struct GpuRule {
    RuleCmd cmds[MAX_RULE_CMDS];
    uint8_t num_cmds;
};

#ifdef __CUDACC__
// Applies a sequence of rules to an input string. 
// Maximum length is assumed to be bounded by Hydra's MAX_PASS_LEN (e.g. 96 or 128)
// Returns false if the rule is rejected (e.g. by <, >, !, / rules).
__host__ __device__ inline bool apply_rule(const uint8_t* in, uint32_t in_len, uint8_t* out, uint32_t& out_len, const GpuRule* rule, uint32_t max_len) {
    // Copy input to output buffer for in-place mutation
    out_len = in_len > max_len ? max_len : in_len;
    for(uint32_t i = 0; i < out_len; ++i) {
        out[i] = in[i];
    }

    for (int c = 0; c < rule->num_cmds; ++c) {
        const RuleCmd& cmd = rule->cmds[c];
        switch (cmd.op) {
            case RULE_NOP:
                break;
            case RULE_LOWER:
                for (uint32_t i = 0; i < out_len; ++i) {
                    if (out[i] >= 'A' && out[i] <= 'Z') out[i] += 32;
                }
                break;
            case RULE_UPPER:
                for (uint32_t i = 0; i < out_len; ++i) {
                    if (out[i] >= 'a' && out[i] <= 'z') out[i] -= 32;
                }
                break;
            case RULE_CAPITALIZE:
                if (out_len > 0) {
                    if (out[0] >= 'a' && out[0] <= 'z') out[0] -= 32;
                    for (uint32_t i = 1; i < out_len; ++i) {
                        if (out[i] >= 'A' && out[i] <= 'Z') out[i] += 32;
                    }
                }
                break;
            case RULE_TITLE_CASE:
                if (out_len > 0) {
                    bool new_word = true;
                    for (uint32_t i = 0; i < out_len; ++i) {
                        if (out[i] == ' ' || out[i] == '\t' || out[i] == '-' || out[i] == '_') {
                            new_word = true;
                        } else {
                            if (new_word) {
                                if (out[i] >= 'a' && out[i] <= 'z') out[i] -= 32;
                                new_word = false;
                            } else {
                                if (out[i] >= 'A' && out[i] <= 'Z') out[i] += 32;
                            }
                        }
                    }
                }
                break;
            case RULE_REVERSE:
                for (uint32_t i = 0; i < out_len / 2; ++i) {
                    uint8_t tmp = out[i];
                    out[i] = out[out_len - 1 - i];
                    out[out_len - 1 - i] = tmp;
                }
                break;
            case RULE_REFLECT:
                if (out_len * 2 <= max_len) {
                    for (uint32_t i = 0; i < out_len; ++i) {
                        out[out_len + i] = out[out_len - 1 - i];
                    }
                    out_len *= 2;
                }
                break;
            case RULE_DUPLICATE:
                if (out_len * 2 <= max_len) {
                    for (uint32_t i = 0; i < out_len; ++i) {
                        out[out_len + i] = out[i];
                    }
                    out_len *= 2;
                }
                break;
            case RULE_APPEND:
                if (out_len < max_len) {
                    out[out_len++] = cmd.p1;
                }
                break;
            case RULE_PREPEND:
                if (out_len < max_len) {
                    for (int32_t i = out_len; i > 0; --i) {
                        out[i] = out[i - 1];
                    }
                    out[0] = cmd.p1;
                    out_len++;
                }
                break;
            case RULE_DELETE_FIRST:
                if (out_len > 0) {
                    for (uint32_t i = 0; i < out_len - 1; ++i) {
                        out[i] = out[i + 1];
                    }
                    out_len--;
                }
                break;
            case RULE_INVERT_CASE:
                if (out_len > 0) {
                    if (out[0] >= 'A' && out[0] <= 'Z') out[0] += 32;
                    for (uint32_t i = 1; i < out_len; ++i) {
                        if (out[i] >= 'a' && out[i] <= 'z') out[i] -= 32;
                    }
                }
                break;
            case RULE_APPEND_DUP: {
                uint32_t N = cmd.p1;
                uint32_t orig_len = out_len;
                for (uint32_t n = 0; n < N && out_len < max_len; ++n) {
                    for (uint32_t i = 0; i < orig_len && out_len < max_len; ++i) {
                        out[out_len++] = out[i];
                    }
                }
                break;
            }
            case RULE_TRUNCATE: {
                uint32_t N = cmd.p1;
                if (out_len > N) {
                    out_len = N;
                }
                break;
            }
            case RULE_EXTRACT: {
                uint32_t N = cmd.p1;
                uint32_t M = cmd.p2;
                if (N < out_len) {
                    uint32_t extract_len = (N + M <= out_len) ? M : (out_len - N);
                    for (uint32_t i = 0; i < extract_len; ++i) {
                        out[i] = out[N + i];
                    }
                    out_len = extract_len;
                } else {
                    out_len = 0;
                }
                break;
            }
            case RULE_DUP_FIRST: {
                uint32_t N = cmd.p1;
                if (N > out_len) N = out_len;
                if (out_len + N <= max_len) {
                    for (int32_t i = out_len - 1; i >= 0; --i) {
                        out[i + N] = out[i];
                    }
                    out_len += N;
                }
                break;
            }
            case RULE_DUP_LAST: {
                uint32_t N = cmd.p1;
                if (N > out_len) N = out_len;
                if (out_len + N <= max_len) {
                    for (uint32_t i = 0; i < N; ++i) {
                        out[out_len + i] = out[out_len - N + i];
                    }
                    out_len += N;
                }
                break;
            }
            case RULE_PURGE_CHAR: {
                uint32_t write_idx = 0;
                for (uint32_t i = 0; i < out_len; ++i) {
                    if (out[i] != cmd.p1) {
                        out[write_idx++] = out[i];
                    }
                }
                out_len = write_idx;
                break;
            }
            case RULE_DUP_FIRST_N_TIMES: {
                uint32_t N = cmd.p1;
                if (out_len > 0 && out_len + N <= max_len) {
                    uint8_t c = out[0];
                    for (int32_t i = out_len - 1; i >= 0; --i) {
                        out[i + N] = out[i];
                    }
                    for (uint32_t i = 0; i < N; ++i) {
                        out[i] = c;
                    }
                    out_len += N;
                }
                break;
            }
            case RULE_DUP_LAST_N_TIMES: {
                uint32_t N = cmd.p1;
                if (out_len > 0 && out_len + N <= max_len) {
                    uint8_t c = out[out_len - 1];
                    for (uint32_t i = 0; i < N; ++i) {
                        out[out_len++] = c;
                    }
                }
                break;
            }
            case RULE_DELETE_LAST:
                if (out_len > 0) {
                    out_len--;
                }
                break;
            case RULE_DELETE_AT:
                if (cmd.p1 < out_len) {
                    for (uint32_t i = cmd.p1; i < out_len - 1; ++i) {
                        out[i] = out[i + 1];
                    }
                    out_len--;
                }
                break;
            case RULE_REPLACE:
                for (uint32_t i = 0; i < out_len; ++i) {
                    if (out[i] == cmd.p1) out[i] = cmd.p2;
                }
                break;
            case RULE_TOGGLE_ALL:
                for (uint32_t i = 0; i < out_len; ++i) {
                    if (out[i] >= 'a' && out[i] <= 'z') out[i] -= 32;
                    else if (out[i] >= 'A' && out[i] <= 'Z') out[i] += 32;
                }
                break;
            case RULE_TOGGLE_AT:
                if (cmd.p1 < out_len) {
                    if (out[cmd.p1] >= 'a' && out[cmd.p1] <= 'z') out[cmd.p1] -= 32;
                    else if (out[cmd.p1] >= 'A' && out[cmd.p1] <= 'Z') out[cmd.p1] += 32;
                }
                break;
            case RULE_OVERWRITE:
                if (cmd.p1 < out_len) {
                    out[cmd.p1] = cmd.p2;
                }
                break;
            case RULE_OMIT:
                if (cmd.p1 < out_len && cmd.p2 > 0) {
                    uint32_t start = cmd.p1;
                    uint32_t count = cmd.p2;
                    if (start + count > out_len) count = out_len - start;
                    for (uint32_t i = start; i + count < out_len; ++i) {
                        out[i] = out[i + count];
                    }
                    out_len -= count;
                }
                break;
            case RULE_INSERT:
                if (cmd.p1 <= out_len && out_len < max_len) {
                    for (int32_t i = out_len; i > cmd.p1; --i) {
                        out[i] = out[i - 1];
                    }
                    out[cmd.p1] = cmd.p2;
                    out_len++;
                }
                break;
            case RULE_INSERT_MISSING:
                if (cmd.p1 > out_len && cmd.p1 < max_len) {
                    while (out_len < cmd.p1) out[out_len++] = ' ';
                    out[out_len++] = cmd.p2;
                }
                break;
            case RULE_INC_AT:
                if (cmd.p1 < out_len) out[cmd.p1]++;
                break;
            case RULE_DEC_AT:
                if (cmd.p1 < out_len) out[cmd.p1]--;
                break;
            case RULE_REPLACE_N_NEXT:
                if (cmd.p1 + 1 < out_len) out[cmd.p1] = out[cmd.p1 + 1];
                break;
            case RULE_REPLACE_N_PREV:
                if (cmd.p1 < out_len && cmd.p1 > 0) out[cmd.p1] = out[cmd.p1 - 1];
                break;
            case RULE_SWAP:
                if (cmd.p1 < out_len && cmd.p2 < out_len) {
                    uint8_t t = out[cmd.p1];
                    out[cmd.p1] = out[cmd.p2];
                    out[cmd.p2] = t;
                }
                break;
            case RULE_SHIFT_LEFT:
                if (out_len > 1) {
                    uint8_t first = out[0];
                    for (uint32_t i = 0; i < out_len - 1; ++i) out[i] = out[i + 1];
                    out[out_len - 1] = first;
                }
                break;
            case RULE_SHIFT_RIGHT:
                if (out_len > 1) {
                    uint8_t last = out[out_len - 1];
                    for (int32_t i = out_len - 1; i > 0; --i) out[i] = out[i - 1];
                    out[0] = last;
                }
                break;
            case RULE_BIT_SHIFT_LEFT:
                if (cmd.p1 < out_len) out[cmd.p1] <<= 1;
                break;
            case RULE_BIT_SHIFT_RIGHT:
                if (cmd.p1 < out_len) out[cmd.p1] >>= 1;
                break;
            case RULE_SWAP_FRONT:
                if (out_len >= 2) {
                    uint8_t t = out[0];
                    out[0] = out[1];
                    out[1] = t;
                }
                break;
            case RULE_SWAP_BACK:
                if (out_len >= 2) {
                    uint8_t t = out[out_len - 1];
                    out[out_len - 1] = out[out_len - 2];
                    out[out_len - 2] = t;
                }
                break;
            case RULE_DUP_ALL:
                if (out_len * 2 <= max_len) {
                    for (int32_t i = out_len - 1; i >= 0; --i) {
                        out[i * 2] = out[i];
                        out[i * 2 + 1] = out[i];
                    }
                    out_len *= 2;
                }
                break;
            case RULE_REJECT_LESS:
                if (out_len >= cmd.p1) return false;
                break;
            case RULE_REJECT_GREATER:
                if (out_len <= cmd.p1) return false;
                break;
            case RULE_REJECT_EQUAL:
                if (out_len != cmd.p1) return false;
                break;
            case RULE_REJECT_CONTAINS: {
                bool found = false;
                for (uint32_t i = 0; i < out_len; ++i) {
                    if (out[i] == cmd.p1) { found = true; break; }
                }
                if (found) return false;
                break;
            }
            case RULE_REJECT_NOT_CONTAINS: {
                bool found = false;
                for (uint32_t i = 0; i < out_len; ++i) {
                    if (out[i] == cmd.p1) { found = true; break; }
                }
                if (!found) return false;
                break;
            }
            case RULE_REJECT_STARTS:
                if (out_len == 0 || out[0] != cmd.p1) return false;
                break;
            case RULE_REJECT_ENDS:
                if (out_len == 0 || out[out_len - 1] != cmd.p1) return false;
                break;
            case RULE_REJECT_EQUAL_AT:
                if (cmd.p1 >= out_len || out[cmd.p1] != cmd.p2) return false;
                break;
        }
    }
    return true;
}
#endif // __CUDACC__

#endif // HYDRA_RULES_CUH
