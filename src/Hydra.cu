/*
 * ======================================================================================
 * HYDRA V5
 * ======================================================================================
 * Tool for recovering a partial private key or partial BIP39 seed.
 *
 * USAGE :
 *   Hex key mode  (unknown nibbles = '#') :
 *     ./Hydra 7cb5da6f77574214a59#f40dc45739eda5e532804f24af675e3##339f#1fe9c4 1AddressBTC
 *     ./Hydra 7cb5da6f77574214a59#f40dc45739eda5e532804f24af675e3##339f#1fe9c4 0x1234...abcd
 *
 *   BIP39 seed mode (unknown words = '#') :
 *     ./Hydra "word1 word2 # word4 # word6 word7 word8 word9 word10 word11 word12" 1AddressBTC
 *
 * OUTPUT :
 *   - "Not found in N candidates"
 *   - "FOUND! Private key: 0x..."
 *
 * DEPENDENCIES :
 *   - CUDA sm_86 / sm_89 / sm_120 release targets
 *   - OpenSSL (libssl-dev) for CPU-side ECC precomputation
 *   - Standalone : ECC.h, Hash.h, Hash.cu, Gray.h, HydraCommon.h
 * ======================================================================================
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <csignal>
#include <cctype>
#include <cmath>
#include <mutex>
#include <atomic>
#include <thread>
#include <ctime>
#include <stdexcept>
#include <limits>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// OpenSSL for CPU-side ECC precomputation
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <openssl/hmac.h>
#include <openssl/evp.h>

#include "HydraCommon.h"
#include "Bloom.h"
#include "RuleParser.h"
#include "ECC.h"
#include "Hash.cuh"
#include "Gray.h"
#include "BsgsCommon.cuh"
#include "BsgsWif.cuh"
#include "BsgsHex.cuh"
#include "BIP39_Dict.h"
#include "ElectrumV1Words.h"
#include "Seed.cuh"
#include "Wif.cuh"
#include "Platform.h"
#include "HttpClient.h"
#include "Brainwallet.cuh"
#include "ElectrumV1.cuh"

// =================================================================================
// SIGNAL HANDLER
// =================================================================================
static volatile sig_atomic_t g_sigint = 0;

static bool brain_gen_table(int w,
    uint64_t** d_gx_out, uint64_t** d_gy_out,
    int* cols_out, int* stride_out);

// =================================================================================
// RESUME SNAPSHOT
// =================================================================================
static const char* HYDRA_RESUME_FILE  = "hydra.resume.log";
static const char* HYDRA_RESUME_MAGIC = "HYDRA_RESUME_V1";
static constexpr uint64_t HYDRA_RESUME_BSGS_HOST_BABY = 1ULL << 0;
static constexpr uint64_t HYDRA_RESUME_BSGS_HEX_BABY75 = 1ULL << 1;
static constexpr uint64_t HYDRA_RESUME_BSGS_HEX_COMPACT6 = 1ULL << 2;
static constexpr uint64_t HYDRA_RESUME_BSGS_WIF_BABY55 = 1ULL << 3;

struct ResumeState {
    bool        active = false;
    std::string mode;
    std::string arg1;
    std::string arg2;
    uint64_t    offset = 0;
    uint64_t    total = 0;
    uint64_t    dict_byte_offset = 0;
    uint64_t    tested = 0;
    uint64_t    bsgs_flags = 0;
};

static bool parse_resume_string(const std::string& line, const char* key, std::string& out) {
    std::string prefix = std::string(key) + " ";
    if (line.rfind(prefix, 0) != 0) return false;
    std::istringstream iss(line.substr(prefix.size()));
    iss >> std::quoted(out);
    return !iss.fail();
}

static bool parse_resume_u64(const std::string& line, const char* key, uint64_t& out) {
    std::string prefix = std::string(key) + " ";
    if (line.rfind(prefix, 0) != 0) return false;
    std::istringstream iss(line.substr(prefix.size()));
    iss >> out;
    return !iss.fail();
}

static bool write_resume_snapshot(const ResumeState& st) {
    std::ostringstream oss;
    oss << "magic " << std::quoted(std::string(HYDRA_RESUME_MAGIC)) << "\n";
    oss << "mode " << std::quoted(st.mode) << "\n";
    oss << "arg1 " << std::quoted(st.arg1) << "\n";
    oss << "arg2 " << std::quoted(st.arg2) << "\n";
    oss << "offset " << st.offset << "\n";
    oss << "total " << st.total << "\n";
    oss << "dict_byte_offset " << st.dict_byte_offset << "\n";
    oss << "tested " << st.tested << "\n";
    oss << "bsgs_flags " << st.bsgs_flags << "\n";

    const std::string payload = oss.str();
    std::string error;
    if (!hydra_platform::write_atomic_file(HYDRA_RESUME_FILE, payload, &error)) {
        std::cerr << "[Resume] Snapshot write failed: " << error << "\n";
        return false;
    }
    return true;
}

static void clear_resume_snapshot() {
    std::string error;
    if (!hydra_platform::remove_file_if_exists(HYDRA_RESUME_FILE, &error) && !error.empty()) {
        std::cerr << "[Resume] Cleanup failed: " << error << "\n";
    }
}

static bool load_resume_snapshot(ResumeState& st) {
    std::ifstream f(HYDRA_RESUME_FILE);
    if (!f.is_open()) return false;

    std::string line, magic;
    while (std::getline(f, line)) {
        if (parse_resume_string(line, "magic", magic)) continue;
        if (parse_resume_string(line, "mode", st.mode)) continue;
        if (parse_resume_string(line, "arg1", st.arg1)) continue;
        if (parse_resume_string(line, "arg2", st.arg2)) continue;
        if (parse_resume_u64(line, "offset", st.offset)) continue;
        if (parse_resume_u64(line, "total", st.total)) continue;
        if (parse_resume_u64(line, "dict_byte_offset", st.dict_byte_offset)) continue;
        if (parse_resume_u64(line, "tested", st.tested)) continue;
        if (parse_resume_u64(line, "bsgs_flags", st.bsgs_flags)) continue;
    }

    if (magic != HYDRA_RESUME_MAGIC || st.mode.empty()) return false;
    st.active = true;
    return true;
}

static ResumeState make_resume_state(const std::string& mode, const std::string& arg1, const std::string& arg2) {
    ResumeState st;
    st.active = true;
    st.mode = mode;
    st.arg1 = arg1;
    st.arg2 = arg2;
    return st;
}

static void print_resume_hint() {
    std::cout << "[Resume] Checkpoint saved -- resume with: ./Hydra resume\n";
}

static constexpr double HYDRA_RESUME_INTERVAL_SEC = 5.0;
static constexpr uint32_t HYDRA_WIF_BSGS_AUTO_MIN_UNKNOWN = 6;
static constexpr uint32_t HYDRA_HEX_BSGS_AUTO_MIN_UNKNOWN = 9;

template <typename TimePoint>
static bool should_write_resume_snapshot(
    const TimePoint& last_snapshot,
    const TimePoint& now,
    bool force_now)
{
    if (force_now) return true;
    return std::chrono::duration<double>(now - last_snapshot).count() >= HYDRA_RESUME_INTERVAL_SEC;
}

static uint64_t hydra_vram_budget_bytes(uint64_t free_bytes) {
    const uint64_t reserve = 512ULL * 1024ULL * 1024ULL;
    const uint64_t by_fraction = (uint64_t)((double)free_bytes * 0.90);
    const uint64_t by_reserve = (free_bytes > reserve) ? (free_bytes - reserve) : (free_bytes / 2);
    return std::min(by_fraction, by_reserve);
}

static uint64_t hydra_host_ram_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return 0;
    return (uint64_t)status.ullTotalPhys;
#else
    struct sysinfo info;
    if (sysinfo(&info) != 0) return 0;
    return (uint64_t)info.totalram * (uint64_t)info.mem_unit;
#endif
}

static uint64_t hydra_host_available_ram_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return 0;
    return (uint64_t)status.ullAvailPhys;
#else
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    uint64_t mem_available_kb = 0;
    while (std::getline(meminfo, line)) {
        if (line.compare(0, 13, "MemAvailable:") == 0) {
            std::istringstream iss(line.substr(13));
            iss >> mem_available_kb;
            return mem_available_kb * 1024ULL;
        }
    }
    // Fallback if MemAvailable is missing
    struct sysinfo info;
    if (sysinfo(&info) != 0) return 0;
    return ((uint64_t)info.freeram + (uint64_t)info.bufferram) * (uint64_t)info.mem_unit;
#endif
}

static bool hydra_is_wsl() {
#ifdef _WIN32
    return false;
#else
    std::ifstream f("/proc/sys/kernel/osrelease");
    std::string s;
    if (!f || !std::getline(f, s)) return false;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    return s.find("microsoft") != std::string::npos || s.find("wsl") != std::string::npos;
#endif
}

static std::string hydra_lower_ascii(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

static bool is_electrum_v2_flag(const std::string& s) {
    return hydra_lower_ascii(s) == "--electrumv2";
}

static bool is_electrum_v1_flag(const std::string& s) {
    return hydra_lower_ascii(s) == "--electrumv1";
}

// =================================================================================
// API CONFIGURATION & NOTIFICATIONS
// =================================================================================
// Credentials loaded from telegram.txt at startup (never hardcoded in source)
// telegram.txt format — one value per line:
//   line 1 : TELEGRAM_BOT_TOKEN
//   line 2 : TELEGRAM_CHAT_ID
static std::string g_telegram_token;
static std::string g_telegram_chat_id;

static const char* TELEGRAM_BOT_TOKEN = nullptr;
static const char* TELEGRAM_CHAT_ID   = nullptr;

static constexpr const char* BLOCKSCOUT_HOST = "eth.blockscout.com";

static void load_tokens(const char* path = "telegram.txt") {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[Token] Warning: " << path << " not found -- Telegram disabled\n";
        return;
    }
    std::string token;
    std::string chat_id;
    std::getline(f, token);
    std::getline(f, chat_id);

    auto trim_right = [](std::string& s) {
        // Strip trailing whitespace/CR
        while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' '))
            s.pop_back();
    };
    trim_right(token);
    trim_right(chat_id);

    g_telegram_token = token;
    g_telegram_chat_id = chat_id;
    if (!g_telegram_token.empty()) TELEGRAM_BOT_TOKEN = g_telegram_token.c_str();
    if (!g_telegram_chat_id.empty()) TELEGRAM_CHAT_ID = g_telegram_chat_id.c_str();

    if (!g_telegram_token.empty())
        std::cout << "[Token] Telegram token loaded\n";
}

static std::mutex g_log_mutex;
static std::atomic<int> g_api_errors{0};  // bloom hits that could not be verified (network error)

enum class BsgsBabyOverride : uint32_t {
    AUTO = 0,
    HEX_BABY75_VRAM,
    HEX_BABY8,
    WIF_BABY5_VRAM,
    WIF_BABY55
};

// Print end-of-run summary (called in all 4 modes)
static void print_search_summary(bool found) {
    if (!found) {
        std::cout << "  No wallet found.\n";
    }
    if (g_api_errors > 0) {
        std::cout << "  /!\\ " << g_api_errors << " unverified API error(s) -- check errors.json\n";
    }
}

// =================================================================================
// NETWORK & NOTIFICATION HELPERS
// =================================================================================

static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if      (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else                out += c;
    }
    return out;
}

// Log error to errors.json (Telegram failure or balance API failure)
static void log_error(const std::string& type, const std::string& detail) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    std::ofstream f("errors.json", std::ios::app);
    if (!f.is_open()) { std::cerr << "[Logger] Cannot open errors.json\n"; return; }

    std::time_t now = std::time(nullptr);
    char tbuf[32];
    std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

    f << "{\"ts\":\"" << tbuf << "\","
      << "\"type\":\"" << json_escape(type) << "\","
      << "\"detail\":\"" << json_escape(detail) << "\"}\n";
    f.flush();
}

// Returns true if the message was delivered (Telegram API ack ok:true)
static bool send_telegram(const std::string& message) {
    if (!TELEGRAM_BOT_TOKEN || !TELEGRAM_CHAT_ID ||
        std::string(TELEGRAM_BOT_TOKEN).empty() ||
        std::string(TELEGRAM_CHAT_ID).empty()) {
        return false;
    }

    std::ostringstream oss;
    oss << std::hex;
    for (unsigned char c : message) {
        if (isalnum(c) || c=='-' || c=='_' || c=='.' || c=='~') oss << c;
        else oss << '%' << std::uppercase << std::setw(2) << std::setfill('0') << (int)c;
    }
    std::string path = "/bot" + std::string(TELEGRAM_BOT_TOKEN) +
                       "/sendMessage?chat_id=" + std::string(TELEGRAM_CHAT_ID) +
                       "&text=" + oss.str() + "&parse_mode=Markdown";
    try {
        std::string resp = hydra_http::https_get("api.telegram.org", path);
        // Telegram API returns {"ok":true,...} or {"ok":false,...}
        if (resp.find("\"ok\":true") != std::string::npos) {
            std::cout << "  [Telegram] Message delivered.\n";
            return true;
        } else {
            std::string err = (resp.size() > 200) ? resp.substr(0, 200) : resp;
            std::cout << "  [Telegram] Failed : " << err << "\n";
            log_error("telegram_failed", err);
            return false;
        }
    } catch (const std::exception& e) {
        std::cout << "  [Telegram] Network error : " << e.what() << "\n";
        log_error("telegram_network_error", e.what());
        return false;
    }
}

static double parse_btc_balance(const std::string& raw) {
    if (raw.empty()) return 0.0;
    try { return std::stold(raw) / 1e8; } catch (...) { return 0.0; }
}

static double parse_eth_balance(const std::string& json) {
    std::string key = "\"result\":\"";
    size_t pos = json.find(key);
    if (pos == std::string::npos) return 0.0;
    size_t s = pos + key.length();
    size_t e = json.find("\"", s);
    if (e == std::string::npos) return 0.0;
    try { return std::stold(json.substr(s, e-s)) / 1e18; } catch (...) { return 0.0; }
}

// Log API error : only log hits we could not verify (network down).
// Sole purpose : review manually after the run.
static void log_api_error(
    const std::string& priv_hex,
    const std::string& btc_legacy,
    const std::string& btc_segwit,
    const std::string& eth_addr,
    const std::string& extra = "")
{
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_api_errors++;

    std::ofstream f("errors.json", std::ios::app);
    if (!f.is_open()) { std::cerr << "[Logger] Cannot open errors.json\n"; return; }

    std::time_t now = std::time(nullptr);
    char tbuf[32];
    std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

    f << "{\"ts\":\"" << tbuf << "\","
      << "\"type\":\"api_balance_error\","
      << "\"pk\":\"" << json_escape(priv_hex) << "\","
      << "\"btc_legacy\":\"" << json_escape(btc_legacy) << "\","
      << "\"btc_segwit\":\"" << json_escape(btc_segwit) << "\","
      << "\"eth\":\"" << json_escape(eth_addr) << "\"";
    if (!extra.empty())
        f << ",\"extra\":\"" << json_escape(extra) << "\"";
    f << "}\n";
    f.flush();
}

// Send a victory notification via Telegram for all modes.
// key_info  : private key hex, WIF, or seed phrase
// addr_info : relevant address(es) to display
static void notify_victory(const std::string& mode_title,
                           const std::string& key_info,
                           const std::string& addr_info) {
    std::ostringstream msg;
    msg << "*HYDRA - " << mode_title << "* \xF0\x9F\x8F\x86\n\n"
        << key_info << "\n\n"
        << addr_info;
    bool ok = send_telegram(msg.str());
    if (!ok) {
        log_error("telegram_victory_lost",
                  mode_title + " | " + key_info + " | " + addr_info);
        std::cout << "  /!\\ Telegram failed -- details saved to errors.json\n";
    }
}

// Returns true if balance > 0 (real hit), false on false positive or network error.
// On network error : logs to errors.json and continues.
static bool check_balances_and_notify(
    const uint8_t* key32,
    const std::string& btc_legacy,
    const std::string& btc_segwit,
    const std::string& eth_addr)
{
    std::ostringstream pk_ss;
    pk_ss << std::hex << std::setfill('0');
    for (int i = 0; i < 32; ++i) pk_ss << std::setw(2) << (int)key32[i];
    const std::string priv_hex = pk_ss.str();

    std::cout << "\n!!! BLOOM HIT !!!\n";
    std::cout << "  Private key : " << priv_hex << "\n";
    std::cout << "  BTC legacy  : " << btc_legacy << "\n";
    std::cout << "  BTC segwit  : " << btc_segwit << "\n";
    std::cout << "  ETH         : " << eth_addr << "\n";
    std::cout << "  [API] Checking balances...\n";

    bool network_error = false;
    double btc_bal = 0.0, eth_bal = 0.0;

    try {
        std::string btc_raw = hydra_http::https_get("blockchain.info", "/q/addressbalance/" + btc_legacy);
        if (btc_raw.empty()) network_error = true;
        else btc_bal = parse_btc_balance(btc_raw);

        std::string eth_path = "/api?module=account&action=balance&address=" +
                               eth_addr + "&tag=latest";
        std::string eth_raw = hydra_http::https_get(BLOCKSCOUT_HOST, eth_path);
        if (eth_raw.empty()) network_error = true;
        else eth_bal = parse_eth_balance(eth_raw);

    } catch (...) { network_error = true; }

    if (network_error) {
        std::cout << "  > Network error -- logged to errors.json, continuing.\n";
        log_api_error(priv_hex, btc_legacy, btc_segwit, eth_addr);
        return false;
    }

    std::cout << "  > BTC : " << std::fixed << std::setprecision(8) << btc_bal << " BTC\n";
    std::cout << "  > ETH : " << std::fixed << std::setprecision(8) << eth_bal << " ETH\n";

    if (btc_bal > 0.0 || eth_bal > 0.0) {
        std::cout << "\n*** NON-ZERO BALANCE -- WALLET FOUND ***\n";
        std::ostringstream key_info, addr_info;
        key_info << "*Private Key:*\n`" << priv_hex << "`";
        if (btc_bal > 0.0 && eth_bal > 0.0) {
            addr_info << "*BTC:* `" << btc_legacy << "`\n`"
                      << std::fixed << std::setprecision(8) << btc_bal << " BTC`\n"
                      << "*ETH:* `" << eth_addr << "`\n`"
                      << std::fixed << std::setprecision(8) << eth_bal << " ETH`";
        } else if (btc_bal > 0.0) {
            addr_info << "*BTC:* `" << btc_legacy << "`\n`"
                      << std::fixed << std::setprecision(8) << btc_bal << " BTC`";
        } else {
            addr_info << "*ETH:* `" << eth_addr << "`\n`"
                      << std::fixed << std::setprecision(8) << eth_bal << " ETH`";
        }
        notify_victory("WALLET FOUND \xF0\x9F\x92\xB8", key_info.str(), addr_info.str());
        return true;
    }

    std::cout << "  > Zero balance -- false positive, continuing.\n";
    return false;
}

// =================================================================================
// 1. CPU HELPERS : SHA256 + BASE58 + KECCAK (for decoding the target address)
// =================================================================================

static const uint32_t K_CPU[64] = {
    0x428A2F98,0x71374491,0xB5C0FBCF,0xE9B5DBA5,0x3956C25B,0x59F111F1,
    0x923F82A4,0xAB1C5ED5,0xD807AA98,0x12835B01,0x243185BE,0x550C7DC3,
    0x72BE5D74,0x80DEB1FE,0x9BDC06A7,0xC19BF174,0xE49B69C1,0xEFBE4786,
    0x0FC19DC6,0x240CA1CC,0x2DE92C6F,0x4A7484AA,0x5CB0A9DC,0x76F988DA,
    0x983E5152,0xA831C66D,0xB00327C8,0xBF597FC7,0xC6E00BF3,0xD5A79147,
    0x06CA6351,0x14292967,0x27B70A85,0x2E1B2138,0x4D2C6DFC,0x53380D13,
    0x650A7354,0x766A0ABB,0x81C2C92E,0x92722C85,0xA2BFE8A1,0xA81A664B,
    0xC24B8B70,0xC76C51A3,0xD192E819,0xD6990624,0xF40E3585,0x106AA070,
    0x19A4C116,0x1E376C08,0x2748774C,0x34B0BCB5,0x391C0CB3,0x4ED8AA4A,
    0x5B9CCA4F,0x682E6FF3,0x748F82EE,0x78A5636F,0x84C87814,0x8CC70208,
    0x90BEFFFA,0xA4506CEB,0xBEF9A3F7,0xC67178F2
};

#define ROTR(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define CH(x,y,z)  (((x)&(y))^(~(x)&(z)))
#define MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))
#define SIG0(x) (ROTR(x,2)^ROTR(x,13)^ROTR(x,22))
#define SIG1(x) (ROTR(x,6)^ROTR(x,11)^ROTR(x,25))
#define sig0(x) (ROTR(x,7)^ROTR(x,18)^((x)>>3))
#define sig1(x) (ROTR(x,17)^ROTR(x,19)^((x)>>10))

static void sha256_cpu_block(uint32_t state[8], const uint8_t block[64]) {
    uint32_t w[64];
    for (int i=0; i<16; i++)
        w[i]=(block[i*4]<<24)|(block[i*4+1]<<16)|(block[i*4+2]<<8)|block[i*4+3];
    for (int i=16; i<64; i++)
        w[i]=sig1(w[i-2])+w[i-7]+sig0(w[i-15])+w[i-16];
    uint32_t a=state[0],b=state[1],c=state[2],d=state[3],
             e=state[4],f=state[5],g=state[6],h=state[7];
    for (int i=0; i<64; i++) {
        uint32_t t1=h+SIG1(e)+CH(e,f,g)+K_CPU[i]+w[i];
        uint32_t t2=SIG0(a)+MAJ(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

static void sha256_cpu(const uint8_t *data, size_t len, uint8_t out[32]) {
    uint32_t state[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                       0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint8_t block[64]={0};
    size_t remain = len;
    const uint8_t *ptr = data;
    while (remain >= 64) { sha256_cpu_block(state, ptr); ptr+=64; remain-=64; }
    memcpy(block, ptr, remain);
    block[remain]=0x80;
    if (remain >= 56) {
        sha256_cpu_block(state, block);
        memset(block, 0, 64);
    }
    uint64_t bits = len*8;
    for (int i=0; i<8; i++) block[63-i]=(bits>>(i*8))&0xFF;
    sha256_cpu_block(state, block);
    for (int i=0; i<8; i++) {
        out[i*4]=(state[i]>>24)&0xFF; out[i*4+1]=(state[i]>>16)&0xFF;
        out[i*4+2]=(state[i]>>8)&0xFF; out[i*4+3]=state[i]&0xFF;
    }
}

// BTC Base58Check -> hash160
static bool base58Decode(const std::string &addr, uint8_t out[25]) {
    static const char *alpha="123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    uint8_t result[25]={0};
    for (char c : addr) {
        const char *p=strchr(alpha, c);
        if (!p) return false;
        int carry=(int)(p-alpha);
        for (int i=24; i>=0; --i) { carry+=58*result[i]; result[i]=carry%256; carry/=256; }
    }
    memcpy(out, result, 25);
    return true;
}

static bool addrToHash160(const std::string &addr, uint8_t hash160[20]) {
    if (addr.size() < 26 || addr.size() > 35) return false;
    uint8_t decoded[25];
    if (!base58Decode(addr, decoded)) return false;
    uint8_t check[32];
    sha256_cpu(decoded, 21, check);
    sha256_cpu(check, 32, check);
    if (memcmp(check, decoded+21, 4)!=0) return false;
    memcpy(hash160, decoded+1, 20);
    return true;
}

// ETH 0x... -> 20 bytes

// SegWit bech32 decode -> hash160 (P2WPKH bc1q...)
static const int8_t BECH32_REV[128] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    15,-1,10,17,21,20,26,30, 7, 5,-1,-1,-1,-1,-1,-1,
    -1,29,-1,24,13,25, 9, 8,23,-1,18,22,31,27,19,-1,
     1, 0, 3,16,11,28,12,14, 6, 4, 2,-1,-1,-1,-1,-1,
    -1,29,-1,24,13,25, 9, 8,23,-1,18,22,31,27,19,-1,
     1, 0, 3,16,11,28,12,14, 6, 4, 2,-1,-1,-1,-1,-1
};
static bool bech32_decode_hash160(const std::string &addr, uint8_t hash160[20]) {
    size_t sep = addr.rfind('1');
    if(sep == std::string::npos || sep < 2) return false;
    if(addr.substr(0,sep) != "bc") return false;
    std::string ds = addr.substr(sep+1);
    std::vector<uint8_t> d5;
    for(char ch : ds){
        unsigned char uc=(unsigned char)ch;
        if(uc>=128) return false;
        int v=BECH32_REV[uc]; if(v<0) return false;
        d5.push_back((uint8_t)v);
    }
    if(d5.size()!=39 || d5[0]!=0) return false;
    uint32_t acc=0; int bits=0,out_idx=0;
    for(int i=1;i<=32;i++){
        acc=(acc<<5)|d5[i]; bits+=5;
        if(bits>=8){ bits-=8; if(out_idx>=20) return false; hash160[out_idx++]=(uint8_t)(acc>>bits); }
    }
    return out_idx==20;
}
static bool addrToHash160Any(const std::string &addr, uint8_t hash160[20]) {
    if(addr.size()>=4 && addr[0]=='b' && addr[1]=='c' && addr[2]=='1')
        return bech32_decode_hash160(addr,hash160);
    return addrToHash160(addr, hash160);
}

static bool ethAddrToBytes(const std::string &addr, uint8_t out[20]) {
    std::string s = addr;
    if (s.size() >= 2 && s[0]=='0' && (s[1]=='x'||s[1]=='X')) s=s.substr(2);
    if (s.size() != 40) return false;
    for (int i=0; i<20; i++) {
        auto h=[](char c)->int{
            if(c>='0'&&c<='9') return c-'0';
            if(c>='a'&&c<='f') return c-'a'+10;
            if(c>='A'&&c<='F') return c-'A'+10;
            return -1;
        };
        int hi=h(s[i*2]), lo=h(s[i*2+1]);
        if(hi<0||lo<0) return false;
        out[i]=(uint8_t)(hi<<4|lo);
    }
    return true;
}

// =================================================================================
// 2. HEX MASK PARSING (HEX MODE)
//    "7cb5...XX...e4" -> k_fixed (var bits = 0) + var_bit_positions[]
// =================================================================================

struct MaskParseResult {
    uint8_t  k_fixed[32];          // fixed key (big-endian, var bits = 0)
    std::vector<int> var_bit_positions; // variable bit positions (LSB=0)
    bool valid = false;
    uint64_t total_candidates = 0;
};

static uint64_t saturated_pow2_u64(int bits)
{
    if (bits < 0) return 0;
    if (bits >= 64) return std::numeric_limits<uint64_t>::max();
    return 1ULL << bits;
}

static double pow2_as_double(int bits)
{
    return std::ldexp(1.0, bits);
}

static std::string format_pow2_candidate_count(int bits)
{
    if (bits < 64)
        return std::to_string(1ULL << bits);
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(6) << pow2_as_double(bits);
    return oss.str();
}

static MaskParseResult parseMask(const std::string &mask) {
    MaskParseResult r;

    // Strip optional 0x prefix
    std::string s = mask;
    if (s.size() >= 2 && s[0]=='0' && (s[1]=='x'||s[1]=='X')) s=s.substr(2);

    if (s.size() != 64) {
        std::cerr << "Error: hex mask must be exactly 64 characters (32 bytes).\n";
        return r;
    }

    memset(r.k_fixed, 0, 32);

    // Walk nibble by nibble (left to right = MSB to LSB)
    for (int n = 0; n < 64; n++) {
        char c = s[n];
        // Nibble position in k_fixed (big-endian) :
        // nibble n -> byte n/2, bits [7-4] if even, [3-0] if odd
        int byte_idx = n / 2;
        int shift = (n % 2 == 0) ? 4 : 0;

        if (c == '#') {
            // 4 variable bits
            // Global bit index (LSB=0) : nibble n maps to bits
            // [255-n*4 .. 255-n*4-3] in MSB=255 notation
            for (int b = 3; b >= 0; b--) {
                int bit_index = (63 - n) * 4 + b; // bit global (LSB=0)
                r.var_bit_positions.push_back(bit_index);
            }
            // k_fixed keeps 0 for these nibbles
        } else {
            int nibble_val = -1;
            if (c>='0'&&c<='9') nibble_val=c-'0';
            else if (c>='a'&&c<='f') nibble_val=c-'a'+10;
            else if (c>='A'&&c<='F') nibble_val=c-'A'+10;
            else {
                std::cerr << "Error: invalid character '" << c << "' in mask.\n";
                return r;
            }
            r.k_fixed[byte_idx] |= (uint8_t)(nibble_val << shift);
        }
    }

    int m = (int)r.var_bit_positions.size();
    if (m == 0) {
        std::cerr << "Error: no variable nibble ('#') in mask.\n";
        return r;
    }
    if (m > MAX_VAR_BITS) {
        std::cerr << "Error: too many variable bits (" << m << " > " << MAX_VAR_BITS << ").\n";
        return r;
    }

    r.total_candidates = saturated_pow2_u64(m);
    r.valid = true;

    std::cout << "Mask parsed  : " << m << " variable bits, "
              << format_pow2_candidate_count(m) << " candidates (2^" << m << ")\n";
    return r;
}

// =================================================================================
// 3. CPU ECC PRECOMPUTATION (OpenSSL)
//    P_base = k_fixed * G
//    Q_i    = 2^(var_bit_positions[i]) * G
// =================================================================================

// Convert uint8_t[32] big-endian to uint64_t[4] little-endian (ECC.h format)
static void be32_to_le4(const uint8_t be[32], uint64_t le[4]) {
    for (int i = 0; i < 4; i++) {
        le[i] = 0;
        for (int b = 0; b < 8; b++)
            le[i] |= ((uint64_t)be[31 - i*8 - b]) << (b*8);
    }
}

// Compute scalar * G and return x,y as little-endian uint64_t[4]
// scalar est un uint8_t[32] big-endian
static bool ec_mul_G(const uint8_t scalar_be[32], uint64_t x_le[4], uint64_t y_le[4]) {
    EC_GROUP *group = EC_GROUP_new_by_curve_name(NID_secp256k1);
    EC_POINT *P     = EC_POINT_new(group);
    BIGNUM   *k     = BN_bin2bn(scalar_be, 32, nullptr);
    BN_CTX   *ctx   = BN_CTX_new();

    bool ok = (EC_POINT_mul(group, P, k, nullptr, nullptr, ctx) == 1);

    if (ok) {
        BIGNUM *bx = BN_new(), *by = BN_new();
        EC_POINT_get_affine_coordinates(group, P, bx, by, ctx);

        uint8_t xb[32]={0}, yb[32]={0};
        BN_bn2binpad(bx, xb, 32);
        BN_bn2binpad(by, yb, 32);
        be32_to_le4(xb, x_le);
        be32_to_le4(yb, y_le);

        BN_free(bx); BN_free(by);
    }

    BN_free(k); BN_CTX_free(ctx);
    EC_POINT_free(P); EC_GROUP_free(group);
    return ok;
}

static void le4_to_be32(const uint64_t le[4], uint8_t be[32]) {
    for (int i = 0; i < 4; i++) {
        uint64_t w = le[3 - i];
        be[i*8+0] = (uint8_t)(w >> 56);
        be[i*8+1] = (uint8_t)(w >> 48);
        be[i*8+2] = (uint8_t)(w >> 40);
        be[i*8+3] = (uint8_t)(w >> 32);
        be[i*8+4] = (uint8_t)(w >> 24);
        be[i*8+5] = (uint8_t)(w >> 16);
        be[i*8+6] = (uint8_t)(w >>  8);
        be[i*8+7] = (uint8_t)(w);
    }
}

static bool ec_point_sub_affine(
    const uint64_t ax_le[4], const uint64_t ay_le[4],
    bool a_infinity,
    const uint64_t bx_le[4], const uint64_t by_le[4],
    bool b_infinity,
    uint64_t rx_le[4], uint64_t ry_le[4],
    bool& r_infinity)
{
    if (a_infinity && b_infinity) {
        memset(rx_le, 0, 4 * sizeof(uint64_t));
        memset(ry_le, 0, 4 * sizeof(uint64_t));
        r_infinity = true;
        return true;
    }
    if (b_infinity) {
        memcpy(rx_le, ax_le, 4 * sizeof(uint64_t));
        memcpy(ry_le, ay_le, 4 * sizeof(uint64_t));
        r_infinity = a_infinity;
        return true;
    }

    EC_GROUP *group = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX   *ctx   = BN_CTX_new();
    EC_POINT *A     = EC_POINT_new(group);
    EC_POINT *B     = EC_POINT_new(group);
    EC_POINT *R     = EC_POINT_new(group);
    bool ok = group && ctx && A && B && R;

    auto set_point = [&](EC_POINT* p, const uint64_t x_le[4], const uint64_t y_le[4], bool infinity) -> bool {
        if (infinity) return EC_POINT_set_to_infinity(group, p) == 1;
        uint8_t xb[32], yb[32];
        le4_to_be32(x_le, xb);
        le4_to_be32(y_le, yb);
        BIGNUM* x = BN_bin2bn(xb, 32, nullptr);
        BIGNUM* y = BN_bin2bn(yb, 32, nullptr);
        bool ret = x && y && (EC_POINT_set_affine_coordinates(group, p, x, y, ctx) == 1);
        BN_free(x); BN_free(y);
        return ret;
    };

    if (ok) ok = set_point(A, ax_le, ay_le, a_infinity);
    if (ok) ok = set_point(B, bx_le, by_le, b_infinity);
    if (ok) ok = (EC_POINT_invert(group, B, ctx) == 1);
    if (ok) ok = (EC_POINT_add(group, R, A, B, ctx) == 1);

    if (ok && EC_POINT_is_at_infinity(group, R)) {
        memset(rx_le, 0, 4 * sizeof(uint64_t));
        memset(ry_le, 0, 4 * sizeof(uint64_t));
        r_infinity = true;
    } else if (ok) {
        BIGNUM *rx = BN_new(), *ry = BN_new();
        ok = rx && ry && (EC_POINT_get_affine_coordinates(group, R, rx, ry, ctx) == 1);
        if (ok) {
            uint8_t xb[32], yb[32];
            BN_bn2binpad(rx, xb, 32);
            BN_bn2binpad(ry, yb, 32);
            be32_to_le4(xb, rx_le);
            be32_to_le4(yb, ry_le);
            r_infinity = false;
        }
        BN_free(rx); BN_free(ry);
    }

    EC_POINT_free(R);
    EC_POINT_free(B);
    EC_POINT_free(A);
    BN_CTX_free(ctx);
    EC_GROUP_free(group);
    return ok;
}

static bool secp256k1_decompress_y_from_x(
    const uint64_t x_le[4],
    uint8_t y_parity,
    uint64_t y_le[4])
{
    BN_CTX* ctx = BN_CTX_new();
    BIGNUM *p = nullptr, *x = BN_new(), *y_sq = BN_new(), *y = BN_new();
    BIGNUM *exp = BN_new(), *three = BN_new(), *seven = BN_new();
    bool ok = ctx && x && y_sq && y && exp && three && seven;

    if (ok) {
        BN_hex2bn(&p, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
        uint8_t xb[32];
        le4_to_be32(x_le, xb);
        BN_bin2bn(xb, 32, x);
        BN_set_word(three, 3);
        BN_set_word(seven, 7);

        // y = sqrt(x^3 + 7) mod p. secp256k1 p % 4 == 3, so sqrt = a^((p+1)/4).
        BN_mod_exp(y_sq, x, three, p, ctx);
        BN_mod_add(y_sq, y_sq, seven, p, ctx);
        BN_copy(exp, p);
        BN_add_word(exp, 1);
        BN_rshift(exp, exp, 2);
        BN_mod_exp(y, y_sq, exp, p, ctx);

        if ((uint8_t)(BN_is_odd(y) ? 1 : 0) != y_parity)
            BN_sub(y, p, y);

        uint8_t yb[32];
        BN_bn2binpad(y, yb, 32);
        be32_to_le4(yb, y_le);
    }

    BN_free(seven);
    BN_free(three);
    BN_free(exp);
    BN_free(y);
    BN_free(y_sq);
    BN_free(x);
    BN_free(p);
    BN_CTX_free(ctx);
    return ok;
}

static bool bsgs_pow_u64(uint32_t radix, uint32_t exp, uint64_t& out)
{
    out = 1;
    for (uint32_t i = 0; i < exp; i++) {
        if (out > UINT64_MAX / radix) return false;
        out *= radix;
    }
    return true;
}

static constexpr uint32_t BSGS_HEX_MAX_SIDE_UNKN = 15;

static bool parse_wif_mask(const std::string &wif_str, WifMask &mask, bool verbose = true);
static uint64_t bsgs_estimate_bloom_bucketed_bytes(uint64_t baby_count);
static uint64_t bsgs_estimate_hex8_bucketed_bytes(uint64_t baby_count);

static void bsgs_set_hex_nibble_weight(int nibble_pos, uint64_t scalar_weight[4])
{
    uint8_t be[32] = {};
    const int byte_idx = nibble_pos / 2;
    const int shift = (nibble_pos % 2 == 0) ? 4 : 0;
    be[byte_idx] = (uint8_t)(1u << shift);
    be32_to_le4(be, scalar_weight);
}

static void bsgs_mul_le4_small(uint64_t v[4], uint32_t m)
{
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        const uint64_t lo = (uint32_t)v[i];
        const uint64_t hi = v[i] >> 32;
        const uint64_t p0 = lo * (uint64_t)m + carry;
        const uint64_t p1 = hi * (uint64_t)m + (p0 >> 32);
        v[i] = (p0 & 0xffffffffULL) | (p1 << 32);
        carry = p1 >> 32;
    }
}

static bool bsgs_product_radices_u64(const BsgsUnknown* unknowns, uint32_t unknown_count, uint64_t& out)
{
    out = 1;
    for (uint32_t i = 0; i < unknown_count; i++) {
        const uint32_t radix = unknowns[i].radix;
        if (radix == 0 || out > std::numeric_limits<uint64_t>::max() / radix)
            return false;
        out *= radix;
    }
    return true;
}

static bool bsgs_mul_be_bytes_small(
    const uint8_t in[WIF_MAX_BYTES],
    int len,
    uint32_t multiplier,
    uint8_t out[WIF_MAX_BYTES])
{
    memset(out, 0, WIF_MAX_BYTES);
    uint32_t carry = 0;
    for (int i = len - 1; i >= 0; i--) {
        const uint32_t v = (uint32_t)in[i] * multiplier + carry;
        out[i] = (uint8_t)(v & 0xFFu);
        carry = v >> 8;
    }
    return carry == 0;
}

static bool parse_bsgs_baby_override(
    const std::vector<std::string>& opts,
    bool wif_mode,
    BsgsBabyOverride& out)
{
    out = BsgsBabyOverride::AUTO;
    for (size_t i = 0; i < opts.size(); ++i) {
        const std::string& opt = opts[i];
        if (opt != "--baby" && opt.rfind("--baby=", 0) != 0) {
            std::cerr << "Error: unsupported BSGS option '" << opt << "'.\n";
            return false;
        }

        std::string value;
        if (opt == "--baby") {
            if (++i >= opts.size()) {
                std::cerr << "Error: --baby expects a split value.\n";
                return false;
            }
            value = opts[i];
        } else {
            value = opt.substr(7);
        }

        for (char& c : value) c = (char)std::tolower((unsigned char)c);
        if (!wif_mode) {
            if (value == "7.5-vram" || value == "baby7.5-vram") out = BsgsBabyOverride::HEX_BABY75_VRAM;
            else if (value == "8" || value == "8.0" || value == "baby8") out = BsgsBabyOverride::HEX_BABY8;
            else {
                std::cerr << "Error: HEX BSGS supports --baby=7.5-vram or --baby=8.\n";
                return false;
            }
        } else {
            if (value == "5-vram" || value == "baby5-vram") out = BsgsBabyOverride::WIF_BABY5_VRAM;
            else if (value == "5.5" || value == "baby5.5") out = BsgsBabyOverride::WIF_BABY55;
            else {
                std::cerr << "Error: WIF BSGS supports --baby=5-vram or --baby=5.5.\n";
                return false;
            }
        }
    }
    return true;
}

static void bsgs_set_hex_unknown(
    BsgsUnknown& u,
    int nibble_pos,
    uint32_t group_pos,
    uint32_t radix,
    uint32_t digit_shift)
{
    memset(&u, 0, sizeof(u));
    u.pos = (uint16_t)nibble_pos;
    u.group_pos = (uint16_t)group_pos;
    u.radix = (uint16_t)radix;
    u._pad = (uint16_t)digit_shift;
    bsgs_set_hex_nibble_weight(nibble_pos, u.scalar_weight);
    if (digit_shift != 0)
        bsgs_mul_le4_small(u.scalar_weight, 1u << digit_shift);
}

static BIGNUM* bsgs_le4_to_bn(const uint64_t le[4])
{
    uint8_t be[32];
    le4_to_be32(le, be);
    return BN_bin2bn(be, 32, nullptr);
}

static bool bsgs_negate_y_cpu(uint64_t y_le[4])
{
    BN_CTX* ctx = BN_CTX_new();
    BIGNUM* p = nullptr;
    BIGNUM* y = bsgs_le4_to_bn(y_le);
    bool ok = ctx && y;
    if (ok) {
        BN_hex2bn(&p, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
        ok = p && (BN_mod_sub(y, p, y, p, ctx) == 1);
    }
    if (ok) {
        uint8_t be[32];
        BN_bn2binpad(y, be, 32);
        be32_to_le4(be, y_le);
    }
    BN_free(y);
    BN_free(p);
    BN_CTX_free(ctx);
    return ok;
}

static bool bsgs_build_contrib_points_cpu(
    const BsgsUnknown* unknowns,
    uint32_t unknown_count,
    uint32_t radix,
    uint32_t wif_shift,
    bool negate_y,
    BsgsPoint contrib[BSGS_MAX_UNKN][BSGS_MAX_RADIX])
{
    if (radix > BSGS_MAX_RADIX) return false;

    BN_CTX* ctx = BN_CTX_new();
    BIGNUM* order = nullptr;
    BIGNUM* value = BN_new();
    BIGNUM* scalar = BN_new();
    BIGNUM* product = BN_new();
    bool ok = ctx && value && scalar && product;
    if (ok) ok = (BN_hex2bn(&order, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141") != 0);

    for (uint32_t i = 0; ok && i < unknown_count; i++) {
        BIGNUM* weight = bsgs_le4_to_bn(unknowns[i].scalar_weight);
        ok = (weight != nullptr);
        for (uint32_t digit = 0; ok && digit < BSGS_MAX_RADIX; digit++) {
            BsgsPoint& out = contrib[i][digit];
            memset(&out, 0, sizeof(out));

            const uint32_t group_radix = unknowns[i].radix ? unknowns[i].radix : radix;
            if (digit >= group_radix || digit == 0) {
                out.flags = BSGS_POINT_INFINITY;
                continue;
            }

            BN_set_word(value, digit);
            if (wif_shift != 0) {
                ok = (BN_mul(product, weight, value, ctx) == 1);
                const uint64_t digit_low = (uint64_t)digit * unknowns[i].wif_low_weight;
                const uint64_t digit_carry = digit_low >> wif_shift;
                if (digit_carry != 0)
                    ok = ok && (BN_add_word(product, (BN_ULONG)digit_carry) == 1);
                if (ok) ok = (BN_mod(scalar, product, order, ctx) == 1);
            } else {
                ok = (BN_mod_mul(scalar, weight, value, order, ctx) == 1);
            }
            if (!ok) break;

            if (BN_is_zero(scalar)) {
                out.flags = BSGS_POINT_INFINITY;
                continue;
            }

            uint8_t scalar_be[32] = {};
            BN_bn2binpad(scalar, scalar_be, 32);
            ok = ec_mul_G(scalar_be, out.x, out.y);
            if (!ok) break;

            if (negate_y)
                ok = bsgs_negate_y_cpu(out.y);
            out.flags = 0;
        }
        BN_free(weight);
    }

    BN_free(product);
    BN_free(scalar);
    BN_free(value);
    BN_free(order);
    BN_CTX_free(ctx);
    return ok;
}

static bool bsgs_build_low_dict_cpu(
    BsgsPlan& plan,
    bool giant)
{
    const uint32_t unknown_count = giant ? plan.giant_unknown : plan.baby_unknown;
    const uint64_t total_count = giant ? plan.giant_count : plan.baby_count;
    uint32_t& low_bits = giant ? plan.giant_low_bits : plan.baby_low_bits;
    uint32_t& high_bits = giant ? plan.giant_high_bits : plan.baby_high_bits;
    uint64_t& high_count = giant ? plan.giant_high_count : plan.baby_high_count;
    BsgsPoint (*contrib)[BSGS_MAX_RADIX] = giant ? plan.giant_contrib : plan.baby_contrib;
    BsgsPoint* low_dict = giant ? plan.giant_low_dict : plan.baby_low_dict;

    const uint32_t total_bits = bsgs_side_bit_width(giant ? plan.giant : plan.baby, unknown_count);
    low_bits = std::min<uint32_t>(LOW_BITS, total_bits);
    high_bits = total_bits - low_bits;
    high_count = total_count >> low_bits;

    BN_CTX* ctx = BN_CTX_new();
    EC_GROUP* group = EC_GROUP_new_by_curve_name(NID_secp256k1);
    EC_POINT* acc = group ? EC_POINT_new(group) : nullptr;
    bool ok = ctx && group && acc;

    auto set_point = [&](EC_POINT* dst, const BsgsPoint& src) -> bool {
        if (src.flags & BSGS_POINT_INFINITY)
            return EC_POINT_set_to_infinity(group, dst) == 1;
        BIGNUM* x = bsgs_le4_to_bn(src.x);
        BIGNUM* y = bsgs_le4_to_bn(src.y);
        bool ret = x && y && (EC_POINT_set_affine_coordinates(group, dst, x, y, ctx) == 1);
        BN_free(x);
        BN_free(y);
        return ret;
    };

    auto store_point = [&](const EC_POINT* src, BsgsPoint& dst) -> bool {
        memset(&dst, 0, sizeof(dst));
        if (EC_POINT_is_at_infinity(group, src)) {
            dst.flags = BSGS_POINT_INFINITY;
            return true;
        }
        BIGNUM* x = BN_new();
        BIGNUM* y = BN_new();
        bool ret = x && y && (EC_POINT_get_affine_coordinates(group, src, x, y, ctx) == 1);
        if (ret) {
            uint8_t be[32];
            BN_bn2binpad(x, be, 32);
            be32_to_le4(be, dst.x);
            BN_bn2binpad(y, be, 32);
            be32_to_le4(be, dst.y);
            dst.flags = 0;
        }
        BN_free(x);
        BN_free(y);
        return ret;
    };

    const uint64_t dict_size = 1ULL << low_bits;
    for (uint64_t k = 0; ok && k < LOW_SIZE; k++) {
        if (k >= dict_size) {
            memset(&low_dict[k], 0, sizeof(low_dict[k]));
            low_dict[k].flags = BSGS_POINT_INFINITY;
            continue;
        }

        ok = (EC_POINT_set_to_infinity(group, acc) == 1);
        for (uint32_t bit = 0; ok && bit < low_bits; bit++) {
            if (((k >> bit) & 1ULL) == 0) continue;
            uint32_t group_pos = 0;
            uint32_t digit = 0;
            if (!bsgs_bit_to_group_digit(giant ? plan.giant : plan.baby, unknown_count, bit, group_pos, digit)) continue;
            EC_POINT* q = EC_POINT_new(group);
            ok = q && set_point(q, contrib[group_pos][digit]) &&
                 (EC_POINT_add(group, acc, acc, q, ctx) == 1);
            EC_POINT_free(q);
        }
        if (ok) ok = store_point(acc, low_dict[k]);
    }

    EC_POINT_free(acc);
    EC_GROUP_free(group);
    BN_CTX_free(ctx);
    return ok;
}

static bool build_bsgs_hex_plan_cpu(
    const std::string& mask_str,
    const TargetData& target,
    uint32_t requested_baby_unknown,
    BsgsLookupBackend lookup_backend,
    BsgsPlan& plan,
    bool baby75_split = false)
{
    if (target.type != TargetType::BTC_PUBKEY && target.type != TargetType::ETH_PUBKEY) {
        std::cerr << "Error: HEX BSGS requires a known secp256k1 pubkey target.\n";
        return false;
    }

    std::string s = mask_str;
    if (s.size() >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s = s.substr(2);
    if (s.size() != 64) {
        std::cerr << "Error: BSGS HEX mask must be exactly 64 hex/# chars.\n";
        return false;
    }

    uint8_t k_base_be[32] = {};
    std::vector<int> unknown_nibbles;
    unknown_nibbles.reserve(64);

    auto hex_val = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };

    for (int n = 0; n < 64; n++) {
        const char c = s[n];
        const int byte_idx = n / 2;
        const int shift = (n % 2 == 0) ? 4 : 0;
        if (c == '#') {
            unknown_nibbles.push_back(n);
            continue;
        }
        const int v = hex_val(c);
        if (v < 0) {
            std::cerr << "Error: invalid HEX BSGS mask character '" << c << "'.\n";
            return false;
        }
        k_base_be[byte_idx] |= (uint8_t)(v << shift);
    }

    const uint32_t total_unknown = (uint32_t)unknown_nibbles.size();
    if (total_unknown == 0) {
        std::cerr << "Error: no unknown nibble (#) in HEX BSGS mask.\n";
        return false;
    }
    if (total_unknown > BSGS_MAX_UNKN) {
        std::cerr << "Error: too many BSGS unknown nibbles (" << total_unknown
                  << " > " << BSGS_MAX_UNKN << ").\n";
        return false;
    }
    if (total_unknown > BSGS_HEX_MAX_SIDE_UNKN * 2u) {
        std::cerr << "Error: HEX BSGS currently supports up to "
                  << (BSGS_HEX_MAX_SIDE_UNKN * 2u)
                  << " unknown nibbles total with uint64_t baby/giant counters.\n";
        return false;
    }

    uint32_t baby_unknown = requested_baby_unknown;
    if (baby_unknown == 0) baby_unknown = total_unknown / 2;
    if (baby75_split) baby_unknown = 8u;
    if (baby_unknown == 0 || baby_unknown >= total_unknown) {
        std::cerr << "Error: invalid BSGS baby split (" << baby_unknown
                  << " / " << total_unknown << ").\n";
        return false;
    }
    uint32_t giant_unknown = total_unknown - baby_unknown;
    if (baby75_split) {
        if (total_unknown <= 8u) {
            std::cerr << "Error: HEX baby7.5 split requires more than 8 unknown nibbles.\n";
            return false;
        }
        giant_unknown = total_unknown - 7u; // high two bits of shared nibble + remaining nibbles
    }
    if (baby_unknown > BSGS_HEX_MAX_SIDE_UNKN ||
        giant_unknown > BSGS_HEX_MAX_SIDE_UNKN) {
        std::cerr << "Error: invalid BSGS split (" << baby_unknown
                  << " / " << giant_unknown << "). Each HEX side must be <= "
                  << BSGS_HEX_MAX_SIDE_UNKN << " nibbles for uint64_t counters.\n";
        return false;
    }

    memset(&plan, 0, sizeof(plan));
    plan.input_kind = BsgsInputKind::HEX;
    plan.lookup_backend = lookup_backend;
    plan.flags = bsgs_flag(BsgsRunFlags::GIANT_NEGATIVE);
    plan.radix = 16;
    plan.total_unknown = total_unknown;
    plan.baby_unknown = baby_unknown;
    plan.giant_unknown = giant_unknown;
    plan.wif_shift = 0;

    be32_to_le4(k_base_be, plan.k_base);

    uint64_t p_base_x[4] = {}, p_base_y[4] = {};
    bool p_base_inf = false;
    bool k_base_zero = true;
    for (uint8_t b : k_base_be) k_base_zero = k_base_zero && (b == 0);
    if (k_base_zero) {
        p_base_inf = true;
    } else if (!ec_mul_G(k_base_be, p_base_x, p_base_y)) {
        std::cerr << "Error: BSGS k_base*G computation failed.\n";
        return false;
    }

    bool p_start_inf = false;
    if (!ec_point_sub_affine(
            target.pubkey_x, target.pubkey_y, false,
            p_base_x, p_base_y, p_base_inf,
            plan.p_start_x, plan.p_start_y, p_start_inf) || p_start_inf) {
        std::cerr << "Error: BSGS P_start computation failed.\n";
        return false;
    }

    if (baby75_split) {
        for (uint32_t i = 0; i < 7u; i++)
            bsgs_set_hex_unknown(plan.baby[i], unknown_nibbles[i], i, 16u, 0u);
        bsgs_set_hex_unknown(plan.baby[7], unknown_nibbles[7], 7u, 4u, 0u);

        bsgs_set_hex_unknown(plan.giant[0], unknown_nibbles[7], 0u, 4u, 2u);
        for (uint32_t i = 1; i < giant_unknown; i++)
            bsgs_set_hex_unknown(plan.giant[i], unknown_nibbles[7u + i], i, 16u, 0u);
    } else {
        for (uint32_t i = 0; i < baby_unknown; i++)
            bsgs_set_hex_unknown(plan.baby[i], unknown_nibbles[i], i, 16u, 0u);
        for (uint32_t i = 0; i < giant_unknown; i++)
            bsgs_set_hex_unknown(plan.giant[i], unknown_nibbles[baby_unknown + i], i, 16u, 0u);
    }

    if (!bsgs_product_radices_u64(plan.baby, plan.baby_unknown, plan.baby_count) ||
        !bsgs_product_radices_u64(plan.giant, plan.giant_unknown, plan.giant_count)) {
        std::cerr << "Error: BSGS split count overflow.\n";
        return false;
    }

    if (!bsgs_build_contrib_points_cpu(
            plan.baby, plan.baby_unknown, plan.radix, plan.wif_shift, false, plan.baby_contrib)) {
        std::cerr << "Error: BSGS baby contribution point generation failed.\n";
        return false;
    }
    if (!bsgs_build_contrib_points_cpu(
            plan.giant, plan.giant_unknown, plan.radix, plan.wif_shift, true, plan.giant_contrib)) {
        std::cerr << "Error: BSGS giant contribution point generation failed.\n";
        return false;
    }
    if (!bsgs_build_low_dict_cpu(plan, false)) {
        std::cerr << "Error: BSGS baby low dictionary generation failed.\n";
        return false;
    }
    if (!bsgs_build_low_dict_cpu(plan, true)) {
        std::cerr << "Error: BSGS giant low dictionary generation failed.\n";
        return false;
    }

    return true;
}

static bool bsgs_wif_precompute_base_weights_cpu(
    const WifMask& mask,
    uint8_t base_bytes[WIF_MAX_BYTES],
    uint8_t weights[WIF_MAX_UNKN][WIF_MAX_BYTES])
{
    memset(base_bytes, 0, WIF_MAX_BYTES);
    memset(weights, 0, WIF_MAX_UNKN * WIF_MAX_BYTES);

    uint8_t b58_base[WIF_MAX_LEN] = {};
    for (int i = 0; i < mask.num_chars; i++)
        b58_base[i] = (mask.known_b58[i] == 0xFF) ? 0 : mask.known_b58[i];

    uint32_t tmp[WIF_MAX_BYTES] = {};
    for (int i = 0; i < mask.num_chars; i++) {
        uint32_t carry = b58_base[i];
        for (int j = mask.decoded_bytes - 1; j >= 0; j--) {
            carry += 58u * tmp[j];
            tmp[j] = carry & 0xFF;
            carry >>= 8;
        }
        if (carry != 0) return false;
    }
    for (int i = 0; i < mask.decoded_bytes; i++)
        base_bytes[i] = (uint8_t)tmp[i];

    for (int x = 0; x < mask.num_unknown; x++) {
        uint32_t tmp2[WIF_MAX_BYTES] = {};
        uint8_t b58_one[WIF_MAX_LEN] = {};
        b58_one[mask.unknown_pos[x]] = 1;
        for (int i = 0; i < mask.num_chars; i++) {
            uint32_t carry = b58_one[i];
            for (int j = mask.decoded_bytes - 1; j >= 0; j--) {
                carry += 58u * tmp2[j];
                tmp2[j] = carry & 0xFF;
                carry >>= 8;
            }
            if (carry != 0) return false;
        }
        for (int i = 0; i < mask.decoded_bytes; i++)
            weights[x][i] = (uint8_t)tmp2[i];
    }
    return true;
}

static uint64_t bsgs_wif_low_bits_from_be(const uint8_t bytes[WIF_MAX_BYTES], uint32_t len, uint32_t shift)
{
    const uint32_t low_bytes = shift / 8u;
    uint64_t v = 0;
    for (uint32_t i = 0; i < low_bytes; i++) {
        v = (v << 8) | bytes[len - low_bytes + i];
    }
    return v;
}

static bool bsgs_wif_project_scalar_weight(
    const uint8_t bytes[WIF_MAX_BYTES],
    uint32_t len,
    uint32_t shift,
    uint64_t scalar_le[4])
{
    BN_CTX* ctx = BN_CTX_new();
    BIGNUM* w = BN_bin2bn(bytes, len, nullptr);
    BIGNUM* high = BN_new();
    BIGNUM* order = nullptr;
    BIGNUM* reduced = BN_new();
    bool ok = ctx && w && high && reduced;
    if (ok) ok = (BN_hex2bn(&order, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141") != 0);
    if (ok) ok = (BN_rshift(high, w, (int)shift) == 1);
    if (ok) ok = (BN_nnmod(reduced, high, order, ctx) == 1);
    if (ok) {
        uint8_t be[32] = {};
        BN_bn2binpad(reduced, be, 32);
        be32_to_le4(be, scalar_le);
    }
    BN_free(reduced);
    BN_free(order);
    BN_free(high);
    BN_free(w);
    BN_CTX_free(ctx);
    return ok;
}

static bool bsgs_compute_wif_top_carry_point_cpu(BsgsPoint& out)
{
    memset(&out, 0, sizeof(out));

    BN_CTX* ctx = BN_CTX_new();
    BIGNUM* n = nullptr;
    BIGNUM* two256 = BN_new();
    BIGNUM* reduced = BN_new();
    bool ok = ctx && two256 && reduced;
    if (ok) ok = (BN_hex2bn(&n, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141") != 0);
    if (ok) ok = (BN_set_bit(two256, 256) == 1);
    if (ok) ok = (BN_mod(reduced, two256, n, ctx) == 1);
    if (ok) {
        uint8_t scalar_be[32] = {};
        BN_bn2binpad(reduced, scalar_be, 32);
        ok = ec_mul_G(scalar_be, out.x, out.y);
        out.flags = ok ? 0 : BSGS_POINT_INFINITY;
    }

    BN_free(reduced);
    BN_free(two256);
    BN_free(n);
    BN_CTX_free(ctx);
    return ok;
}

static bool bsgs_compute_wif_version_prefix_point_cpu(BsgsPoint& out)
{
    memset(&out, 0, sizeof(out));

    BN_CTX* ctx = BN_CTX_new();
    BIGNUM* n = nullptr;
    BIGNUM* scalar = BN_new();
    bool ok = ctx && scalar;
    if (ok) ok = (BN_hex2bn(&n, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141") != 0);
    if (ok) ok = (BN_set_bit(scalar, 256) == 1);
    if (ok) ok = (BN_mul_word(scalar, 0x80u) == 1);
    if (ok) ok = (BN_mod(scalar, scalar, n, ctx) == 1);
    if (ok) {
        uint8_t scalar_be[32] = {};
        BN_bn2binpad(scalar, scalar_be, 32);
        ok = ec_mul_G(scalar_be, out.x, out.y);
        out.flags = ok ? 0 : BSGS_POINT_INFINITY;
    }

    BN_free(scalar);
    BN_free(n);
    BN_CTX_free(ctx);
    return ok;
}

static uint32_t bsgs_count_wif_unknowns(const std::string& wif_str)
{
    uint32_t n = 0;
    for (char c : wif_str) if (c == '#') n++;
    return n;
}

static uint32_t choose_bsgs_wif_baby_split(
    uint32_t total_unknown,
    uint64_t vram_budget_bytes,
    bool have_vram_budget)
{
    if (total_unknown <= 1) return 0;

    // 58^11 overflows uint64_t, so keep both counters <= 58^10.
    const uint32_t max_side = 10;
    const uint32_t min_baby = (total_unknown > max_side) ? (total_unknown - max_side) : 1u;
    const uint32_t max_baby = std::min<uint32_t>(max_side, total_unknown - 1u);
    if (min_baby > max_baby) return 0;

    if (!have_vram_budget) {
        uint32_t conservative = std::min<uint32_t>(4, total_unknown / 2);
        if (conservative < min_baby) conservative = min_baby;
        if (conservative > max_baby) conservative = max_baby;
        return conservative;
    }

    uint32_t best = 0;
    for (uint32_t baby = min_baby; baby <= max_baby; baby++) {
        uint64_t baby_count = 0;
        if (!bsgs_pow_u64(58, baby, baby_count)) break;
        const uint64_t need = bsgs_estimate_bloom_bucketed_bytes(baby_count);
        if (need <= vram_budget_bytes) best = baby;
        else break;
    }
    return best ? best : min_baby;
}

static bool build_bsgs_wif_plan_cpu(
    const std::string& wif_str,
    const TargetData& target,
    uint32_t requested_baby_unknown,
    BsgsLookupBackend lookup_backend,
    BsgsPlan& plan,
    bool auto_rebalance_split = true,
    bool baby55_split = false)
{
    if (target.type != TargetType::BTC_PUBKEY && target.type != TargetType::ETH_PUBKEY) {
        std::cerr << "Error: WIF BSGS requires a known secp256k1 pubkey target.\n";
        return false;
    }

    WifMask mask = {};
    if (!parse_wif_mask(wif_str, mask, false)) return false;

    const uint32_t total_unknown = mask.num_unknown;
    if (baby55_split && total_unknown < 7u) {
        std::cerr << "Error: WIF baby5.5 requires at least 7 unknown characters.\n";
        return false;
    }
    if (baby55_split && total_unknown + 1u > BSGS_MAX_UNKN) {
        std::cerr << "Error: WIF baby5.5 needs one extra split group ("
                  << (total_unknown + 1u) << " > " << BSGS_MAX_UNKN << ").\n";
        return false;
    }

    const bool auto_baby_split = !baby55_split && (auto_rebalance_split || requested_baby_unknown == 0);
    uint32_t baby_unknown = requested_baby_unknown;
    if (baby55_split)
        baby_unknown = 6u; // 5 full Base58 chars + low 3 bits of the shared char
    else if (baby_unknown == 0)
        baby_unknown = choose_bsgs_wif_baby_split(total_unknown, 0, false);
    const uint32_t initial_baby_unknown = baby_unknown;
    const uint32_t plan_group_count = baby55_split ? (total_unknown + 1u) : total_unknown;
    if (baby_unknown == 0 || baby_unknown >= plan_group_count) {
        std::cerr << "Error: invalid WIF BSGS baby split (" << baby_unknown
                  << " / " << plan_group_count << ").\n";
        return false;
    }

    uint8_t base_bytes[WIF_MAX_BYTES] = {};
    uint8_t weights[WIF_MAX_UNKN][WIF_MAX_BYTES] = {};
    if (!bsgs_wif_precompute_base_weights_cpu(mask, base_bytes, weights)) {
        std::cerr << "Error: WIF BSGS Base58 precomputation failed.\n";
        return false;
    }

    const uint32_t wif_shift = mask.is_compressed ? 40u : 32u;
    std::vector<uint32_t> unknown_order(total_unknown);
    std::vector<uint8_t> scalar_affects_key(total_unknown);
    std::vector<uint8_t> checksum_tail_unknown(total_unknown);
    uint32_t scalar_unknown_count = 0;
    uint32_t non_tail_unknown_count = 0;
    for (uint32_t i = 0; i < total_unknown; i++) {
        unknown_order[i] = i;
        const bool tail = mask.unknown_pos[i] >= (uint8_t)((mask.num_chars > 4) ? (mask.num_chars - 4) : 0);
        checksum_tail_unknown[i] = tail ? 1u : 0u;
        if (!tail) non_tail_unknown_count++;
        uint64_t projected[4] = {};
        if (!bsgs_wif_project_scalar_weight(weights[i], mask.decoded_bytes, wif_shift, projected)) {
            std::cerr << "Error: WIF BSGS scalar projection classification failed.\n";
            return false;
        }
        const bool nonzero =
            projected[0] != 0 || projected[1] != 0 ||
            projected[2] != 0 || projected[3] != 0;
        scalar_affects_key[i] = nonzero ? 1u : 0u;
        if (nonzero) scalar_unknown_count++;
    }

    std::stable_sort(unknown_order.begin(), unknown_order.end(),
        [&](uint32_t a, uint32_t b) {
            if (checksum_tail_unknown[a] != checksum_tail_unknown[b])
                return checksum_tail_unknown[a] < checksum_tail_unknown[b];
            return scalar_affects_key[a] > scalar_affects_key[b];
        });

    if (auto_baby_split && non_tail_unknown_count > 0 && baby_unknown > non_tail_unknown_count) {
        const uint32_t min_baby = (total_unknown > 10u) ? (total_unknown - 10u) : 1u;
        baby_unknown = std::max<uint32_t>(min_baby, non_tail_unknown_count);
        if (baby_unknown >= total_unknown) baby_unknown = total_unknown - 1u;
    } else if (auto_baby_split && scalar_unknown_count > 0 && baby_unknown > scalar_unknown_count) {
        const uint32_t min_baby = (total_unknown > 10u) ? (total_unknown - 10u) : 1u;
        baby_unknown = std::max<uint32_t>(min_baby, scalar_unknown_count);
        if (baby_unknown >= total_unknown) baby_unknown = total_unknown - 1u;
    }
    const uint32_t giant_unknown = plan_group_count - baby_unknown;

    memset(&plan, 0, sizeof(plan));
    plan.input_kind = mask.is_compressed ? BsgsInputKind::WIF_COMPRESSED : BsgsInputKind::WIF_UNCOMPRESSED;
    plan.lookup_backend = lookup_backend;
    plan.flags = bsgs_flag(BsgsRunFlags::GIANT_NEGATIVE);
    plan.radix = 58;
    plan.total_unknown = total_unknown;
    plan.baby_unknown = baby_unknown;
    plan.giant_unknown = giant_unknown;
    plan.wif_shift = wif_shift;
    plan.wif_split_initial_baby = initial_baby_unknown;
    plan.wif_split_non_tail = non_tail_unknown_count;
    plan.wif_split_scalar = scalar_unknown_count;
    plan.wif_split_auto = auto_baby_split ? 1u : 0u;

    WifMask plan_mask = mask;
    uint8_t ordered_weights[BSGS_MAX_UNKN][WIF_MAX_BYTES] = {};
    uint8_t ordered_pos[BSGS_MAX_UNKN] = {};
    uint16_t ordered_multiplier[BSGS_MAX_UNKN] = {};

    if (baby55_split) {
        for (uint32_t i = 0; i < 5u; i++) {
            const uint32_t src = unknown_order[i];
            ordered_pos[i] = mask.unknown_pos[src];
            ordered_multiplier[i] = 1u;
            memcpy(ordered_weights[i], weights[src], WIF_MAX_BYTES);
        }

        const uint32_t split_src = unknown_order[5];
        ordered_pos[5] = mask.unknown_pos[split_src];
        ordered_multiplier[5] = 1u; // low digit: 0..7
        if (!bsgs_mul_be_bytes_small(weights[split_src], mask.decoded_bytes, 1u, ordered_weights[5]))
            return false;

        ordered_pos[6] = mask.unknown_pos[split_src];
        ordered_multiplier[6] = 8u; // high digit: real Base58 digit += 8 * high
        if (!bsgs_mul_be_bytes_small(weights[split_src], mask.decoded_bytes, 8u, ordered_weights[6]))
            return false;

        for (uint32_t i = 6u; i < total_unknown; i++) {
            const uint32_t dst = i + 1u;
            const uint32_t src = unknown_order[i];
            ordered_pos[dst] = mask.unknown_pos[src];
            ordered_multiplier[dst] = 1u;
            memcpy(ordered_weights[dst], weights[src], WIF_MAX_BYTES);
        }
    } else {
        for (uint32_t i = 0; i < total_unknown; i++) {
            const uint32_t src = unknown_order[i];
            ordered_pos[i] = mask.unknown_pos[src];
            ordered_multiplier[i] = 1u;
            memcpy(ordered_weights[i], weights[src], WIF_MAX_BYTES);
            plan_mask.unknown_pos[i] = mask.unknown_pos[src];
        }
    }

    plan.wif_mask = plan_mask;
    memcpy(plan.wif_base_bytes, base_bytes, sizeof(plan.wif_base_bytes));
    memcpy(plan.wif_weight_bytes, ordered_weights, sizeof(plan.wif_weight_bytes));

    if (!bsgs_wif_project_scalar_weight(base_bytes, mask.decoded_bytes, plan.wif_shift, plan.k_base)) {
        std::cerr << "Error: WIF BSGS k_base projection failed.\n";
        return false;
    }
    plan.wif_low_base[0] = bsgs_wif_low_bits_from_be(base_bytes, mask.decoded_bytes, plan.wif_shift);
    if (!bsgs_compute_wif_top_carry_point_cpu(plan.wif_top_carry_contrib)) {
        std::cerr << "Error: WIF BSGS top-carry point generation failed.\n";
        return false;
    }
    BsgsPoint wif_version_prefix = {};
    if (!bsgs_compute_wif_version_prefix_point_cpu(wif_version_prefix)) {
        std::cerr << "Error: WIF BSGS version-prefix point generation failed.\n";
        return false;
    }

    uint8_t k_base_be[32] = {};
    le4_to_be32(plan.k_base, k_base_be);
    uint64_t p_base_x[4] = {}, p_base_y[4] = {};
    bool p_base_inf = true;
    for (uint8_t b : k_base_be) {
        if (b != 0) {
            p_base_inf = false;
            break;
        }
    }
    if (!p_base_inf && !ec_mul_G(k_base_be, p_base_x, p_base_y)) {
        std::cerr << "Error: WIF BSGS k_base*G computation failed.\n";
        return false;
    }

    uint64_t target_adjusted_x[4] = {};
    uint64_t target_adjusted_y[4] = {};
    uint64_t prefix_neg_y[4] = {};
    memcpy(prefix_neg_y, wif_version_prefix.y, sizeof(prefix_neg_y));
    if (!bsgs_negate_y_cpu(prefix_neg_y)) {
        std::cerr << "Error: WIF BSGS version-prefix negation failed.\n";
        return false;
    }
    bool target_adjusted_inf = false;
    if (!ec_point_sub_affine(
            target.pubkey_x, target.pubkey_y, false,
            wif_version_prefix.x, prefix_neg_y, false,
            target_adjusted_x, target_adjusted_y, target_adjusted_inf) || target_adjusted_inf) {
        std::cerr << "Error: WIF BSGS target prefix adjustment failed.\n";
        return false;
    }

    bool p_start_inf = false;
    if (!ec_point_sub_affine(
            target_adjusted_x, target_adjusted_y, false,
            p_base_x, p_base_y, p_base_inf,
            plan.p_start_x, plan.p_start_y, p_start_inf) || p_start_inf) {
        std::cerr << "Error: WIF BSGS P_start computation failed.\n";
        return false;
    }

    for (uint32_t i = 0; i < baby_unknown; i++) {
        BsgsUnknown& u = plan.baby[i];
        u.pos = ordered_pos[i];
        u.group_pos = (uint16_t)i;
        u.radix = (baby55_split && i == 5u) ? 8u : 58u;
        u._pad = ordered_multiplier[i] ? ordered_multiplier[i] : 1u;
        u.wif_low_weight = bsgs_wif_low_bits_from_be(ordered_weights[i], mask.decoded_bytes, plan.wif_shift);
        if (!bsgs_wif_project_scalar_weight(ordered_weights[i], mask.decoded_bytes, plan.wif_shift, u.scalar_weight)) {
            std::cerr << "Error: WIF BSGS baby scalar projection failed.\n";
            return false;
        }
    }
    for (uint32_t i = 0; i < giant_unknown; i++) {
        const uint32_t src = baby_unknown + i;
        BsgsUnknown& u = plan.giant[i];
        u.pos = ordered_pos[src];
        u.group_pos = (uint16_t)i;
        u.radix = (baby55_split && src == 6u) ? 8u : 58u;
        u._pad = ordered_multiplier[src] ? ordered_multiplier[src] : 1u;
        u.wif_low_weight = bsgs_wif_low_bits_from_be(ordered_weights[src], mask.decoded_bytes, plan.wif_shift);
        if (!bsgs_wif_project_scalar_weight(ordered_weights[src], mask.decoded_bytes, plan.wif_shift, u.scalar_weight)) {
            std::cerr << "Error: WIF BSGS giant scalar projection failed.\n";
            return false;
        }
    }

    if (!bsgs_product_radices_u64(plan.baby, plan.baby_unknown, plan.baby_count) ||
        !bsgs_product_radices_u64(plan.giant, plan.giant_unknown, plan.giant_count)) {
        std::cerr << "Error: WIF BSGS split count overflow.\n";
        return false;
    }

    if (!bsgs_build_contrib_points_cpu(
            plan.baby, plan.baby_unknown, plan.radix, plan.wif_shift, false, plan.baby_contrib)) {
        std::cerr << "Error: WIF BSGS baby contribution point generation failed.\n";
        return false;
    }
    if (!bsgs_build_contrib_points_cpu(
            plan.giant, plan.giant_unknown, plan.radix, plan.wif_shift, true, plan.giant_contrib)) {
        std::cerr << "Error: WIF BSGS giant contribution point generation failed.\n";
        return false;
    }

    return true;
}

static void print_bsgs_plan(const BsgsPlan& plan)
{
    auto backend_name = [](BsgsLookupBackend backend) -> const char* {
        switch (backend) {
            case BsgsLookupBackend::BLOOM_BUCKETED: return "Production";
        }
        return "Unknown";
    };

    auto print_point = [](const char* label, const uint64_t v[4]) {
        std::cout << label << std::hex << std::setfill('0');
        for (int i = 3; i >= 0; i--) std::cout << std::setw(16) << v[i];
        std::cout << std::dec << std::setfill(' ') << "\n";
    };

    auto input_name = [](BsgsInputKind kind) -> const char* {
        switch (kind) {
            case BsgsInputKind::HEX: return "HEX";
            case BsgsInputKind::WIF_COMPRESSED: return "WIF compressed";
            case BsgsInputKind::WIF_UNCOMPRESSED: return "WIF uncompressed";
        }
        return "Unknown";
    };

    const bool is_wif = (plan.input_kind == BsgsInputKind::WIF_COMPRESSED ||
                         plan.input_kind == BsgsInputKind::WIF_UNCOMPRESSED);
    const char* unit_label = is_wif ? "chars" : "nibbles";

    std::cout << "======== HYDRA BSGS PLAN (" << input_name(plan.input_kind)
              << " / " << backend_name(plan.lookup_backend) << ") ========\n";
    std::cout << "Input        : " << input_name(plan.input_kind) << " + PubKey\n";
    std::cout << "Radix        : " << plan.radix << "\n";
    std::cout << "Unknowns     : " << plan.total_unknown
              << " (baby=" << plan.baby_unknown
              << ", giant=" << plan.giant_unknown << ")\n";
    if (is_wif && plan.baby_unknown + plan.giant_unknown != plan.total_unknown) {
        std::cout << "Split groups : " << (plan.baby_unknown + plan.giant_unknown)
                  << " (one Base58 char split across baby/giant)\n";
    }
    std::cout << "Baby count   : " << plan.baby_count << "\n";
    std::cout << "Giant count  : " << plan.giant_count << "\n";
    std::cout << "Baby split   : high=" << plan.baby_high_bits
              << " bits, low=" << plan.baby_low_bits
              << " bits (" << (1ULL << plan.baby_low_bits) << " dict)\n";
    std::cout << "Giant split  : high=" << plan.giant_high_bits
              << " bits, low=" << plan.giant_low_bits
              << " bits (" << (1ULL << plan.giant_low_bits) << " dict)\n";
    std::cout << "Backend      : " << backend_name(plan.lookup_backend) << "\n";
    std::cout << "Flags        : 0x" << std::hex << plan.flags << std::dec << "\n";
    if (is_wif) {
        std::cout << "WIF shift    : " << plan.wif_shift << " bits\n";
        std::cout << "WIF low base : " << plan.wif_low_base[0] << "\n";
        std::cout << "WIF split    : scalar-first, checksum-tail giant\n";
        std::cout << "WIF carry    : exact low carry from baby residue + giant residue\n";
    }
    print_point("k_base      : ", plan.k_base);
    print_point("P_start.x   : ", plan.p_start_x);
    print_point("P_start.y   : ", plan.p_start_y);

    std::cout << "Baby " << unit_label << " :";
    for (uint32_t i = 0; i < plan.baby_unknown; i++) std::cout << " " << plan.baby[i].pos;
    std::cout << "\n";
    std::cout << "Giant " << unit_label << ":";
    for (uint32_t i = 0; i < plan.giant_unknown; i++) std::cout << " " << plan.giant[i].pos;
    std::cout << "\n";
    std::cout << "======================================================\n";
}

static uint64_t bsgs_bloom_bits_for_baby_count(uint64_t baby_count)
{
    uint64_t target = baby_count * 64ULL;
    const uint64_t min_bits = 1ULL << 20; // 128 KiB
    if (target < min_bits) target = min_bits;

    uint64_t bits = 1;
    while (bits < target && bits < (1ULL << 34)) bits <<= 1;
    return bits;
}

static uint32_t bsgs_bucket_bits_for_baby_count(uint64_t baby_count)
{
    // Target ~16 baby entries per bucket. This keeps exact scans short while
    // keeping offsets compact enough for production runs.
    uint64_t buckets = 1;
    uint32_t bits = 0;
    const uint64_t target = std::max<uint64_t>(1, (baby_count + 15ULL) / 16ULL);
    while (buckets < target && bits < 24) {
        buckets <<= 1;
        bits++;
    }
    return std::max<uint32_t>(8, bits);
}

static uint32_t bsgs_ram_bucket_bits_for_baby_count(uint64_t baby_count)
{
    // RAM-baby resolves only Bloom hits on CPU, so fatter buckets are a good
    // tradeoff: less cursor/offset pressure during scatter, still cheap scans.
    uint32_t bits = bsgs_bucket_bits_for_baby_count(baby_count);
    if (baby_count >= 500000000ULL && bits > 22u) bits = 22u;
    return bits;
}

static uint32_t bsgs_count_hex_unknowns(const std::string& mask_str)
{
    std::string s = mask_str;
    if (s.size() >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s = s.substr(2);
    uint32_t n = 0;
    for (char c : s) if (c == '#') n++;
    return n;
}

static uint64_t bsgs_estimate_bloom_bucketed_bytes(uint64_t baby_count)
{
    const uint64_t bloom_bits = bsgs_bloom_bits_for_baby_count(baby_count);
    const uint32_t bucket_bits = bsgs_bucket_bits_for_baby_count(baby_count);
    const uint64_t bucket_count = 1ULL << bucket_bits;

    uint64_t bytes = 0;
    bytes += baby_count * sizeof(BsgsBabyEntry); // raw baby entries
    bytes += baby_count * sizeof(BsgsBabyEntry); // bucketed baby entries
    bytes += bloom_bits / 8ULL;
    bytes += (bucket_count + 1ULL) * sizeof(uint32_t); // offsets
    bytes += bucket_count * sizeof(uint32_t);          // counts
    bytes += bucket_count * sizeof(uint32_t);          // cursor
    bytes += 16ULL * 1024ULL * 1024ULL;                // plan/hits/runtime margin
    return bytes;
}

static uint64_t bsgs_estimate_hex8_bucketed_bytes(uint64_t baby_count)
{
    const uint64_t bloom_bits = bsgs_bloom_bits_for_baby_count(baby_count);
    const uint32_t bucket_bits = bsgs_bucket_bits_for_baby_count(baby_count);
    const uint64_t bucket_count = 1ULL << bucket_bits;

    uint64_t bytes = 0;
    bytes += baby_count * sizeof(BsgsHex8Entry);       // compact bucketed baby entries
    bytes += bloom_bits / 8ULL;
    bytes += (bucket_count + 1ULL) * sizeof(uint32_t); // offsets
    bytes += bucket_count * sizeof(uint32_t);          // counts
    bytes += bucket_count * sizeof(uint32_t);          // cursor
    bytes += 16ULL * 1024ULL * 1024ULL;                // plan/hits/runtime margin
    return bytes;
}

static uint64_t bsgs_estimate_hex8_bucketed_bytes_with_bloom(
    uint64_t baby_count,
    uint64_t bloom_bits)
{
    const uint32_t bucket_bits = bsgs_bucket_bits_for_baby_count(baby_count);
    const uint64_t bucket_count = 1ULL << bucket_bits;

    uint64_t bytes = 0;
    bytes += baby_count * sizeof(BsgsHex8Entry);
    bytes += bloom_bits / 8ULL;
    bytes += (bucket_count + 1ULL) * sizeof(uint32_t);
    bytes += bucket_count * sizeof(uint32_t);
    bytes += bucket_count * sizeof(uint32_t);
    bytes += 16ULL * 1024ULL * 1024ULL;
    return bytes;
}

static uint64_t bsgs_hex_host_bloom_bits(uint64_t baby_count, uint64_t vram_budget);

static uint64_t bsgs_estimate_wif_ram_baby_host_bytes(uint64_t baby_count)
{
    const uint32_t bucket_bits = bsgs_ram_bucket_bits_for_baby_count(baby_count);
    const uint64_t bucket_count = 1ULL << bucket_bits;
    const uint64_t chunk_capacity = std::min<uint64_t>(1ULL << 22, baby_count);

    uint64_t bytes = 0;
    bytes += baby_count * 12ULL;                       // BsgsHostBaby12
    bytes += (bucket_count + 1ULL) * sizeof(uint32_t); // offsets
    bytes += bucket_count * sizeof(uint32_t);          // cursor
    bytes += chunk_capacity * sizeof(BsgsBabyEntry);   // temporary GPU->CPU chunk
    return bytes;
}

static uint64_t bsgs_estimate_hex_host_baby_vram_bytes(
    uint64_t baby_count,
    uint64_t vram_budget_bytes)
{
    const uint64_t bloom_bits = bsgs_hex_host_bloom_bits(baby_count, vram_budget_bytes);
    const uint64_t chunk_capacity = std::min<uint64_t>(1ULL << 22, baby_count);
    const uint64_t hit_capacity = 1ULL << 22;

    uint64_t bytes = 0;
    bytes += bloom_bits / 8ULL;
    bytes += chunk_capacity * sizeof(BsgsBabyEntry);
    bytes += hit_capacity * sizeof(BsgsHit);
    bytes += 64ULL * 1024ULL * 1024ULL;
    return bytes;
}

static uint64_t bsgs_estimate_hex_host_baby_host_bytes(uint64_t baby_count)
{
    const uint64_t bucket_count = 1ULL << 24;
    const uint64_t chunk_capacity = std::min<uint64_t>(1ULL << 22, baby_count);
    const uint64_t hex_host_entry_size = 6ULL;          // BsgsHostHex6

    uint64_t bytes = 0;
    bytes += (bucket_count + 1ULL) * sizeof(uint64_t);
    bytes += bucket_count * (sizeof(uint32_t) + sizeof(uint64_t));
    bytes += chunk_capacity * sizeof(BsgsBabyEntry);
    bytes += baby_count * hex_host_entry_size;
    return bytes;
}

static bool bsgs_host_ram_budget_allows(uint64_t needed_bytes)
{
    const uint64_t available = hydra_host_available_ram_bytes();
    const uint64_t total = hydra_host_ram_bytes();
    if (available == 0 && total == 0) return false;

    if (hydra_is_wsl() && needed_bytes >= (16ULL << 30))
        return false;

    // For medium host tables such as HEX Compact6 baby7.5, MemAvailable can
    // fluctuate after a previous large run because Linux/WSL retains cache.
    // Use installed/visible RAM with a conservative reserve to keep the
    // scheduler stable; allocation still fails cleanly if the host is actually
    // too busy. Larger Baby8-class paths are opt-in only.
    const uint64_t medium_table_limit = 12ULL << 30;
    const uint64_t base = (needed_bytes <= medium_table_limit && total != 0)
        ? total
        : (available ? available : total);
    const uint64_t reserve = std::max<uint64_t>(2ULL << 30, total / 10ULL);
    return base > needed_bytes + reserve;
}

static bool bsgs_should_auto_use_wif_ram_baby(
    uint32_t total_unknown,
    bool have_vram_budget,
    uint64_t vram_budget_bytes)
{
    // RAM baby=5 is a search-scale path. For 7-9 WIF unknowns, the VRAM
    // bucketed path is much faster because it avoids rebuilding 58^5 babies.
    if (total_unknown < 10)
        return false;

    uint64_t baby5_count = 0;
    if (!bsgs_pow_u64(58, 5, baby5_count))
        return false;

    const uint64_t baby5_vram_need = bsgs_estimate_bloom_bucketed_bytes(baby5_count);
    if (have_vram_budget && baby5_vram_need <= vram_budget_bytes)
        return false;

    const uint64_t baby5_host_need = bsgs_estimate_wif_ram_baby_host_bytes(baby5_count);
    const uint64_t host_ram = hydra_host_ram_bytes();
    if (host_ram == 0)
        return true;

    const uint64_t reserve = 512ULL * 1024ULL * 1024ULL;
    if (baby5_host_need + reserve <= host_ram)
        return true;

    return baby5_host_need <= (uint64_t)((double)host_ram * 0.95);
}

static uint32_t bsgs_cap_auto_wif_baby_split(uint32_t baby_unknown)
{
    // WIF baby5 VRAM is an experimental opt-in path. Auto may still choose the
    // tested RAM-baby fallback when the VRAM path does not fit.
    return std::min<uint32_t>(baby_unknown, 4u);
}

static uint32_t choose_bsgs_hex_baby_split(
    uint32_t total_unknown,
    uint64_t vram_budget_bytes,
    bool have_vram_budget)
{
    if (total_unknown <= 1) return 0;
    const uint32_t min_baby = (total_unknown > BSGS_HEX_MAX_SIDE_UNKN)
        ? (total_unknown - BSGS_HEX_MAX_SIDE_UNKN) : 1u;
    const uint32_t max_baby = std::min<uint32_t>(BSGS_HEX_MAX_SIDE_UNKN, total_unknown - 1u);
    if (min_baby > max_baby) return 0;

    uint32_t balanced = total_unknown / 2;
    if (balanced < min_baby) balanced = min_baby;
    if (balanced > max_baby) balanced = max_baby;
    if (!have_vram_budget) return balanced;

    const uint32_t target = std::min<uint32_t>(
        max_baby, std::max<uint32_t>(min_baby, (total_unknown + 1u) / 2u));

    for (uint32_t baby = target; baby >= min_baby; baby--) {
        uint64_t baby_count = 0;
        if (!bsgs_pow_u64(16, baby, baby_count)) break;
        const uint64_t need = (baby_count <= (uint64_t)UINT32_MAX + 1ULL)
            ? bsgs_estimate_hex8_bucketed_bytes(baby_count)
            : bsgs_estimate_bloom_bucketed_bytes(baby_count);
        if (need <= vram_budget_bytes) return baby;
        if (baby == 0) break;
    }
    return min_baby;
}

static std::string format_bytes_mb(uint64_t bytes)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1)
        << (double)bytes / 1024.0 / 1024.0 << " MB";
    return oss.str();
}

static void bsgs_apply_hex_digits_to_key(
    uint8_t key_be[32],
    const BsgsUnknown* unknowns,
    uint32_t unknown_count,
    uint64_t idx,
    uint32_t radix)
{
    for (int group = (int)unknown_count - 1; group >= 0; group--) {
        const uint32_t group_radix = unknowns[group].radix ? unknowns[group].radix : radix;
        const uint32_t digit = (uint32_t)(idx % group_radix);
        idx /= group_radix;
        const uint16_t nibble_pos = unknowns[group].pos;
        const int byte_idx = nibble_pos / 2;
        const int shift = (nibble_pos % 2 == 0) ? 4 : 0;
        const uint32_t digit_shift = unknowns[group]._pad;
        const uint32_t mask = (group_radix - 1u) << digit_shift;
        key_be[byte_idx] &= (uint8_t)~(mask << shift);
        key_be[byte_idx] |= (uint8_t)((digit << digit_shift) << shift);
    }
}

static std::string hex_key_from_be32(const uint8_t key_be[32])
{
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < 32; i++) oss << std::setw(2) << (int)key_be[i];
    return oss.str();
}

static bool bsgs_reconstruct_hex_key_cpu(
    const BsgsPlan& plan,
    uint64_t idx_a,
    uint64_t idx_b,
    uint8_t key_be[32])
{
    le4_to_be32(plan.k_base, key_be);
    bsgs_apply_hex_digits_to_key(key_be, plan.baby, plan.baby_unknown, idx_a, plan.radix);
    bsgs_apply_hex_digits_to_key(key_be, plan.giant, plan.giant_unknown, idx_b, plan.radix);
    return true;
}

static void bsgs_apply_wif_digits_to_b58(
    uint8_t b58[WIF_MAX_LEN],
    const BsgsUnknown* unknowns,
    uint32_t unknown_count,
    uint64_t idx,
    uint32_t radix)
{
    for (int group = (int)unknown_count - 1; group >= 0; group--) {
        const uint32_t group_radix = unknowns[group].radix ? unknowns[group].radix : radix;
        const uint32_t digit = (uint32_t)(idx % group_radix);
        idx /= group_radix;
        const uint32_t mul = unknowns[group]._pad ? unknowns[group]._pad : 1u;
        b58[unknowns[group].pos] = (uint8_t)(b58[unknowns[group].pos] + digit * mul);
    }
}

static bool bsgs_wif_digits_valid_b58(
    const WifMask& mask,
    const uint8_t b58[WIF_MAX_LEN])
{
    for (int i = 0; i < mask.num_unknown; i++) {
        const uint8_t pos = mask.unknown_pos[i];
        if (b58[pos] >= 58u) return false;
    }
    return true;
}

static int bsgs_wif_weight_index_for_pos(const WifMask& mask, uint16_t pos)
{
    for (int i = 0; i < mask.num_unknown; i++) {
        if (mask.unknown_pos[i] == pos) return i;
    }
    return -1;
}

static void bsgs_add_wif_weight_digit_cpu(
    uint8_t raw[WIF_MAX_BYTES],
    const uint8_t weight[WIF_MAX_BYTES],
    int len,
    uint32_t digit)
{
    uint32_t carry = 0;
    for (int i = len - 1; i >= 0; i--) {
        const uint32_t v = (uint32_t)raw[i] + digit * (uint32_t)weight[i] + carry;
        raw[i] = (uint8_t)(v & 0xFFu);
        carry = v >> 8;
    }
}

[[maybe_unused]] static bool bsgs_apply_wif_idx_to_raw_cpu(
    uint8_t raw[WIF_MAX_BYTES],
    const WifMask& mask,
    const uint8_t weights[WIF_MAX_UNKN][WIF_MAX_BYTES],
    const BsgsUnknown* unknowns,
    uint32_t unknown_count,
    uint64_t idx,
    uint32_t radix)
{
    for (int group = (int)unknown_count - 1; group >= 0; group--) {
        const uint32_t group_radix = unknowns[group].radix ? unknowns[group].radix : radix;
        const uint32_t digit = (uint32_t)(idx % group_radix);
        idx /= group_radix;
        if (digit == 0) continue;

        const int weight_idx = bsgs_wif_weight_index_for_pos(mask, unknowns[group].pos);
        if (weight_idx < 0) return false;
        bsgs_add_wif_weight_digit_cpu(raw, weights[weight_idx], mask.decoded_bytes, digit);
    }
    return true;
}

static bool bsgs_apply_wif_plan_idx_to_raw_cpu(
    uint8_t raw[WIF_MAX_BYTES],
    const WifMask& mask,
    const uint8_t weights[BSGS_MAX_UNKN][WIF_MAX_BYTES],
    const BsgsUnknown* unknowns,
    uint32_t unknown_count,
    uint64_t idx,
    uint32_t radix,
    uint32_t weight_base)
{
    for (int group = (int)unknown_count - 1; group >= 0; group--) {
        const uint32_t group_radix = unknowns[group].radix ? unknowns[group].radix : radix;
        const uint32_t digit = (uint32_t)(idx % group_radix);
        idx /= group_radix;
        if (digit == 0) continue;

        const uint32_t weight_idx = weight_base + (uint32_t)group;
        if (weight_idx >= BSGS_MAX_UNKN) return false;
        bsgs_add_wif_weight_digit_cpu(raw, weights[weight_idx], mask.decoded_bytes, digit);
    }
    return true;
}

static bool bsgs_verify_wif_raw_checksum_cpu(const WifMask& mask, const uint8_t raw[WIF_MAX_BYTES])
{
    if (raw[0] != 0x80) return false;
    if (mask.is_compressed && raw[33] != 0x01) return false;

    uint8_t h1[32], h2[32];
    sha256_cpu(raw, mask.payload_len, h1);
    sha256_cpu(h1, 32, h2);
    for (int i = 0; i < 4; i++) {
        if (raw[mask.checksum_offset + i] != h2[i]) return false;
    }
    return true;
}

static void bsgs_fill_wif_b58_digits_cpu(
    const WifMask& mask,
    const BsgsPlan& plan,
    uint64_t idx_a,
    uint64_t idx_b,
    uint8_t b58[WIF_MAX_LEN])
{
    for (int i = 0; i < mask.num_chars; i++)
        b58[i] = (mask.known_b58[i] == 0xFF) ? 0 : mask.known_b58[i];
    bsgs_apply_wif_digits_to_b58(b58, plan.baby, plan.baby_unknown, idx_a, plan.radix);
    bsgs_apply_wif_digits_to_b58(b58, plan.giant, plan.giant_unknown, idx_b, plan.radix);
}

static std::string bsgs_wif_string_from_digits(const uint8_t b58[WIF_MAX_LEN], uint32_t len)
{
    static const char* alpha = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    std::string out;
    out.reserve(len);
    for (uint32_t i = 0; i < len; i++)
        out.push_back(alpha[b58[i]]);
    return out;
}

static bool bsgs_reconstruct_wif_candidate_fast_cpu(
    const WifMask& mask,
    const uint8_t base_bytes[WIF_MAX_BYTES],
    const uint8_t weights[WIF_MAX_UNKN][WIF_MAX_BYTES],
    const BsgsPlan& plan,
    uint64_t idx_a,
    uint64_t idx_b,
    uint8_t key_be[32],
    std::string& out_wif)
{
    if ((plan.input_kind == BsgsInputKind::WIF_COMPRESSED) != (mask.is_compressed != 0))
        return false;

    uint8_t raw[WIF_MAX_BYTES] = {};
    memcpy(raw, base_bytes, WIF_MAX_BYTES);
    (void)weights;
    if (!bsgs_apply_wif_plan_idx_to_raw_cpu(
            raw, mask, plan.wif_weight_bytes, plan.baby, plan.baby_unknown, idx_a, plan.radix, 0)) {
        return false;
    }
    if (!bsgs_apply_wif_plan_idx_to_raw_cpu(
            raw, mask, plan.wif_weight_bytes, plan.giant, plan.giant_unknown, idx_b, plan.radix, plan.baby_unknown)) {
        return false;
    }
    if (!bsgs_verify_wif_raw_checksum_cpu(mask, raw)) return false;

    uint8_t b58[WIF_MAX_LEN] = {};
    bsgs_fill_wif_b58_digits_cpu(mask, plan, idx_a, idx_b, b58);
    if (!bsgs_wif_digits_valid_b58(mask, b58)) return false;
    memcpy(key_be, raw + WIF_KEY_OFFSET, 32);
    out_wif = bsgs_wif_string_from_digits(b58, mask.num_chars);
    return true;
}

static bool bsgs_verify_key_against_target_pubkey(
    const uint8_t key_be[32],
    const TargetData& target)
{
    uint64_t x[4], y[4];
    if (!ec_mul_G(key_be, x, y)) return false;

    return x[0] == target.pubkey_x[0] &&
           x[1] == target.pubkey_x[1] &&
           x[2] == target.pubkey_x[2] &&
           x[3] == target.pubkey_x[3] &&
           (uint8_t)(y[0] & 1ULL) == target.pubkey_y_parity;
}

#pragma pack(push, 1)
struct BsgsHostBaby12 {
    uint64_t fp_lo;
    uint32_t idx_a;
};
#pragma pack(pop)

static_assert(sizeof(BsgsHostBaby12) == 12, "BsgsHostBaby12 must remain 12 bytes");

#pragma pack(push, 1)
struct BsgsHostBaby13 {
    uint64_t fp_lo;
    uint8_t  idx_a40[5];
};
#pragma pack(pop)

static_assert(sizeof(BsgsHostBaby13) == 13, "BsgsHostBaby13 must remain 13 bytes");

static inline void bsgs_store_idx40(uint8_t out[5], uint64_t v)
{
    out[0] = (uint8_t)v;
    out[1] = (uint8_t)(v >> 8);
    out[2] = (uint8_t)(v >> 16);
    out[3] = (uint8_t)(v >> 24);
    out[4] = (uint8_t)(v >> 32);
}

static inline uint64_t bsgs_load_idx40(const uint8_t in[5])
{
    return (uint64_t)in[0] |
           ((uint64_t)in[1] << 8) |
           ((uint64_t)in[2] << 16) |
           ((uint64_t)in[3] << 24) |
           ((uint64_t)in[4] << 32);
}

struct BsgsHostHex8 {
    uint32_t key_fp;
    uint32_t idx_a;
};

static_assert(sizeof(BsgsHostHex8) == 8, "BsgsHostHex8 must remain 8 bytes");

#pragma pack(push, 1)
struct BsgsHostHex6 {
    uint32_t idx_a;
    uint16_t key_fp16;
};
#pragma pack(pop)

static_assert(sizeof(BsgsHostHex6) == 6, "BsgsHostHex6 must remain 6 bytes");

static uint64_t bsgs_host_mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static uint64_t bsgs_host_key_fingerprint(
    const uint64_t x[4],
    uint8_t y_parity,
    uint8_t flags)
{
    uint64_t fp = 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < 4; i++)
        fp ^= bsgs_host_mix64(x[i] + 0x9e3779b97f4a7c15ULL + (fp << 6) + (fp >> 2));
    fp ^= ((uint64_t)y_parity << 8) | (uint64_t)flags;
    return bsgs_host_mix64(fp);
}

static uint32_t bsgs_host_key_fingerprint_hi(
    const uint64_t x[4],
    uint8_t y_parity,
    uint8_t flags)
{
    uint64_t fp = 0xd1b54a32d192ed03ULL;
    for (int i = 0; i < 4; i++)
        fp ^= bsgs_host_mix64(x[3 - i] + 0x94d049bb133111ebULL + (fp << 7) + (fp >> 3));
    fp ^= ((uint64_t)flags << 40) | ((uint64_t)y_parity << 32) | 0x517cc1b727220a95ULL;
    return (uint32_t)bsgs_host_mix64(fp);
}

static uint32_t bsgs_host_hex8_fingerprint_key(
    const uint64_t x[4],
    uint8_t y_parity,
    uint8_t flags)
{
    const uint64_t fp = bsgs_host_key_fingerprint(x, y_parity, flags);
    const uint32_t hi = bsgs_host_key_fingerprint_hi(x, y_parity, flags);
    return (uint32_t)fp ^ (uint32_t)(fp >> 32) ^ hi;
}

static uint32_t bsgs_host_rotl32(uint32_t x, int r)
{
    return (x << r) | (x >> (32 - r));
}

static uint32_t bsgs_host_fmix32(uint32_t h)
{
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

static uint32_t bsgs_host_murmur_mix_block(uint32_t h, uint32_t k)
{
    k *= 0xcc9e2d51u;
    k = bsgs_host_rotl32(k, 15);
    k *= 0x1b873593u;
    h ^= k;
    h = bsgs_host_rotl32(h, 13);
    return h * 5u + 0xe6546b64u;
}

static uint32_t bsgs_host_bloom_hash_key(
    const uint64_t x[4],
    uint8_t y_parity,
    uint8_t flags,
    uint32_t seed)
{
    uint32_t h = seed;
    for (int i = 0; i < 4; i++) {
        h = bsgs_host_murmur_mix_block(h, (uint32_t)x[i]);
        h = bsgs_host_murmur_mix_block(h, (uint32_t)(x[i] >> 32));
    }
    const uint32_t tail = (uint32_t)y_parity | ((uint32_t)flags << 8);
    h = bsgs_host_murmur_mix_block(h, tail);
    h ^= 36u;
    return bsgs_host_fmix32(h);
}

static uint32_t bsgs_host_bucket_index_key(
    const uint64_t x[4],
    uint8_t y_parity,
    uint8_t flags,
    uint32_t bucket_mask)
{
    return bsgs_host_bloom_hash_key(x, y_parity, flags, 0x6d2b79f5u) & bucket_mask;
}

// DictX, DictY, DictValid sont définis dans Gray.h (inclus ci-dessus)

static bool precompute_ecc(const MaskParseResult &mask_r, HydraData &hd) {
    std::cout << "ECC precomputation (CPU/OpenSSL)...\n";

    int total_bits = (int)mask_r.var_bit_positions.size();
    int low_bits   = std::min(LOW_BITS, total_bits);  // bits bas -> dictionnaire
    int high_bits  = total_bits - low_bits;            // bits hauts -> Gray Code

    hd.num_var_bits     = (uint32_t)total_bits;
    hd.num_high_bits    = (uint32_t)high_bits;
    hd.total_candidates = saturated_pow2_u64(total_bits);
    hd.high_candidates  = (uint64_t)1 << high_bits;
    hd.gray_offset_start = 0;

    // P_base = k_fixed * G (high bits=0 AND low bits=0)
    if (!ec_mul_G(mask_r.k_fixed, hd.base_x, hd.base_y)) {
        std::cerr << "Error: P_base computation failed.\n"; return false;
    }
    std::cout << "  P_base computed\n";

    // HIGH deltas : Q_i = 2^(var_bit_positions[low_bits + i]) * G
    for (int i = 0; i < high_bits; i++) {
        int bit_pos = mask_r.var_bit_positions[low_bits + i];
        uint8_t scalar[32] = {0};
        int byte_idx = 31 - (bit_pos / 8);
        int bit_off  = bit_pos % 8;
        if (byte_idx >= 0 && byte_idx < 32)
            scalar[byte_idx] = (uint8_t)(1 << bit_off);
        if (!ec_mul_G(scalar, hd.delta_x[i], hd.delta_y[i])) {
            std::cerr << "Error: high delta " << i << " failed\n"; return false;
        }
    }
    std::cout << "  " << high_bits << " high deltas computed\n";

    // -----------------------------------------------------------------------
    // LOW DICTIONARY : precompute 2^low_bits affine points
    // R_k = sum of (bit_j * Q_j) for j in the active low_bits of k
    // Built by Gray Code : R_0=infinity, R_k = R_{k-1} ± Q_j
    // -----------------------------------------------------------------------
    // Base intermediate points : Q_j = 2^(var_bit_positions[j]) * G
    uint64_t Q_low_x[LOW_BITS][4], Q_low_y[LOW_BITS][4];
    for (int j = 0; j < low_bits; j++) {
        int bit_pos = mask_r.var_bit_positions[j];
        uint8_t scalar[32] = {0};
        int byte_idx = 31 - (bit_pos / 8);
        int bit_off  = bit_pos % 8;
        if (byte_idx >= 0 && byte_idx < 32)
            scalar[byte_idx] = (uint8_t)(1 << bit_off);
        if (!ec_mul_G(scalar, Q_low_x[j], Q_low_y[j])) {
            std::cerr << "Error: low delta " << j << " failed\n"; return false;
        }
    }

    // Allocate dictionary on CPU side
    // Build by direct binary decoding via OpenSSL EC_POINT_add
    static uint64_t dict_x[LOW_SIZE][4];
    static uint64_t dict_y[LOW_SIZE][4];
    static uint8_t  dict_valid[LOW_SIZE];

    // For each k, add the active Q_low[j] points via OpenSSL EC_POINT_add
    // O(2^low_bits * low_bits) additions = 64*6 = 384 ops -> negligible

    // Initialize with OpenSSL
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX   *ctx = BN_CTX_new();

    auto limbs_to_bn = [](const uint64_t limbs[4]) -> BIGNUM* {
        BIGNUM *bn = BN_new();
        uint8_t buf[32];
        for (int i=0;i<4;i++){
            uint64_t w = limbs[3-i];
            buf[i*8+0]=(w>>56)&0xFF; buf[i*8+1]=(w>>48)&0xFF;
            buf[i*8+2]=(w>>40)&0xFF; buf[i*8+3]=(w>>32)&0xFF;
            buf[i*8+4]=(w>>24)&0xFF; buf[i*8+5]=(w>>16)&0xFF;
            buf[i*8+6]=(w>> 8)&0xFF; buf[i*8+7]=(w    )&0xFF;
        }
        BN_bin2bn(buf, 32, bn);
        return bn;
    };

    auto bn_to_limbs = [](const BIGNUM *bn, uint64_t limbs[4]) {
        uint8_t buf[32] = {0};
        int len = BN_num_bytes(bn);
        BN_bn2bin(bn, buf + (32 - len));
        for (int i=0;i<4;i++){
            limbs[3-i] = ((uint64_t)buf[i*8+0]<<56)|((uint64_t)buf[i*8+1]<<48)
                        |((uint64_t)buf[i*8+2]<<40)|((uint64_t)buf[i*8+3]<<32)
                        |((uint64_t)buf[i*8+4]<<24)|((uint64_t)buf[i*8+5]<<16)
                        |((uint64_t)buf[i*8+6]<< 8)|((uint64_t)buf[i*8+7]);
        }
    };

    // Create EC_POINT for each Q_low[j]
    EC_POINT *Qpts[LOW_BITS];
    for (int j=0;j<low_bits;j++){
        Qpts[j] = EC_POINT_new(grp);
        BIGNUM *qx = limbs_to_bn(Q_low_x[j]);
        BIGNUM *qy = limbs_to_bn(Q_low_y[j]);
        EC_POINT_set_affine_coordinates(grp, Qpts[j], qx, qy, ctx);
        BN_free(qx); BN_free(qy);
    }

    // Build dict[k] for k=0..LOW_SIZE-1
    // k=0 : identity (DictValid=0)
    dict_valid[0] = 0;
    memset(dict_x[0], 0, sizeof(dict_x[0]));
    memset(dict_y[0], 0, sizeof(dict_y[0]));

    EC_POINT *acc = EC_POINT_new(grp);
    for (int k=1; k<LOW_SIZE; k++){
        // Decode k in binary and add the active Q_low[j] points
        EC_POINT_set_to_infinity(grp, acc);
        for (int j=0;j<low_bits;j++){
            if ((k>>j)&1){
                EC_POINT_add(grp, acc, acc, Qpts[j], ctx);
            }
        }
        if (EC_POINT_is_at_infinity(grp, acc)){
            dict_valid[k]=0;
            memset(dict_x[k],0,sizeof(dict_x[k]));
            memset(dict_y[k],0,sizeof(dict_y[k]));
        } else {
            BIGNUM *rx=BN_new(), *ry=BN_new();
            EC_POINT_get_affine_coordinates(grp, acc, rx, ry, ctx);
            bn_to_limbs(rx, dict_x[k]);
            bn_to_limbs(ry, dict_y[k]);
            BN_free(rx); BN_free(ry);
            dict_valid[k]=1;
        }
    }

    // Upload dictionary to constant memory
    cudaMemcpyToSymbol(DictX,     dict_x,     sizeof(dict_x));
    cudaMemcpyToSymbol(DictY,     dict_y,     sizeof(dict_y));
    cudaMemcpyToSymbol(DictValid, dict_valid, sizeof(dict_valid));

    // Cleanup
    EC_POINT_free(acc);
    for (int j=0;j<low_bits;j++) EC_POINT_free(Qpts[j]);
    EC_GROUP_free(grp); BN_CTX_free(ctx);

    std::cout << "  " << LOW_SIZE << " dictionary points precomputed (low bits)\n";
    return true;
}


// =================================================================================
// CPU : PRIVATE KEY -> BTC ADDRESSES (legacy, segwit) + ETH
// =================================================================================

// RIPEMD-160 (compact implementation)

static void ripemd160_cpu(const uint8_t* data, size_t len, uint8_t out[20]) {
    #define RMD_F(x,y,z) ((x)^(y)^(z))
    #define RMD_G(x,y,z) (((x)&(y))|(~(x)&(z)))
    #define RMD_H(x,y,z) (((x)|(~(y)))^(z))
    #define RMD_I(x,y,z) (((x)&(z))|((y)&~(z)))
    #define RMD_J(x,y,z) ((x)^((y)|(~(z))))
    #define ROL32(x,n) (((x)<<(n))|((x)>>(32-(n))))
    #define RMD_FF(a,b,c,d,e,x,s)  a=ROL32(a+RMD_F(b,c,d)+x,s)+e;c=ROL32(c,10)
    #define RMD_GG(a,b,c,d,e,x,s)  a=ROL32(a+RMD_G(b,c,d)+x+0x5A827999u,s)+e;c=ROL32(c,10)
    #define RMD_HH(a,b,c,d,e,x,s)  a=ROL32(a+RMD_H(b,c,d)+x+0x6ED9EBA1u,s)+e;c=ROL32(c,10)
    #define RMD_II(a,b,c,d,e,x,s)  a=ROL32(a+RMD_I(b,c,d)+x+0x8F1BBCDCu,s)+e;c=ROL32(c,10)
    #define RMD_JJ(a,b,c,d,e,x,s)  a=ROL32(a+RMD_J(b,c,d)+x+0xA953FD4Eu,s)+e;c=ROL32(c,10)
    #define RMD_FFF(a,b,c,d,e,x,s) a=ROL32(a+RMD_F(b,c,d)+x,s)+e;c=ROL32(c,10)
    #define RMD_GGG(a,b,c,d,e,x,s) a=ROL32(a+RMD_G(b,c,d)+x+0x7A6D76E9u,s)+e;c=ROL32(c,10)
    #define RMD_HHH(a,b,c,d,e,x,s) a=ROL32(a+RMD_H(b,c,d)+x+0x6D703EF3u,s)+e;c=ROL32(c,10)
    #define RMD_III(a,b,c,d,e,x,s) a=ROL32(a+RMD_I(b,c,d)+x+0x5C4DD124u,s)+e;c=ROL32(c,10)
    #define RMD_JJJ(a,b,c,d,e,x,s) a=ROL32(a+RMD_J(b,c,d)+x+0x50A28BE6u,s)+e;c=ROL32(c,10)
    uint32_t h0=0x67452301u,h1=0xEFCDAB89u,h2=0x98BADCFEu,h3=0x10325476u,h4=0xC3D2E1F0u;
    size_t total=((len+9+63)/64)*64;
    std::vector<uint8_t> msg(total,0);
    memcpy(msg.data(),data,len); msg[len]=0x80;
    uint64_t bits=(uint64_t)len*8;
    for(int i=0;i<8;i++) msg[total-8+i]=(uint8_t)(bits>>(i*8));
    for(size_t blk=0;blk<total;blk+=64){
        uint32_t X[16];
        for(int i=0;i<16;i++) X[i]=(uint32_t)msg[blk+i*4]|((uint32_t)msg[blk+i*4+1]<<8)|((uint32_t)msg[blk+i*4+2]<<16)|((uint32_t)msg[blk+i*4+3]<<24);
        uint32_t a=h0,b=h1,c=h2,d=h3,e=h4,aa=h0,bb=h1,cc=h2,dd=h3,ee=h4;
        RMD_FF(a,b,c,d,e,X[0],11);RMD_FF(e,a,b,c,d,X[1],14);RMD_FF(d,e,a,b,c,X[2],15);RMD_FF(c,d,e,a,b,X[3],12);
        RMD_FF(b,c,d,e,a,X[4],5);RMD_FF(a,b,c,d,e,X[5],8);RMD_FF(e,a,b,c,d,X[6],7);RMD_FF(d,e,a,b,c,X[7],9);
        RMD_FF(c,d,e,a,b,X[8],11);RMD_FF(b,c,d,e,a,X[9],13);RMD_FF(a,b,c,d,e,X[10],14);RMD_FF(e,a,b,c,d,X[11],15);
        RMD_FF(d,e,a,b,c,X[12],6);RMD_FF(c,d,e,a,b,X[13],7);RMD_FF(b,c,d,e,a,X[14],9);RMD_FF(a,b,c,d,e,X[15],8);
        RMD_GG(e,a,b,c,d,X[7],7);RMD_GG(d,e,a,b,c,X[4],6);RMD_GG(c,d,e,a,b,X[13],8);RMD_GG(b,c,d,e,a,X[1],13);
        RMD_GG(a,b,c,d,e,X[10],11);RMD_GG(e,a,b,c,d,X[6],9);RMD_GG(d,e,a,b,c,X[15],7);RMD_GG(c,d,e,a,b,X[3],15);
        RMD_GG(b,c,d,e,a,X[12],7);RMD_GG(a,b,c,d,e,X[0],12);RMD_GG(e,a,b,c,d,X[9],15);RMD_GG(d,e,a,b,c,X[5],9);
        RMD_GG(c,d,e,a,b,X[2],11);RMD_GG(b,c,d,e,a,X[14],7);RMD_GG(a,b,c,d,e,X[11],13);RMD_GG(e,a,b,c,d,X[8],12);
        RMD_HH(d,e,a,b,c,X[3],11);RMD_HH(c,d,e,a,b,X[10],13);RMD_HH(b,c,d,e,a,X[14],6);RMD_HH(a,b,c,d,e,X[4],7);
        RMD_HH(e,a,b,c,d,X[9],14);RMD_HH(d,e,a,b,c,X[15],9);RMD_HH(c,d,e,a,b,X[8],13);RMD_HH(b,c,d,e,a,X[1],15);
        RMD_HH(a,b,c,d,e,X[2],14);RMD_HH(e,a,b,c,d,X[7],8);RMD_HH(d,e,a,b,c,X[0],13);RMD_HH(c,d,e,a,b,X[6],6);
        RMD_HH(b,c,d,e,a,X[13],5);RMD_HH(a,b,c,d,e,X[11],12);RMD_HH(e,a,b,c,d,X[5],7);RMD_HH(d,e,a,b,c,X[12],5);
        RMD_II(c,d,e,a,b,X[1],11);RMD_II(b,c,d,e,a,X[9],12);RMD_II(a,b,c,d,e,X[11],14);RMD_II(e,a,b,c,d,X[10],15);
        RMD_II(d,e,a,b,c,X[0],14);RMD_II(c,d,e,a,b,X[8],15);RMD_II(b,c,d,e,a,X[12],9);RMD_II(a,b,c,d,e,X[4],8);
        RMD_II(e,a,b,c,d,X[13],9);RMD_II(d,e,a,b,c,X[3],14);RMD_II(c,d,e,a,b,X[7],5);RMD_II(b,c,d,e,a,X[15],6);
        RMD_II(a,b,c,d,e,X[14],8);RMD_II(e,a,b,c,d,X[5],6);RMD_II(d,e,a,b,c,X[6],5);RMD_II(c,d,e,a,b,X[2],12);
        RMD_JJ(b,c,d,e,a,X[4],9);RMD_JJ(a,b,c,d,e,X[0],15);RMD_JJ(e,a,b,c,d,X[5],5);RMD_JJ(d,e,a,b,c,X[9],11);
        RMD_JJ(c,d,e,a,b,X[7],6);RMD_JJ(b,c,d,e,a,X[12],8);RMD_JJ(a,b,c,d,e,X[2],13);RMD_JJ(e,a,b,c,d,X[10],12);
        RMD_JJ(d,e,a,b,c,X[14],5);RMD_JJ(c,d,e,a,b,X[1],12);RMD_JJ(b,c,d,e,a,X[3],13);RMD_JJ(a,b,c,d,e,X[8],14);
        RMD_JJ(e,a,b,c,d,X[11],11);RMD_JJ(d,e,a,b,c,X[6],8);RMD_JJ(c,d,e,a,b,X[15],5);RMD_JJ(b,c,d,e,a,X[13],6);
        RMD_JJJ(aa,bb,cc,dd,ee,X[5],8);RMD_JJJ(ee,aa,bb,cc,dd,X[14],9);RMD_JJJ(dd,ee,aa,bb,cc,X[7],9);RMD_JJJ(cc,dd,ee,aa,bb,X[0],11);
        RMD_JJJ(bb,cc,dd,ee,aa,X[9],13);RMD_JJJ(aa,bb,cc,dd,ee,X[2],15);RMD_JJJ(ee,aa,bb,cc,dd,X[11],15);RMD_JJJ(dd,ee,aa,bb,cc,X[4],5);
        RMD_JJJ(cc,dd,ee,aa,bb,X[13],7);RMD_JJJ(bb,cc,dd,ee,aa,X[6],7);RMD_JJJ(aa,bb,cc,dd,ee,X[15],8);RMD_JJJ(ee,aa,bb,cc,dd,X[8],11);
        RMD_JJJ(dd,ee,aa,bb,cc,X[1],14);RMD_JJJ(cc,dd,ee,aa,bb,X[10],14);RMD_JJJ(bb,cc,dd,ee,aa,X[3],12);RMD_JJJ(aa,bb,cc,dd,ee,X[12],6);
        RMD_III(ee,aa,bb,cc,dd,X[6],9);RMD_III(dd,ee,aa,bb,cc,X[11],13);RMD_III(cc,dd,ee,aa,bb,X[3],15);RMD_III(bb,cc,dd,ee,aa,X[7],7);
        RMD_III(aa,bb,cc,dd,ee,X[0],12);RMD_III(ee,aa,bb,cc,dd,X[13],8);RMD_III(dd,ee,aa,bb,cc,X[5],9);RMD_III(cc,dd,ee,aa,bb,X[10],11);
        RMD_III(bb,cc,dd,ee,aa,X[14],7);RMD_III(aa,bb,cc,dd,ee,X[15],7);RMD_III(ee,aa,bb,cc,dd,X[8],12);RMD_III(dd,ee,aa,bb,cc,X[12],7);
        RMD_III(cc,dd,ee,aa,bb,X[4],6);RMD_III(bb,cc,dd,ee,aa,X[9],15);RMD_III(aa,bb,cc,dd,ee,X[1],13);RMD_III(ee,aa,bb,cc,dd,X[2],11);
        RMD_HHH(dd,ee,aa,bb,cc,X[15],9);RMD_HHH(cc,dd,ee,aa,bb,X[5],7);RMD_HHH(bb,cc,dd,ee,aa,X[1],15);RMD_HHH(aa,bb,cc,dd,ee,X[3],11);
        RMD_HHH(ee,aa,bb,cc,dd,X[7],8);RMD_HHH(dd,ee,aa,bb,cc,X[14],6);RMD_HHH(cc,dd,ee,aa,bb,X[6],6);RMD_HHH(bb,cc,dd,ee,aa,X[9],14);
        RMD_HHH(aa,bb,cc,dd,ee,X[11],12);RMD_HHH(ee,aa,bb,cc,dd,X[8],13);RMD_HHH(dd,ee,aa,bb,cc,X[12],5);RMD_HHH(cc,dd,ee,aa,bb,X[2],14);
        RMD_HHH(bb,cc,dd,ee,aa,X[10],13);RMD_HHH(aa,bb,cc,dd,ee,X[0],13);RMD_HHH(ee,aa,bb,cc,dd,X[4],7);RMD_HHH(dd,ee,aa,bb,cc,X[13],5);
        RMD_GGG(cc,dd,ee,aa,bb,X[8],15);RMD_GGG(bb,cc,dd,ee,aa,X[6],5);RMD_GGG(aa,bb,cc,dd,ee,X[4],8);RMD_GGG(ee,aa,bb,cc,dd,X[1],11);
        RMD_GGG(dd,ee,aa,bb,cc,X[3],14);RMD_GGG(cc,dd,ee,aa,bb,X[11],14);RMD_GGG(bb,cc,dd,ee,aa,X[15],6);RMD_GGG(aa,bb,cc,dd,ee,X[0],14);
        RMD_GGG(ee,aa,bb,cc,dd,X[5],6);RMD_GGG(dd,ee,aa,bb,cc,X[12],9);RMD_GGG(cc,dd,ee,aa,bb,X[2],12);RMD_GGG(bb,cc,dd,ee,aa,X[13],9);
        RMD_GGG(aa,bb,cc,dd,ee,X[9],12);RMD_GGG(ee,aa,bb,cc,dd,X[7],5);RMD_GGG(dd,ee,aa,bb,cc,X[10],15);RMD_GGG(cc,dd,ee,aa,bb,X[14],8);
        RMD_FFF(bb,cc,dd,ee,aa,X[12],8);RMD_FFF(aa,bb,cc,dd,ee,X[15],5);RMD_FFF(ee,aa,bb,cc,dd,X[10],12);RMD_FFF(dd,ee,aa,bb,cc,X[4],9);
        RMD_FFF(cc,dd,ee,aa,bb,X[1],12);RMD_FFF(bb,cc,dd,ee,aa,X[5],5);RMD_FFF(aa,bb,cc,dd,ee,X[8],14);RMD_FFF(ee,aa,bb,cc,dd,X[7],6);
        RMD_FFF(dd,ee,aa,bb,cc,X[6],8);RMD_FFF(cc,dd,ee,aa,bb,X[2],13);RMD_FFF(bb,cc,dd,ee,aa,X[13],6);RMD_FFF(aa,bb,cc,dd,ee,X[14],5);
        RMD_FFF(ee,aa,bb,cc,dd,X[0],15);RMD_FFF(dd,ee,aa,bb,cc,X[3],13);RMD_FFF(cc,dd,ee,aa,bb,X[9],11);RMD_FFF(bb,cc,dd,ee,aa,X[11],11);
        uint32_t t=h1+c+dd;h1=h2+d+ee;h2=h3+e+aa;h3=h4+a+bb;h4=h0+b+cc;h0=t;
    }
    auto rle=[](uint32_t v,uint8_t* p){p[0]=v&0xFF;p[1]=(v>>8)&0xFF;p[2]=(v>>16)&0xFF;p[3]=(v>>24)&0xFF;};
    rle(h0,out);rle(h1,out+4);rle(h2,out+8);rle(h3,out+12);rle(h4,out+16);
    #undef RMD_F
    #undef RMD_G
    #undef RMD_H
    #undef RMD_I
    #undef RMD_J
    #undef ROL32
    #undef RMD_FF
    #undef RMD_GG
    #undef RMD_HH
    #undef RMD_II
    #undef RMD_JJ
    #undef RMD_FFF
    #undef RMD_GGG
    #undef RMD_HHH
    #undef RMD_III
    #undef RMD_JJJ
}
// hash160 = RIPEMD160(SHA256(data))
static void hash160_cpu(const uint8_t* data, size_t len, uint8_t h160[20]) {
    uint8_t sha[32];
    sha256_cpu(data, len, sha);
    ripemd160_cpu(sha, 32, h160);
}

// Keccak-256 (for ETH) -- pure C implementation
static void keccak256_cpu(const uint8_t* data, size_t len, uint8_t out[32]) {
    // Keccak-256 pure C implementation
    static const uint64_t RC[24] = {
        0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808AULL,0x8000000080008000ULL,
        0x000000000000808BULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
        0x000000000000008AULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000AULL,
        0x000000008000808BULL,0x800000000000008BULL,0x8000000000008089ULL,0x8000000000008003ULL,
        0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800AULL,0x800000008000000AULL,
        0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL
    };
    static const int ROT[25] = {0,1,62,28,27,36,44,6,55,20,3,10,43,25,39,41,45,15,21,8,18,2,61,56,14};
    static const int PI[25]  = {0,10,20,5,15,16,1,11,21,6,7,17,2,12,22,23,8,18,3,13,14,24,9,19,4};
    auto rotl = [](uint64_t x, int n){ return (x<<n)|(x>>(64-n)); };
    std::vector<uint8_t> msg(data, data+len);
    msg.push_back(0x01);
    while(msg.size()%136) msg.push_back(0x00);
    msg.back() |= 0x80;
    uint64_t st[25]={};
    for(size_t bs=0;bs<msg.size();bs+=136){
        for(int i=0;i<17;i++){
            uint64_t w=0;
            for(int b=0;b<8;b++) w|=(uint64_t)msg[bs+i*8+b]<<(b*8);
            st[i]^=w;
        }
        for(int r=0;r<24;r++){
            uint64_t C[5],D[5];
            for(int x=0;x<5;x++) C[x]=st[x]^st[x+5]^st[x+10]^st[x+15]^st[x+20];
            for(int x=0;x<5;x++) D[x]=C[(x+4)%5]^rotl(C[(x+1)%5],1);
            for(int i=0;i<25;i++) st[i]^=D[i%5];
            uint64_t B[25];
            for(int i=0;i<25;i++) B[PI[i]]=rotl(st[i],ROT[i]);
            for(int y=0;y<5;y++) for(int x=0;x<5;x++) st[x+5*y]=B[x+5*y]^((~B[(x+1)%5+5*y])&B[(x+2)%5+5*y]);
            st[0]^=RC[r];
        }
    }
    for(int i=0;i<4;i++) for(int b=0;b<8;b++) out[i*8+b]=(st[i]>>(b*8))&0xFF;
}

// Base58Check encode
static const char B58C[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static std::string base58check_enc(const uint8_t* payload, size_t plen) {
    uint8_t chk[32]; sha256_cpu(payload, plen, chk);
    uint8_t chk2[32]; sha256_cpu(chk, 32, chk2);
    std::vector<uint8_t> full(payload, payload+plen);
    full.insert(full.end(), chk2, chk2+4);
    // Convert to base58
    std::vector<uint8_t> digits;
    for (auto b : full) {
        int carry = b;
        for (auto& d : digits) { carry += 256*d; d = carry%58; carry /= 58; }
        while (carry) { digits.push_back(carry%58); carry /= 58; }
    }
    std::string res;
    for (auto b : full) { if(b==0) res+='1'; else break; }
    for (int i=(int)digits.size()-1;i>=0;i--) res+=B58C[digits[i]];
    return res;
}

// Bech32 encode (P2WPKH)
static std::string bech32_enc_addr(const uint8_t h160[20]) {
    static const char CHARSET[] = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
    auto polymod = [](const std::vector<int>& v) -> uint32_t {
        static const uint32_t G[]={0x3b6a57b2,0x26508e6d,0x1ea119fa,0x3d4233dd,0x2a1462b3};
        uint32_t chk=1;
        for(int val:v){int b=chk>>25;chk=((chk&0x1ffffff)<<5)^val;for(int i=0;i<5;i++)if((b>>i)&1)chk^=G[i];}
        return chk;
    };
    // Convert 8-bit to 5-bit groups
    std::vector<int> d5; d5.push_back(0); // witness version
    int acc=0,bits=0;
    for(int i=0;i<20;i++){acc=(acc<<8)|h160[i];bits+=8;while(bits>=5){bits-=5;d5.push_back((acc>>bits)&31);}}
    if(bits) d5.push_back((acc<<(5-bits))&31);
    // Expand HRP
    std::string hrp="bc";
    std::vector<int> enc;
    for(char c:hrp) enc.push_back(c>>5);
    enc.push_back(0);
    for(char c:hrp) enc.push_back(c&31);
    auto data=d5;
    for(int i=0;i<6;i++) data.push_back(0);
    for(auto x:data) enc.push_back(x);
    uint32_t pm=(polymod(enc)^1);
    for(int i=0;i<6;i++) d5.push_back((pm>>5*(5-i))&31);
    std::string res="bc1";
    for(int x:d5) res+=CHARSET[x];
    return res;
}

// Private key (32 bytes BE) -> BTC legacy, BTC segwit, ETH
static void key_to_addresses(const uint8_t key[32],
    std::string& btc_legacy, std::string& btc_segwit, std::string& eth) {
    // 1. ECC : key -> public point
    uint64_t px[4], py[4];
    if (!ec_mul_G(key, px, py)) { btc_legacy=btc_segwit=eth="ERROR"; return; }

    // 2. Compressed pubkey (33 bytes)
    uint8_t pub33[33];
    pub33[0] = (py[0]&1) ? 0x03 : 0x02;
    for (int i=0;i<4;i++) for(int b=0;b<8;b++) pub33[1+(3-i)*8+(7-b)]=(px[i]>>(b*8))&0xFF;

    // 3. hash160(pub33)
    uint8_t h160[20];
    hash160_cpu(pub33, 33, h160);

    // 4. BTC Legacy (P2PKH)
    uint8_t pl[21]; pl[0]=0x00; memcpy(pl+1, h160, 20);
    btc_legacy = base58check_enc(pl, 21);

    // 5. BTC SegWit (P2WPKH bech32)
    btc_segwit = bech32_enc_addr(h160);

    // 6. ETH : keccak256(pub64)[12:]
    uint8_t pub64[64];
    for (int i=0;i<4;i++) for(int b=0;b<8;b++){
        pub64[(3-i)*8+(7-b)] = (px[i]>>(b*8))&0xFF;
        pub64[32+(3-i)*8+(7-b)] = (py[i]>>(b*8))&0xFF;
    }
    uint8_t kh[32];
    keccak256_cpu(pub64, 64, kh);
    char ethbuf[43]; ethbuf[0]='0'; ethbuf[1]='x';
    for(int i=0;i<20;i++) snprintf(ethbuf+2+i*2, 3, "%02x", kh[12+i]);
    eth = std::string(ethbuf, 42);
}

static void key_to_wif_addresses(const uint8_t key[32], bool compressed,
    std::string& btc_legacy, std::string& btc_segwit, std::string& eth) {
    if (compressed) {
        key_to_addresses(key, btc_legacy, btc_segwit, eth);
        return;
    }

    uint64_t px[4], py[4];
    if (!ec_mul_G(key, px, py)) { btc_legacy=btc_segwit=eth="ERROR"; return; }

    uint8_t pub65[65];
    pub65[0] = 0x04;
    for (int i=0;i<4;i++) for(int b=0;b<8;b++){
        pub65[1+(3-i)*8+(7-b)] = (px[i]>>(b*8))&0xFF;
        pub65[33+(3-i)*8+(7-b)] = (py[i]>>(b*8))&0xFF;
    }

    uint8_t h160[20];
    hash160_cpu(pub65, 65, h160);
    uint8_t pl[21]; pl[0]=0x00; memcpy(pl+1, h160, 20);
    btc_legacy = base58check_enc(pl, 21);
    btc_segwit = "N/A (uncompressed WIF)";

    uint8_t pub64[64];
    memcpy(pub64, pub65 + 1, 64);
    uint8_t kh[32];
    keccak256_cpu(pub64, 64, kh);
    char ethbuf[43]; ethbuf[0]='0'; ethbuf[1]='x';
    for(int i=0;i<20;i++) snprintf(ethbuf+2+i*2, 3, "%02x", kh[12+i]);
    eth = std::string(ethbuf, 42);
}

static std::unordered_map<std::string,uint16_t> build_electrumv1_word_map() {
    std::unordered_map<std::string,uint16_t> m;
    m.reserve(ELECTRUM_V1_WORD_COUNT * 2);
    for (int i = 0; i < ELECTRUM_V1_WORD_COUNT; ++i) {
        m.emplace(std::string((const char*)h_ELECTRUM_V1_BLOB + h_ELECTRUM_V1_OFFS[i],
                              h_ELECTRUM_V1_LENS[i]), (uint16_t)i);
    }
    return m;
}

static std::string electrumv1_phrase_from_words(const uint16_t words[ELECTRUM_V1_WORDS]) {
    std::string phrase;
    for (int i = 0; i < ELECTRUM_V1_WORDS; ++i) {
        if (i) phrase += ' ';
        phrase += std::string((const char*)h_ELECTRUM_V1_BLOB + h_ELECTRUM_V1_OFFS[words[i]],
                              h_ELECTRUM_V1_LENS[words[i]]);
    }
    return phrase;
}

static bool parse_electrumv1_mask(const std::string& phrase, ElectrumV1Mask& mask) {
    auto wmap = build_electrumv1_word_map();
    std::vector<std::string> tokens;
    std::stringstream ss(phrase);
    std::string tok;
    while (ss >> tok) tokens.push_back(tok);
    if (tokens.size() != ELECTRUM_V1_WORDS) {
        std::cerr << "Error: Electrum V1 expects exactly 12 words.\n";
        return false;
    }
    mask = {};
    mask.lookahead = ELECTRUM_V1_LOOKAHEAD;
    mask.total_candidates = 1;
    for (int i = 0; i < ELECTRUM_V1_WORDS; ++i) {
        if (tokens[i] == "#") {
            if (mask.num_unknown >= ELECTRUM_V1_MAX_X) {
                std::cerr << "Error: too many unknown Electrum V1 words (max "
                          << ELECTRUM_V1_MAX_X << ")\n";
                return false;
            }
            mask.unknown_pos[mask.num_unknown++] = (uint8_t)i;
            mask.word_indices[i] = 0xFFFF;
            mask.total_candidates *= ELECTRUM_V1_WORD_COUNT;
        } else {
            auto it = wmap.find(tokens[i]);
            if (it == wmap.end()) {
                std::cerr << "Error: word not found in Electrum V1 wordlist: \""
                          << tokens[i] << "\"\n";
                return false;
            }
            mask.word_indices[i] = it->second;
        }
    }
    return true;
}

static bool parse_electrumv1_path(const std::string& s, uint32_t& change, uint32_t& address_index) {
    if (s.size() < 5 || s[0] != 'm' || s[1] != '/') return false;
    size_t slash = s.find('/', 2);
    if (slash == std::string::npos) return false;
    std::string change_s = s.substr(2, slash - 2);
    std::string index_s = s.substr(slash + 1);
    if (change_s.empty() || index_s.empty()) return false;
    for (char c : change_s) if (!std::isdigit((unsigned char)c)) return false;
    for (char c : index_s) if (!std::isdigit((unsigned char)c)) return false;
    unsigned long ch = std::stoul(change_s);
    unsigned long ix = std::stoul(index_s);
    if (ch > 1 || ix > 1000000UL) return false;
    change = (uint32_t)ch;
    address_index = (uint32_t)ix;
    return true;
}

static bool apply_electrumv1_cli_options(ElectrumV1Mask& mask, const std::vector<std::string>& opts) {
    for (size_t i = 0; i < opts.size(); ++i) {
        const std::string& opt = opts[i];
        if (opt == "--path" || opt.rfind("--path=", 0) == 0) {
            std::string value;
            if (opt == "--path") {
                if (++i >= opts.size()) {
                    std::cerr << "Error: --path expects m/<change>/<index>, e.g. --path m/0/0\n";
                    return false;
                }
                value = opts[i];
            } else {
                value = opt.substr(7);
            }
            uint32_t change = 0, index = 0;
            if (!parse_electrumv1_path(value, change, index)) {
                std::cerr << "Error: invalid Electrum V1 path '" << value
                          << "' (expected m/0/N or m/1/N).\n";
                return false;
            }
            mask.single_path = 1;
            mask.path_change = change;
            mask.path_index = index;
        } else if (opt == "--gap" || opt == "--lookahead" ||
                   opt.rfind("--gap=", 0) == 0 || opt.rfind("--lookahead=", 0) == 0) {
            std::string value;
            if (opt == "--gap" || opt == "--lookahead") {
                if (++i >= opts.size()) {
                    std::cerr << "Error: " << opt << " expects a positive integer.\n";
                    return false;
                }
                value = opts[i];
            } else {
                size_t eq = opt.find('=');
                value = opt.substr(eq + 1);
            }
            if (value.empty()) {
                std::cerr << "Error: empty Electrum V1 lookahead value.\n";
                return false;
            }
            for (char c : value) {
                if (!std::isdigit((unsigned char)c)) {
                    std::cerr << "Error: invalid Electrum V1 lookahead value '" << value << "'.\n";
                    return false;
                }
            }
            unsigned long gap = std::stoul(value);
            if (gap < 1 || gap > 1000000UL) {
                std::cerr << "Error: Electrum V1 lookahead must be between 1 and 1000000.\n";
                return false;
            }
            mask.lookahead = (uint32_t)gap;
        } else {
            std::cerr << "Error: unknown Electrum V1 option '" << opt << "'.\n";
            return false;
        }
    }
    return true;
}

static uint64_t pack_electrumv1_resume_flags(const ElectrumV1Mask& mask) {
    uint64_t flags = ((uint64_t)(mask.lookahead & 0xFFFFu) << 16);
    if (mask.single_path) {
        flags |= 1ULL;
        flags |= ((uint64_t)(mask.path_change & 0x1u) << 1);
        flags |= ((uint64_t)mask.path_index << 32);
    }
    return flags;
}

static void apply_electrumv1_resume_flags(ElectrumV1Mask& mask, uint64_t flags) {
    if (flags == 0) return;
    uint32_t lookahead = (uint32_t)((flags >> 16) & 0xFFFFu);
    if (lookahead != 0) mask.lookahead = lookahead;
    if (flags & 1ULL) {
        mask.single_path = 1;
        mask.path_change = (uint32_t)((flags >> 1) & 0x1u);
        mask.path_index = (uint32_t)(flags >> 32);
    }
}

static void electrumv1_words_for_index(
    const ElectrumV1Mask& mask, uint64_t idx, uint16_t out[ELECTRUM_V1_WORDS])
{
    for (int i = 0; i < ELECTRUM_V1_WORDS; ++i) out[i] = mask.word_indices[i];
    for (int x = (int)mask.num_unknown - 1; x >= 0; --x) {
        out[mask.unknown_pos[x]] = (uint16_t)(idx % ELECTRUM_V1_WORD_COUNT);
        idx /= ELECTRUM_V1_WORD_COUNT;
    }
}

static void electrumv1_decode_seed_ascii(const uint16_t words[ELECTRUM_V1_WORDS], uint8_t seed_ascii[32]) {
    static const char* hex = "0123456789abcdef";
    for (int group = 0; group < 4; ++group) {
        uint32_t w1 = words[group * 3 + 0];
        uint32_t w2 = words[group * 3 + 1];
        uint32_t w3 = words[group * 3 + 2];
        uint32_t n = ELECTRUM_V1_WORD_COUNT;
        uint32_t chunk = w1 + n * ((w2 + n - w1) % n) + n * n * ((w3 + n - w2) % n);
        for (int i = 0; i < 4; ++i) {
            uint8_t b = (uint8_t)(chunk >> (24 - i * 8));
            seed_ascii[group * 8 + i * 2 + 0] = (uint8_t)hex[b >> 4];
            seed_ascii[group * 8 + i * 2 + 1] = (uint8_t)hex[b & 15];
        }
    }
}

static void electrumv1_stretch_seed(const uint8_t seed_ascii[32], uint8_t master_priv[32]) {
    memcpy(master_priv, seed_ascii, 32);
    uint8_t msg[64], next[32];
    memcpy(msg + 32, seed_ascii, 32);
    for (int i = 0; i < 100000; ++i) {
        memcpy(msg, master_priv, 32);
        sha256_cpu(msg, 64, next);
        memcpy(master_priv, next, 32);
    }
}

static bool add_privkeys_mod_n_cpu(const uint8_t a[32], const uint8_t b[32], uint8_t out[32]) {
    BN_CTX* ctx = BN_CTX_new();
    BIGNUM *A = BN_bin2bn(a, 32, nullptr), *B = BN_bin2bn(b, 32, nullptr);
    BIGNUM *N = nullptr, *R = BN_new();
    EC_GROUP* grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    if (!ctx || !A || !B || !R || !grp) {
        if (ctx) BN_CTX_free(ctx); if (A) BN_free(A); if (B) BN_free(B);
        if (R) BN_free(R); if (grp) EC_GROUP_free(grp);
        return false;
    }
    N = BN_new();
    EC_GROUP_get_order(grp, N, ctx);
    BN_mod_add(R, A, B, N, ctx);
    BN_bn2binpad(R, out, 32);
    BN_free(A); BN_free(B); BN_free(N); BN_free(R);
    EC_GROUP_free(grp); BN_CTX_free(ctx);
    return true;
}

static bool cpu_derive_key_electrumv1_path(
    const std::string& phrase, uint32_t change, uint32_t address_index, uint8_t privkey[32])
{
    ElectrumV1Mask mask = {};
    if (!parse_electrumv1_mask(phrase, mask) || mask.num_unknown != 0) return false;
    uint8_t seed_ascii[32], master_priv[32];
    electrumv1_decode_seed_ascii(mask.word_indices, seed_ascii);
    electrumv1_stretch_seed(seed_ascii, master_priv);

    uint64_t mx[4], my[4];
    if (!ec_mul_G(master_priv, mx, my)) return false;
    uint8_t mpk[64];
    for (int i = 0; i < 4; ++i) for (int b = 0; b < 8; ++b) {
        mpk[(3-i)*8+(7-b)] = (mx[i] >> (b*8)) & 0xFF;
        mpk[32+(3-i)*8+(7-b)] = (my[i] >> (b*8)) & 0xFF;
    }

    std::string prefix = std::to_string(address_index) + ":" + std::to_string(change) + ":";
    std::vector<uint8_t> msg(prefix.begin(), prefix.end());
    msg.insert(msg.end(), mpk, mpk + 64);
    uint8_t seq1[32], seq2[32];
    sha256_cpu(msg.data(), msg.size(), seq1);
    sha256_cpu(seq1, 32, seq2);
    return add_privkeys_mod_n_cpu(seq2, master_priv, privkey);
}

// =================================================================================
// 4. KEY RECONSTRUCTION FROM GRAY INDEX

static void print_key(const uint8_t key[32]) {
    for (int i=0; i<32; i++) std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)key[i];
    std::cout << std::dec << "\n";
}

// =================================================================================
// 5. MAIN GPU LOOP (HEX MODE)
// =================================================================================


// =================================================================================
// BLOOM FILTER -- Load into VRAM
// =================================================================================
static const char* BLOOM_FILTER_FILE = "resources/bloom.bin";

static bool load_bloom_to_target(TargetData& target) {
    std::ifstream f(BLOOM_FILTER_FILE, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        std::cerr << "Error: cannot open '" << BLOOM_FILTER_FILE << "'\n";
        std::cerr << "  -> Create filter with : python3 tools/create_bloom.py <addresses.txt> resources/bloom.bin\n";
        return false;
    }
    size_t bytes = (size_t)f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(bytes);
    f.read((char*)buf.data(), bytes);
    if (!f) { std::cerr << "Error reading bloom.bin\n"; return false; }

    uint64_t* d_bloom = nullptr;
    cudaError_t err = cudaMalloc(&d_bloom, bytes);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMalloc failed for bloom filter ("
                  << (bytes / (1024*1024)) << " MB): "
                  << cudaGetErrorString(err) << "\n";
        return false;
    }
    err = cudaMemcpy(d_bloom, buf.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMemcpy failed for bloom filter: "
                  << cudaGetErrorString(err) << "\n";
        cudaFree(d_bloom);
        return false;
    }

    // Do not overwrite target.type if already set (BLOOM_BTC / BLOOM_ETH)
    if(target.type != TargetType::BLOOM_BTC && target.type != TargetType::BLOOM_ETH)
        target.type = TargetType::BLOOM;
    target.d_bloom_filter = d_bloom;
    target.bloom_m_bits   = (uint64_t)bytes * 8ULL;

    std::cout << "Bloom filter : " << (bytes / (1024*1024)) << " MB loaded into VRAM ("
              << target.bloom_m_bits << " bits)\n";
    return true;
}

// Returns the bloom TargetType if the arg is a bloom keyword, else BTC (sentinel)
static TargetType get_bloom_type(const std::string& s) {
    std::string l = s;
    for (auto& c : l) c = tolower(c);
    if (l == "bloombtc") return TargetType::BLOOM_BTC;
    if (l == "bloometh") return TargetType::BLOOM_ETH;
    if (l == "bloom")    return TargetType::BLOOM;
    return TargetType::BTC; // not a bloom arg
}

static bool is_bloom_arg(const std::string& s) {
    std::string l = s;
    for (auto& c : l) c = tolower(c);
    return l == "bloom" || l == "bloombtc" || l == "bloometh";
}


// =================================================================================
// PUBKEY FETCH -- BTC
// Tries to retrieve the compressed public key for a BTC address by inspecting
// its first outgoing transaction on the blockchain.
//
// The public key is only revealed when an address *spends* (in the scriptSig
// for P2PKH legacy, or in the witness stack for P2WPKH segwit).
// If the address has never spent, or if the API is unreachable, returns false
// and the caller falls back to standard hash160 comparison (BTC_EXACT mode).
//
// On success : fills target.pubkey_x[4] + target.pubkey_y_parity
//              and sets target.type = TargetType::BTC_PUBKEY
// =================================================================================

// Parse a P2PKH scriptSig hex string -> compressed pubkey hex (66 chars) or ""
// Format : <push_sig(1B)><sig(67-73B)><push_pubkey(1B)><pubkey(33 or 65B)>
static std::string parse_p2pkh_script(const std::string& script_hex) {
    if (script_hex.size() < 70) return "";
    auto hb = [](char c) -> int {
        if (c>='0'&&c<='9') return c-'0';
        if (c>='a'&&c<='f') return c-'a'+10;
        if (c>='A'&&c<='F') return c-'A'+10;
        return -1;
    };
    try {
        std::vector<uint8_t> data;
        data.reserve(script_hex.size() / 2);
        for (size_t i = 0; i + 1 < script_hex.size(); i += 2) {
            int hi = hb(script_hex[i]), lo = hb(script_hex[i+1]);
            if (hi < 0 || lo < 0) return "";
            data.push_back((uint8_t)(hi << 4 | lo));
        }
        size_t pos = 0;
        if (pos >= data.size()) return "";
        uint8_t sig_len = data[pos++];
        if (sig_len < 67 || sig_len > 73) return "";
        pos += sig_len;
        if (pos >= data.size()) return "";
        uint8_t pub_len = data[pos++];
        if (pos + pub_len > data.size()) return "";
        if (pub_len == 33 && (data[pos] == 0x02 || data[pos] == 0x03)) {
            std::string hex; hex.reserve(66);
            static const char* hx = "0123456789abcdef";
            for (size_t i = 0; i < 33; i++) { hex += hx[data[pos+i] >> 4]; hex += hx[data[pos+i] & 0xF]; }
            return hex;
        }
        if (pub_len == 65 && data[pos] == 0x04) {
            std::string hex; hex.reserve(130);
            static const char* hx = "0123456789abcdef";
            for (size_t i = 0; i < 65; i++) { hex += hx[data[pos+i] >> 4]; hex += hx[data[pos+i] & 0xF]; }
            return hex;
        }
    } catch (...) {}
    return "";
}

// Convert a pubkey hex string (33 or 65 bytes) into target.pubkey_x[4] + y_parity
static bool pubkey_hex_to_target(const std::string& pubkey_hex, TargetData& target) {
    auto hb = [](char c) -> int {
        if (c>='0'&&c<='9') return c-'0';
        if (c>='a'&&c<='f') return c-'a'+10;
        if (c>='A'&&c<='F') return c-'A'+10;
        return -1;
    };
    uint8_t xb[32] = {};
    uint8_t yb[32] = {};
    bool has_full_y = false;
    if (pubkey_hex.size() == 66) {
        if (pubkey_hex[0] != '0') return false;
        if (pubkey_hex[1] != '2' && pubkey_hex[1] != '3') return false;
        target.pubkey_y_parity = (pubkey_hex[1] == '3') ? 1 : 0;
        for (int i = 0; i < 32; i++) {
            int hi = hb(pubkey_hex[2+i*2]), lo = hb(pubkey_hex[2+i*2+1]);
            if (hi < 0 || lo < 0) return false;
            xb[i] = (uint8_t)(hi<<4|lo);
        }
    } else if (pubkey_hex.size() == 130) {
        if (pubkey_hex[0] != '0' || pubkey_hex[1] != '4') return false;
        // y parity from last byte of y (bytes 33..64 = hex pos 66..129)
        int y_lo = hb(pubkey_hex[129]);
        if (y_lo < 0) return false;
        target.pubkey_y_parity = (uint8_t)(y_lo & 1);
        for (int i = 0; i < 32; i++) {
            int hi = hb(pubkey_hex[2+i*2]), lo = hb(pubkey_hex[2+i*2+1]);
            if (hi < 0 || lo < 0) return false;
            xb[i] = (uint8_t)(hi<<4|lo);
        }
        for (int i = 0; i < 32; i++) {
            int hi = hb(pubkey_hex[66+i*2]), lo = hb(pubkey_hex[66+i*2+1]);
            if (hi < 0 || lo < 0) return false;
            yb[i] = (uint8_t)(hi<<4|lo);
        }
        has_full_y = true;
    } else {
        return false;
    }
    // Big-endian bytes -> little-endian uint64_t[4] (ECC.h limb format)
    for (int i = 0; i < 4; i++) {
        target.pubkey_x[i] = 0;
        target.pubkey_y[i] = 0;
        for (int b = 0; b < 8; b++)
            target.pubkey_x[i] |= (uint64_t)xb[31 - i*8 - b] << (b*8);
        if (has_full_y) {
            for (int b = 0; b < 8; b++)
                target.pubkey_y[i] |= (uint64_t)yb[31 - i*8 - b] << (b*8);
        }
    }
    if (!has_full_y && !secp256k1_decompress_y_from_x(
            target.pubkey_x, target.pubkey_y_parity, target.pubkey_y))
        return false;
    return true;
}

static bool parse_pubkey_target_arg(const std::string& arg, TargetData& target) {
    std::string h = arg;
    if (h.size() >= 2 && h[0] == '0' && (h[1] == 'x' || h[1] == 'X'))
        h = h.substr(2);
    if (h.size() != 66 && h.size() != 130) return false;
    if (!pubkey_hex_to_target(h, target)) return false;
    target.type = TargetType::BTC_PUBKEY;
    return true;
}

// Fetch BTC pubkey via mempool.space REST API.
// GET https://mempool.space/api/address/<addr>/txs
// Returns an array of transactions. We scan vin[].witness (P2WPKH)
// and vin[].scriptsig (P2PKH) for a valid 33-byte compressed pubkey.
// Falls back to hash160 mode (BTC_EXACT) silently on any failure.
static bool try_fetch_pubkey_btc(const std::string& btc_addr, TargetData& target) {
    std::cout << "  [PubKey] Querying mempool.space for known pubkey...\n";

    // Single API call : GET /api/address/<addr>/txs
    // If our address appears as scriptpubkey_address in a vin.prevout → it has spent → pubkey available
    // This avoids a separate /api/address/<addr> call, saving one TLS round-trip.
    std::string resp_txs;
    try {
        resp_txs = hydra_http::https_get("mempool.space", "/api/address/" + btc_addr + "/txs");
    } catch (...) {
        std::cout << "  [PubKey] API unreachable -- using hash160 mode\n";
        return false;
    }
    if (resp_txs.empty() || resp_txs == "[]") {
        std::cout << "  [PubKey] No transactions -- using hash160 mode\n";
        return false;
    }
    // Quick check : does our address appear as a spender?
    const std::string addr_check = "\"scriptpubkey_address\":\"" + btc_addr + "\"";
    if (resp_txs.find(addr_check) == std::string::npos) {
        std::cout << "  [PubKey] Address has never spent -- pubkey unknown, using hash160 mode\n";
        return false;
    }

    // 3. Scan vin[] for pubkey
    // KEY INSIGHT : scriptsig/witness contains the SPENDER's pubkey.
    // We must only read inputs where prevout.scriptpubkey_address == btc_addr.
    //
    // mempool.space vin object field order:
    //   "txid", "vout", "prevout":{...}, "scriptsig":"...", "scriptsig_asm":"...",
    //   "witness":[...], "is_coinbase", "sequence"
    //
    // Strategy: find "scriptpubkey_address":"<addr>" inside a vin.prevout,
    // then only scan until the end of that same vin object.
    // This avoids picking the pubkey of another input in multi-input spends.

    std::string pubkey_hex;
    size_t pos = 0;

    while (pubkey_hex.empty() && pos < resp_txs.size()) {

        // Find next occurrence of our address in a scriptpubkey_address field
        size_t addr_pos = resp_txs.find(addr_check, pos);
        if (addr_pos == std::string::npos) break;
        pos = addr_pos + addr_check.size();

        // Restrict the search to the current vin object.
        // mempool.space vin order is:
        //   ..., "prevout":{...,"scriptpubkey_address":"<addr>",...},
        //   "scriptsig":"...", "scriptsig_asm":"...", "witness":[...],
        //   "is_coinbase":false, "sequence":...
        size_t search_end = resp_txs.find("\"sequence\":", pos);
        if (search_end == std::string::npos) search_end = resp_txs.size();

        // If another prevout starts before the next sequence, the current match was ambiguous.
        size_t next_prevout = resp_txs.find("\"prevout\":{", pos);
        if (next_prevout != std::string::npos && next_prevout < search_end) {
            search_end = next_prevout;
        }

        // Try witness first (P2WPKH) — contains pubkey directly as 66-char hex
        size_t w = resp_txs.find("\"witness\":[", pos);
        if (w != std::string::npos && w < search_end) {
            size_t wend = resp_txs.find(']', w + 10);
            if (wend != std::string::npos && wend < search_end) {
                size_t wp = w + 10;
                while (wp < wend) {
                    size_t q1 = resp_txs.find('"', wp);
                    if (q1 == std::string::npos || q1 >= wend) break;
                    size_t q2 = resp_txs.find('"', q1 + 1);
                    if (q2 == std::string::npos || q2 > wend) break;
                    std::string item = resp_txs.substr(q1 + 1, q2 - q1 - 1);
                    if (item.size() == 66 &&
                        item[0] == '0' && (item[1] == '2' || item[1] == '3')) {
                        bool ok = true;
                        for (char ch : item) if (!isxdigit((unsigned char)ch)) { ok=false; break; }
                        if (ok) { pubkey_hex = item; break; }
                    }
                    wp = q2 + 1;
                }
                if (!pubkey_hex.empty()) break;
            }
        }

        // Try scriptsig (P2PKH legacy) — contains DER sig + pubkey
        size_t s = resp_txs.find("\"scriptsig\":\"", pos);
        if (s != std::string::npos && s < search_end) {
            s += 13; // skip key + opening quote
            size_t send = resp_txs.find('"', s);
            if (send != std::string::npos && send < search_end) {
                std::string script = resp_txs.substr(s, send - s);
                pubkey_hex = parse_p2pkh_script(script);
            }
        }
    }

    if (pubkey_hex.empty()) {
        std::cout << "  [PubKey] No pubkey found in transactions -- using hash160 mode\n";
        return false;
    }

    if (!pubkey_hex_to_target(pubkey_hex, target)) {
        std::cout << "  [PubKey] Pubkey parse error -- using hash160 mode\n";
        return false;
    }

    target.type = TargetType::BTC_PUBKEY;
    std::cout << "  [PubKey] OK -- SHA256+RIPEMD160 bypassed ("
              << (pubkey_hex.size() == 66 ? "compressed" : "uncompressed") << ")\n";
    std::cout << "  [PubKey] " << pubkey_hex.substr(0, 20) << "..." << pubkey_hex.substr(pubkey_hex.size()-8) << "\n";
    return true;
}


// =================================================================================
// PUBKEY FETCH -- ETH
//
// ETH addresses are keccak256(pubkey)[12:] -- the pubkey is never stored on-chain.
// However, any signed transaction reveals the pubkey via ECDSA signature recovery :
//   pubkey = ecrecover(tx_signing_hash, v, r, s)
//
// The tx signing hash is keccak256(RLP(tx_fields_without_signature)).
// For legacy/EIP-155 txs : RLP(nonce, gasPrice, gas, to, value, data [, chainId, 0, 0])
// For EIP-1559 (type 2) : 0x02 || RLP(chainId, nonce, maxPrioFee, maxFee, gas, to, value, data, [])
//
// We use Blockscout to fetch outgoing transactions and JSON-RPC tx details.
// =================================================================================

// ---- Minimal RLP encoder (big-endian integers, byte strings, lists) ----

static std::vector<uint8_t> rlp_int(uint64_t n) {
    if (n == 0) return {0x80};
    // Encode n as big-endian minimal bytes
    uint8_t buf[8]; int len = 0;
    uint64_t tmp = n;
    while (tmp > 0) { buf[7 - len++] = (uint8_t)(tmp & 0xFF); tmp >>= 8; }
    uint8_t* p = buf + (8 - len);
    if (len == 1 && p[0] < 0x80) return {p[0]};
    std::vector<uint8_t> r; r.push_back(0x80 + len);
    r.insert(r.end(), p, p + len);
    return r;
}

static std::vector<uint8_t> rlp_int256(const std::string& hex) {
    // Decode hex string (with or without 0x) → big-endian bytes, strip leading zeros
    std::string h = hex;
    if (h.size() >= 2 && h[0] == '0' && (h[1] == 'x' || h[1] == 'X')) h = h.substr(2);
    // Ensure even length
    if (h.size() & 1) h = "0" + h;
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < h.size(); i += 2) {
        auto hb = [](char c) -> int {
            if (c>='0'&&c<='9') return c-'0';
            if (c>='a'&&c<='f') return c-'a'+10;
            if (c>='A'&&c<='F') return c-'A'+10;
            return 0;
        };
        bytes.push_back((uint8_t)(hb(h[i]) << 4 | hb(h[i+1])));
    }
    // Strip leading zero bytes
    size_t start = 0;
    while (start < bytes.size() - 1 && bytes[start] == 0) start++;
    bytes = std::vector<uint8_t>(bytes.begin() + start, bytes.end());
    if (bytes.empty() || (bytes.size() == 1 && bytes[0] == 0)) return {0x80};
    if (bytes.size() == 1 && bytes[0] < 0x80) return bytes;
    std::vector<uint8_t> r; r.push_back((uint8_t)(0x80 + bytes.size()));
    r.insert(r.end(), bytes.begin(), bytes.end());
    return r;
}

static std::vector<uint8_t> rlp_bytes(const std::vector<uint8_t>& b) {
    std::vector<uint8_t> r;
    if (b.empty()) { r.push_back(0x80); return r; }
    if (b.size() == 1 && b[0] < 0x80) { r.push_back(b[0]); return r; }
    r.reserve(b.size() + 9);
    if (b.size() <= 55) {
        r.push_back((uint8_t)(0x80 + b.size()));
    } else {
        uint8_t lb[8]; int ll = 0;
        size_t sz = b.size(); while (sz > 0) { lb[7-ll++] = (uint8_t)(sz&0xFF); sz >>= 8; }
        r.push_back((uint8_t)(0xB7 + ll));
        for (int i = 8-ll; i < 8; i++) r.push_back(lb[i]);
    }
    for (uint8_t byte : b) r.push_back(byte);
    return r;
}

static std::vector<uint8_t> rlp_addr(const std::string& hex_addr) {
    // 20-byte address
    std::string h = hex_addr;
    if (h.size() >= 2 && h[0] == '0' && (h[1]=='x'||h[1]=='X')) h = h.substr(2);
    if (h.empty()) return {0x80}; // null address (contract creation)
    std::vector<uint8_t> b;
    for (size_t i = 0; i + 1 < h.size(); i += 2) {
        auto hb = [](char c)->int{
            if(c>='0'&&c<='9') return c-'0';
            if(c>='a'&&c<='f') return c-'a'+10;
            if(c>='A'&&c<='F') return c-'A'+10;
            return 0;};
        b.push_back((uint8_t)(hb(h[i])<<4|hb(h[i+1])));
    }
    return rlp_bytes(b);
}

static std::vector<uint8_t> rlp_data(const std::string& hex_data) {
    if (hex_data.empty() || hex_data == "0x" || hex_data == "0X")
        return {0x80};
    std::string h = hex_data;
    if (h.size() >= 2 && h[0]=='0' && (h[1]=='x'||h[1]=='X')) h = h.substr(2);
    if (h.empty()) return {0x80};
    std::vector<uint8_t> b;
    for (size_t i = 0; i+1 < h.size(); i += 2) {
        auto hb=[](char c)->int{
            if(c>='0'&&c<='9') return c-'0';
            if(c>='a'&&c<='f') return c-'a'+10;
            if(c>='A'&&c<='F') return c-'A'+10;
            return 0;};
        b.push_back((uint8_t)(hb(h[i])<<4|hb(h[i+1])));
    }
    return rlp_bytes(b);
}

static std::vector<uint8_t> rlp_list(const std::vector<std::vector<uint8_t>>& items) {
    std::vector<uint8_t> payload;
    for (auto& it : items) payload.insert(payload.end(), it.begin(), it.end());
    std::vector<uint8_t> r;
    if (payload.size() <= 55) {
        r.push_back((uint8_t)(0xC0 + payload.size()));
    } else {
        uint8_t lb[8]; int ll = 0;
        size_t sz = payload.size(); while (sz > 0) { lb[7-ll++]=(uint8_t)(sz&0xFF); sz>>=8; }
        r.push_back((uint8_t)(0xF7 + ll));
        r.insert(r.end(), lb + (8-ll), lb + 8);
    }
    r.insert(r.end(), payload.begin(), payload.end());
    return r;
}

// ---- keccak256 CPU (for signing hash) ----
static void keccak256_cpu_eth(const uint8_t* data, size_t len, uint8_t out[32]) {
    // Reuse existing keccak256_cpu function
    keccak256_cpu(data, len, out);
}

// ---- secp256k1 ecrecover (CPU via OpenSSL) ----
// Recovers the public key from an ECDSA signature (r, s, rec_id) and message hash.
// Returns true and fills pub_x[32], pub_y[32] (big-endian) on success.
static bool secp256k1_ecrecover(
    const uint8_t msg_hash[32],
    uint8_t rec_id,          // 0 or 1
    const uint8_t r_be[32],
    const uint8_t s_be[32],
    uint8_t pub_x[32],
    uint8_t pub_y[32])
{
    EC_GROUP* grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX*   ctx = BN_CTX_new();
    bool ok = false;

    BIGNUM* r  = BN_bin2bn(r_be, 32, nullptr);
    BIGNUM* s  = BN_bin2bn(s_be, 32, nullptr);
    BIGNUM* e  = BN_bin2bn(msg_hash, 32, nullptr);
    BIGNUM* n  = BN_new(); EC_GROUP_get_order(grp, n, ctx);
    BIGNUM* p  = BN_new();
    {
        const EC_POINT* gen = EC_GROUP_get0_generator(grp);
        BIGNUM* gx = BN_new(), *gy = BN_new();
        EC_POINT_get_affine_coordinates(grp, gen, gx, gy, ctx);
        // p = field prime (not group order)
        // Get it from curve parameters
        BN_free(gx); BN_free(gy);
    }
    // Get field prime from curve
    BIGNUM* a = BN_new(), *b_coef = BN_new(), *field_p = BN_new();
    EC_GROUP_get_curve(grp, field_p, a, b_coef, ctx);
    BN_free(a); BN_free(b_coef);

    // x = r + rec_id * n  (rec_id & 2 is essentially impossible for secp256k1)
    BIGNUM* x = BN_dup(r);
    if (rec_id & 2) BN_add(x, x, n);

    // Check x < field_p
    if (BN_cmp(x, field_p) >= 0) goto cleanup;

    {
        // Compute R = point with x-coord, y chosen by rec_id parity
        EC_POINT* R = EC_POINT_new(grp);
        BIGNUM* y_sq = BN_new();
        BIGNUM* y    = BN_new();
        BIGNUM* exp  = BN_new();
        BIGNUM* three = BN_new(); BN_set_word(three, 3);
        BIGNUM* seven = BN_new(); BN_set_word(seven, 7);

        // y^2 = x^3 + 7 mod p
        BN_mod_exp(y_sq, x, three, field_p, ctx);
        BN_mod_add(y_sq, y_sq, seven, field_p, ctx);

        // y = sqrt(y_sq) mod p  (p ≡ 3 mod 4, so y = y_sq^((p+1)/4) mod p)
        BN_copy(exp, field_p);
        BN_add_word(exp, 1);
        BN_rshift(exp, exp, 2);
        BN_mod_exp(y, y_sq, exp, field_p, ctx);

        // Choose correct y parity
        if ((BN_is_odd(y) ? 1 : 0) != (rec_id & 1))
            BN_sub(y, field_p, y);

        EC_POINT_set_affine_coordinates(grp, R, x, y, ctx);

        // Q = r^-1 * (s * R - e * G)
        BIGNUM* r_inv = BN_new();
        BN_mod_inverse(r_inv, r, n, ctx);

        // sR
        EC_POINT* sR = EC_POINT_new(grp);
        EC_POINT_mul(grp, sR, nullptr, R, s, ctx);

        // eG (negated : -e mod n)
        BIGNUM* neg_e = BN_new();
        BN_mod_sub(neg_e, n, e, n, ctx); // neg_e = n - e (equiv to -e mod n)
        EC_POINT* eG = EC_POINT_new(grp);
        EC_POINT_mul(grp, eG, neg_e, nullptr, nullptr, ctx);

        // sR + (-eG) = sR - eG
        EC_POINT* Q = EC_POINT_new(grp);
        EC_POINT_add(grp, Q, sR, eG, ctx);

        // Q = r_inv * Q
        EC_POINT_mul(grp, Q, nullptr, Q, r_inv, ctx);

        // Extract coordinates
        BIGNUM* qx = BN_new(), *qy = BN_new();
        EC_POINT_get_affine_coordinates(grp, Q, qx, qy, ctx);
        BN_bn2binpad(qx, pub_x, 32);
        BN_bn2binpad(qy, pub_y, 32);
        ok = true;

        BN_free(qx); BN_free(qy);
        BN_free(r_inv); BN_free(neg_e);
        EC_POINT_free(sR); EC_POINT_free(eG); EC_POINT_free(Q); EC_POINT_free(R);
        BN_free(y_sq); BN_free(y); BN_free(exp); BN_free(three); BN_free(seven);
    }

cleanup:
    BN_free(r); BN_free(s); BN_free(e); BN_free(n); BN_free(x); BN_free(field_p);
    BN_CTX_free(ctx);
    EC_GROUP_free(grp);
    return ok;
}

// ---- Parse hex string → uint64_t ----
static uint64_t hex_to_u64(const std::string& h) {
    if (h.empty()) return 0;
    std::string s = h;
    if (s.size() >= 2 && s[0]=='0' && (s[1]=='x'||s[1]=='X')) s = s.substr(2);
    uint64_t v = 0;
    for (char c : s) {
        v <<= 4;
        if (c>='0'&&c<='9') v |= c-'0';
        else if (c>='a'&&c<='f') v |= c-'a'+10;
        else if (c>='A'&&c<='F') v |= c-'A'+10;
    }
    return v;
}

// ---- Compute signing hash for a legacy / EIP-155 transaction ----
static bool compute_legacy_signing_hash(
    const std::string& nonce_h,
    const std::string& gas_price_h,
    const std::string& gas_h,
    const std::string& to_h,
    const std::string& value_h,
    const std::string& input_h,
    uint64_t chain_id,
    uint8_t out_hash[32])
{
    std::vector<std::vector<uint8_t>> fields = {
        rlp_int256(nonce_h),
        rlp_int256(gas_price_h),
        rlp_int256(gas_h),
        rlp_addr(to_h),
        rlp_int256(value_h),
        rlp_data(input_h),
    };
    if (chain_id > 0) {
        // EIP-155 : append chainId, 0, 0
        fields.push_back(rlp_int(chain_id));
        fields.push_back({0x80}); // 0
        fields.push_back({0x80}); // 0
    }
    auto encoded = rlp_list(fields);
    keccak256_cpu_eth(encoded.data(), encoded.size(), out_hash);
    return true;
}

// ---- Compute signing hash for an EIP-1559 (type 2) transaction ----
static bool compute_eip1559_signing_hash(
    uint64_t chain_id,
    const std::string& nonce_h,
    const std::string& max_prio_h,
    const std::string& max_fee_h,
    const std::string& gas_h,
    const std::string& to_h,
    const std::string& value_h,
    const std::string& input_h,
    uint8_t out_hash[32])
{
    std::vector<std::vector<uint8_t>> fields = {
        rlp_int(chain_id),
        rlp_int256(nonce_h),
        rlp_int256(max_prio_h),
        rlp_int256(max_fee_h),
        rlp_int256(gas_h),
        rlp_addr(to_h),
        rlp_int256(value_h),
        rlp_data(input_h),
        {0xC0},  // empty access list
    };
    auto payload = rlp_list(fields);
    // Prepend type byte 0x02
    std::vector<uint8_t> full; full.push_back(0x02);
    full.insert(full.end(), payload.begin(), payload.end());
    keccak256_cpu_eth(full.data(), full.size(), out_hash);
    return true;
}

// ---- JSON field extraction helpers ----
static std::string json_str_field(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\":\"";
    size_t p = json.find(needle);
    if (p == std::string::npos) return "";
    p += needle.size();
    size_t e = json.find('"', p);
    if (e == std::string::npos) return "";
    return json.substr(p, e - p);
}



// ---- Blockscout API fetch ----
static std::string blockscout_get(const std::string& params) {
    std::string path = "/api?" + params;
    return hydra_http::https_get(BLOCKSCOUT_HOST, path);
}

static std::string blockscout_rpc_get_transaction(const std::string& txhash) {
    std::string body = "{\"jsonrpc\":\"2.0\",\"method\":\"eth_getTransactionByHash\",\"params\":[\"" +
                       txhash + "\"],\"id\":1}";
    return hydra_http::https_post(BLOCKSCOUT_HOST, "/api/eth-rpc", body);
}

// ---- Main ETH pubkey fetch function ----
static bool try_fetch_pubkey_eth(const std::string& eth_addr, TargetData& target) {
    std::cout << "  [PubKey] Querying Blockscout for known ETH pubkey...\n";

    // 1. Fetch outgoing transactions
    std::string txlist;
    try {
        txlist = blockscout_get(
            "module=account&action=txlist&address=" + eth_addr +
            "&startblock=0&endblock=99999999&sort=asc&page=1&offset=5");
    } catch (...) {
        std::cout << "  [PubKey] Blockscout unreachable -- using keccak mode\n";
        return false;
    }

    if (txlist.empty()) {
        std::cout << "  [PubKey] Blockscout unreachable -- using keccak mode\n";
        return false;
    }

    // Blockscout account API success : "status":"1" and "result":[...]
    if (txlist.find("\"status\":\"1\"") == std::string::npos) {
        std::cout << "  [PubKey] No outgoing transactions -- pubkey unknown\n";
        return false;
    }

    // Extract tx hashes where from == eth_addr
    // Strategy : scan each tx object by finding "from":"<addr>" first,
    // then look backward for "hash":"0x..." within the same object.
    // This avoids confusion with "blockHash" field.
    std::string addr_low = eth_addr;
    for (auto& ch : addr_low) ch = (char)tolower((unsigned char)ch);
    std::string txlist_low = txlist;
    for (auto& ch : txlist_low) ch = (char)tolower((unsigned char)ch);

    const std::string from_needle = "\"from\":\"" + addr_low + "\"";
    std::vector<std::string> tx_hashes;
    size_t pos = 0;

    while (tx_hashes.size() < 5 && pos < txlist.size()) {
        // Find a "from":"<our_addr>" occurrence
        size_t fp = txlist_low.find(from_needle, pos);
        if (fp == std::string::npos) break;
        pos = fp + from_needle.size();

        // Look backward for the tx hash field.
        // The txlist JSON has both "blockHash" and "hash" — we want only "hash".
        // We match on ,"hash":" or {"hash":" to exclude "blockHash":"
        size_t search_start = (fp > 600) ? fp - 600 : 0;
        size_t hp = std::string::npos;
        // Try both separators
        for (const char* needle : {",\"hash\":\"0x", "{\"hash\":\"0x"}) {
            size_t scan = search_start;
            while (scan < fp) {
                size_t found = txlist_low.find(needle, scan);
                if (found == std::string::npos || found >= fp) break;
                hp = found;
                scan = found + 1;
            }
            if (hp != std::string::npos) break;
        }
        if (hp == std::string::npos) continue;
        // Skip to the 0x part
        size_t hval = txlist.find("\"0x", hp) + 1;
        size_t he = txlist.find('"', hval);
        if (he == std::string::npos) continue;
        std::string txhash = txlist.substr(hval, he - hval);
        if (txhash.size() == 66)
            tx_hashes.push_back(txhash);
    }

    if (tx_hashes.empty()) {
        std::cout << "  [PubKey] No outgoing tx from this address -- pubkey unknown\n";
        return false;
    }

    // 2. For each tx, fetch detail and attempt ecrecover
    for (const auto& txhash : tx_hashes) {
        std::string detail;
        try {
            detail = blockscout_rpc_get_transaction(txhash);
        } catch (...) {
            std::cout << "  [PubKey] Blockscout RPC unreachable -- using keccak mode\n";
            return false;
        }

        if (detail.empty()) continue;

        // Parse fields
        std::string type_s    = json_str_field(detail, "type");
        std::string nonce_s   = json_str_field(detail, "nonce");
        std::string gas_s     = json_str_field(detail, "gas");
        std::string to_s      = json_str_field(detail, "to");
        std::string value_s   = json_str_field(detail, "value");
        std::string input_s   = json_str_field(detail, "input");
        std::string chain_s   = json_str_field(detail, "chainId");
        std::string v_s       = json_str_field(detail, "v");
        std::string r_s       = json_str_field(detail, "r");
        std::string s_s       = json_str_field(detail, "s");

        if (r_s.empty() || s_s.empty() || v_s.empty()) continue;

        uint64_t tx_type  = hex_to_u64(type_s);
        uint64_t v_raw    = hex_to_u64(v_s);
        uint64_t chain_id = chain_s.empty() ? 1 : hex_to_u64(chain_s);
        uint64_t signing_chain_id = chain_id;

        // Compute recovery id
        uint8_t rec_id;
        if (tx_type == 2) {
            rec_id = (uint8_t)(v_raw & 1);
        } else if (v_raw == 27 || v_raw == 28) {
            rec_id = (uint8_t)(v_raw - 27);
            signing_chain_id = 0;
        } else if (v_raw >= 35) {
            uint64_t rid = v_raw - chain_id * 2 - 35;
            if (rid > 1) continue;
            rec_id = (uint8_t)rid;
        } else {
            rec_id = (uint8_t)(v_raw & 1);
        }

        // Compute signing hash
        uint8_t signing_hash[32];
        bool hash_ok = false;

        if (tx_type == 2) {
            std::string maxprio_s = json_str_field(detail, "maxPriorityFeePerGas");
            std::string maxfee_s  = json_str_field(detail, "maxFeePerGas");
            hash_ok = compute_eip1559_signing_hash(
                chain_id, nonce_s, maxprio_s, maxfee_s,
                gas_s, to_s, value_s, input_s, signing_hash);
        } else {
            hash_ok = compute_legacy_signing_hash(
                nonce_s, json_str_field(detail, "gasPrice"),
                gas_s, to_s, value_s, input_s, signing_chain_id, signing_hash);
        }
        if (!hash_ok) continue;

        // Decode r and s as 32-byte big-endian
        auto hex32 = [](const std::string& h, uint8_t out[32]) {
            std::string s = h;
            if (s.size()>=2&&s[0]=='0'&&(s[1]=='x'||s[1]=='X')) s=s.substr(2);
            while (s.size() < 64) s = "0" + s;
            for (int i = 0; i < 32; i++) {
                auto hb=[](char c)->int{
                    if(c>='0'&&c<='9')return c-'0';
                    if(c>='a'&&c<='f')return c-'a'+10;
                    if(c>='A'&&c<='F')return c-'A'+10;
                    return 0;};
                out[i] = (uint8_t)(hb(s[i*2])<<4|hb(s[i*2+1]));
            }
        };

        uint8_t r_be[32], s_be[32];
        hex32(r_s, r_be);
        hex32(s_s, s_be);

        // ecrecover
        uint8_t pub_x[32], pub_y[32];
        if (!secp256k1_ecrecover(signing_hash, rec_id, r_be, s_be, pub_x, pub_y))
            continue;

        // Verify : keccak256(pub_x || pub_y)[12:] == eth_addr
        uint8_t pub64[64];
        memcpy(pub64, pub_x, 32);
        memcpy(pub64+32, pub_y, 32);
        uint8_t addr_hash[32];
        keccak256_cpu(pub64, 64, addr_hash);
        char recovered[43]; recovered[0]='0'; recovered[1]='x';
        for (int i=0;i<20;i++) snprintf(recovered+2+i*2,3,"%02x",addr_hash[12+i]);

        std::string rec_addr(recovered, 42);
        std::string eth_low = eth_addr;
        for (auto& ch : eth_low) ch=(char)tolower((unsigned char)ch);
        if (rec_addr != eth_low) continue; // hash mismatch, try next tx

        // Store in target
        for (int i = 0; i < 4; i++) {
            target.pubkey_x[i] = 0;
            target.pubkey_y[i] = 0;
            for (int b = 0; b < 8; b++) {
                target.pubkey_x[i] |= (uint64_t)pub_x[31 - i*8 - b] << (b*8);
                target.pubkey_y[i] |= (uint64_t)pub_y[31 - i*8 - b] << (b*8);
            }
        }
        target.pubkey_y_parity = pub_y[31] & 1;
        target.type = TargetType::ETH_PUBKEY;

        std::cout << "  [PubKey] OK -- keccak256 bypassed (ecrecover succeeded)\n";
        std::cout << "  [PubKey] px=" << std::hex;
        for (int i=0;i<4;i++) std::cout << std::setw(16) << std::setfill('0') << target.pubkey_x[3-i];
        std::cout << std::dec << "\n";
        return true;
    }

    std::cout << "  [PubKey] ecrecover failed on all txs -- using keccak mode\n";
    return false;
}

static bool resolve_bsgs_pubkey_target(const std::string& target_arg, TargetData& target, const char* input_label = "HEX")
{
    if (parse_pubkey_target_arg(target_arg, target)) {
        std::cout << "Mode : " << input_label << " BSGS + PubKey\n";
        return true;
    }

    if (target_arg.size() >= 2 && target_arg[0] == '0' &&
        (target_arg[1] == 'x' || target_arg[1] == 'X')) {
        uint8_t eth20[20] = {};
        if (!ethAddrToBytes(target_arg, eth20)) {
            std::cerr << "Error: invalid ETH address/pubkey for BSGS.\n";
            return false;
        }
        target = {};
        if (!try_fetch_pubkey_eth(target_arg, target) || target.type != TargetType::ETH_PUBKEY) {
            std::cerr << "Error: BSGS requires the exact public key. "
                      << "No ETH pubkey could be recovered for this address.\n";
            return false;
        }
        std::cout << "Mode : " << input_label << " BSGS + ETH PubKey (auto)\n";
        return true;
    }

    uint8_t h160[20] = {};
    if (!addrToHash160Any(target_arg, h160)) {
        std::cerr << "Error: BSGS target must be a pubkey, BTC address, or ETH address.\n";
        return false;
    }
    target = {};
    if (!try_fetch_pubkey_btc(target_arg, target) || target.type != TargetType::BTC_PUBKEY) {
        std::cerr << "Error: BSGS requires the exact public key. "
                  << "No BTC pubkey could be recovered for this address.\n";
        return false;
    }
    std::cout << "Mode : " << input_label << " BSGS + BTC PubKey (auto)\n";
    return true;
}

static bool try_resolve_btc_pubkey_target_for_scheduler(const std::string& target_arg, TargetData& target)
{
    if (parse_pubkey_target_arg(target_arg, target))
        return true;

    uint8_t h160[20] = {};
    if (!addrToHash160Any(target_arg, h160))
        return false;

    return try_fetch_pubkey_btc(target_arg, target) && target.type == TargetType::BTC_PUBKEY;
}

static bool try_resolve_any_pubkey_target_for_scheduler(const std::string& target_arg, TargetData& target)
{
    if (parse_pubkey_target_arg(target_arg, target))
        return true;

    if (target_arg.size() >= 2 && target_arg[0] == '0' &&
        (target_arg[1] == 'x' || target_arg[1] == 'X')) {
        return try_fetch_pubkey_eth(target_arg, target) && target.type == TargetType::ETH_PUBKEY;
    }

    uint8_t h160[20] = {};
    if (!addrToHash160Any(target_arg, h160))
        return false;

    return try_fetch_pubkey_btc(target_arg, target) && target.type == TargetType::BTC_PUBKEY;
}

static int run_wif_bsgs_ram_baby_mode(
    const std::string& wif_str,
    const std::string& pubkey_str,
    const TargetData& target,
    const BsgsPlan& plan,
    uint64_t free_bytes,
    uint64_t vram_budget,
    bool have_vram,
    const ResumeState* resume,
    const std::chrono::steady_clock::time_point& t_total0)
{
    using ClockLocal = std::chrono::steady_clock;
    auto elapsed_ms = [](ClockLocal::time_point a, ClockLocal::time_point b) -> double {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    const uint64_t bloom_bits = bsgs_bloom_bits_for_baby_count(plan.baby_count);
    const uint64_t bloom_mb = bloom_bits / 8ULL / 1024ULL / 1024ULL;
    const uint32_t bucket_bits = bsgs_ram_bucket_bits_for_baby_count(plan.baby_count);
    const uint32_t bucket_count = 1u << bucket_bits;
    const uint64_t hit_capacity = 1ULL << 20;
    const bool wide_baby_idx = plan.baby_count > (uint64_t)UINT32_MAX + 1ULL;
    const uint64_t host_entry_size = wide_baby_idx ? sizeof(BsgsHostBaby13) : sizeof(BsgsHostBaby12);
    const char* host_entry_label = wide_baby_idx ? "Baby13 idx40" : "Baby12";

    uint64_t chunk_capacity = 1ULL << 22; // 4M compact GPU entries ~= 128 MiB
    if (chunk_capacity > plan.baby_count) chunk_capacity = plan.baby_count;

    const uint64_t estimated_vram =
        bloom_bits / 8ULL +
        chunk_capacity * sizeof(BsgsBabyEntry) +
        hit_capacity * sizeof(BsgsHit) +
        64ULL * 1024ULL * 1024ULL;
    const uint64_t estimated_ram =
        plan.baby_count * host_entry_size +
        ((uint64_t)bucket_count + 1ULL) * sizeof(uint64_t) +
        (uint64_t)bucket_count * (wide_baby_idx ? sizeof(uint64_t) : sizeof(uint32_t)) +
        chunk_capacity * sizeof(BsgsBabyEntry);

    if (have_vram) {
        std::cout << "[BSGS] VRAM free: " << format_bytes_mb(free_bytes)
                  << " / budget: " << format_bytes_mb(vram_budget)
                  << " / RAM-baby estimated VRAM: " << format_bytes_mb(estimated_vram) << "\n";
    }
    std::cout << "[BSGS] RAM baby table: " << host_entry_label << " "
              << format_bytes_mb(plan.baby_count * host_entry_size)
              << " / total host estimate: " << format_bytes_mb(estimated_ram) << "\n";
    std::cout << "[BSGS] Bloom-only GPU filter: " << bloom_mb << " MB ("
              << bloom_bits << " bits, k=" << BLOOM_K_HASHES << ")\n";
    const double avg_bucket_entries = bucket_count ? (double)plan.baby_count / (double)bucket_count : 0.0;
    std::cout << "[BSGS] RAM buckets: " << bucket_count
              << " (2^" << bucket_bits << ", avg ~" << std::fixed << std::setprecision(1)
              << avg_bucket_entries << " entries/bucket)\n" << std::defaultfloat;
    std::cout << "[BSGS] Baby RAM chunk: " << chunk_capacity << " entries ("
              << format_bytes_mb(chunk_capacity * sizeof(BsgsBabyEntry)) << " GPU temp)\n";

    BsgsGpuBuffers buffers = {};
    cudaError_t cerr = bsgs_alloc_bloom_buffers(
        buffers, chunk_capacity, hit_capacity, bloom_bits);
    if (cerr != cudaSuccess) {
        std::cout << "[BSGS] RAM-baby GPU allocation skipped: "
                  << cudaGetErrorString(cerr) << "\n";
        return 0;
    }

    cerr = bsgs_upload_plan(plan, buffers);
    if (cerr != cudaSuccess) {
        std::cout << "[BSGS] Plan upload failed: " << cudaGetErrorString(cerr) << "\n";
        bsgs_free_buffers(buffers);
        return 0;
    }

    std::vector<BsgsBabyEntry> chunk_entries((size_t)chunk_capacity);
    std::vector<uint32_t> bucket_counts((size_t)bucket_count, 0);
    const uint32_t bucket_mask = bucket_count - 1u;

    auto emit_chunk = [&](uint64_t offset, uint64_t count, bool add_bloom) -> cudaError_t {
        cudaError_t e = bsgs_launch_baby_bloom_bucket_emit(
            buffers, offset, count, bucket_mask, add_bloom);
        if (e != cudaSuccess) return e;
        return cudaMemcpy(chunk_entries.data(), buffers.d_baby_entries,
                          (size_t)count * sizeof(BsgsBabyEntry),
                          cudaMemcpyDeviceToHost);
    };

    const auto t_baby0 = ClockLocal::now();
    auto print_baby_ram_progress = [&](const char* label, uint64_t done, ClockLocal::time_point started) {
        const auto now = ClockLocal::now();
        const double elapsed = std::chrono::duration<double>(now - started).count();
        const double speed = elapsed > 0.0 ? (double)done / elapsed / 1e6 : 0.0;
        const double pct = plan.baby_count ? (100.0 * (double)done / (double)plan.baby_count) : 100.0;
        const double eta = speed > 0.0 ? (double)(plan.baby_count - done) / (speed * 1e6) : 0.0;
        const int eh = (int)(eta / 3600.0);
        const int em = (int)((eta - eh * 3600.0) / 60.0);
        const int es = (int)((long long)eta % 60LL);
        std::cout << "\r[BSGS] Baby RAM " << label << " " << std::fixed << std::setprecision(1)
                  << pct << "% | " << speed << " Mentries/s | ETA "
                  << std::setfill('0') << std::setw(2) << eh << ":"
                  << std::setw(2) << em << ":" << std::setw(2) << es
                  << std::setfill(' ') << std::flush;
    };

    std::cout << "[BSGS] Baby RAM pass 1/2: Bloom + bucket counts\n" << std::flush;
    auto t_pass_progress = t_baby0;
    for (uint64_t offset = 0; offset < plan.baby_count; offset += chunk_capacity) {
        const uint64_t count = std::min<uint64_t>(chunk_capacity, plan.baby_count - offset);
        cerr = emit_chunk(offset, count, true);
        if (cerr != cudaSuccess) break;
        for (uint64_t i = 0; i < count; i++)
            bucket_counts[(size_t)(chunk_entries[(size_t)i].bucket & bucket_mask)]++;
        const uint64_t done = offset + count;
        const auto now = ClockLocal::now();
        if (done == plan.baby_count || std::chrono::duration<double>(now - t_pass_progress).count() >= 0.75) {
            print_baby_ram_progress("count", done, t_baby0);
            t_pass_progress = now;
        }
    }
    std::cout << "\n";
    const auto t_count1 = ClockLocal::now();
    if (cerr != cudaSuccess) {
        std::cout << "[BSGS] Baby RAM count pass failed: " << cudaGetErrorString(cerr) << "\n";
        bsgs_free_buffers(buffers);
        return 0;
    }

    std::vector<uint64_t> bucket_offsets((size_t)bucket_count + 1u, 0);
    for (uint32_t i = 0; i < bucket_count; i++)
        bucket_offsets[(size_t)i + 1u] = bucket_offsets[i] + (uint64_t)bucket_counts[i];
    if (bucket_offsets.back() != plan.baby_count) {
        std::cout << "[BSGS] RAM bucket count overflow/mismatch: "
                  << bucket_offsets.back() << " != " << plan.baby_count << "\n";
        bsgs_free_buffers(buffers);
        return 0;
    }

    std::vector<BsgsHostBaby12> ram_entries;
    std::vector<BsgsHostBaby13> ram_entries_wide;
    std::vector<uint64_t> bucket_cursor = bucket_offsets;
    try {
        if (wide_baby_idx)
            ram_entries_wide.resize((size_t)plan.baby_count);
        else
            ram_entries.resize((size_t)plan.baby_count);
    } catch (const std::bad_alloc&) {
        std::cout << "[BSGS] Host RAM allocation failed for " << host_entry_label << " table ("
                  << format_bytes_mb(plan.baby_count * host_entry_size) << ")\n";
        bsgs_free_buffers(buffers);
        return 0;
    }

    std::cout << "[BSGS] Baby RAM pass 2/2: scatter " << host_entry_label << " table\n" << std::flush;
    t_pass_progress = t_count1;
    for (uint64_t offset = 0; offset < plan.baby_count; offset += chunk_capacity) {
        const uint64_t count = std::min<uint64_t>(chunk_capacity, plan.baby_count - offset);
        cerr = emit_chunk(offset, count, false);
        if (cerr != cudaSuccess) break;
        for (uint64_t i = 0; i < count; i++) {
            const BsgsBabyEntry& e = chunk_entries[(size_t)i];
            const uint32_t bucket = e.bucket & bucket_mask;
            const uint64_t slot = bucket_cursor[(size_t)bucket]++;
            if (wide_baby_idx) {
                BsgsHostBaby13& dst = ram_entries_wide[(size_t)slot];
                dst.fp_lo = e.key_fp;
                bsgs_store_idx40(dst.idx_a40, e.idx_a);
            } else {
                ram_entries[(size_t)slot] = {
                    e.key_fp,
                    (uint32_t)e.idx_a
                };
            }
        }
        const uint64_t done = offset + count;
        const auto now = ClockLocal::now();
        if (done == plan.baby_count || std::chrono::duration<double>(now - t_pass_progress).count() >= 0.75) {
            print_baby_ram_progress("scatter", done, t_count1);
            t_pass_progress = now;
        }
    }
    std::cout << "\n";
    const auto t_baby1 = ClockLocal::now();
    if (cerr != cudaSuccess) {
        std::cout << "[BSGS] Baby RAM scatter pass failed: " << cudaGetErrorString(cerr) << "\n";
        bsgs_free_buffers(buffers);
        return 0;
    }
    chunk_entries.clear();
    chunk_entries.shrink_to_fit();

    WifMask verify_mask = {};
    uint8_t verify_base_bytes[WIF_MAX_BYTES] = {};
    uint8_t verify_weights[WIF_MAX_UNKN][WIF_MAX_BYTES] = {};
    const bool verify_cache_ok =
        parse_wif_mask(wif_str, verify_mask, false) &&
        bsgs_wif_precompute_base_weights_cpu(verify_mask, verify_base_bytes, verify_weights);

    auto validate_ram_hits = [&](uint32_t hit_count) -> bool {
        const uint32_t copied = (uint32_t)std::min<uint64_t>((uint64_t)hit_count, buffers.hit_capacity);
        if (copied == 0) return false;

        std::vector<BsgsHit> hits(copied);
        cudaError_t e = cudaMemcpy(hits.data(), buffers.d_hits,
                                   (size_t)copied * sizeof(BsgsHit),
                                   cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            cerr = e;
            return false;
        }

        for (const BsgsHit& hit : hits) {
            const uint64_t fp_lo = bsgs_host_key_fingerprint(hit.lookup_x, hit.lookup_y_parity, 0);
            const uint32_t bucket = bsgs_host_bucket_index_key(hit.lookup_x, hit.lookup_y_parity, 0, bucket_mask);
            const uint64_t begin = bucket_offsets[(size_t)bucket];
            const uint64_t end = bucket_offsets[(size_t)bucket + 1u];
            for (uint64_t i = begin; i < end; i++) {
                uint64_t baby_idx = 0;
                if (wide_baby_idx) {
                    const BsgsHostBaby13& baby = ram_entries_wide[(size_t)i];
                    if (baby.fp_lo != fp_lo) continue;
                    baby_idx = bsgs_load_idx40(baby.idx_a40);
                } else {
                    const BsgsHostBaby12& baby = ram_entries[(size_t)i];
                    if (baby.fp_lo != fp_lo) continue;
                    baby_idx = (uint64_t)baby.idx_a;
                }

                uint8_t key_be[32] = {};
                std::string found_wif;
                const bool checksum_ok = verify_cache_ok && bsgs_reconstruct_wif_candidate_fast_cpu(
                    verify_mask, verify_base_bytes, verify_weights,
                    plan, baby_idx, hit.idx_b, key_be, found_wif);
                const bool pubkey_ok = checksum_ok && bsgs_verify_key_against_target_pubkey(key_be, target);
                if (!pubkey_ok)
                    continue;

                clear_resume_snapshot();
                std::cout << "[BSGS] Giant Bloom-only MATCH via RAM " << host_entry_label << "\n";
                std::cout << "  idx_a=" << baby_idx
                          << " idx_b=" << hit.idx_b
                          << " probe=" << (uint32_t)hit.carry << "\n";
                std::cout << "\n======== BSGS WIF VICTORY ===========================\n";
                std::cout << "WIF         : " << found_wif << "\n";
                std::cout << "Private key : " << hex_key_from_be32(key_be) << "\n";
                std::cout << "idx_a       : " << baby_idx << "\n";
                std::cout << "idx_b       : " << hit.idx_b << "\n";
                std::cout << "======================================================\n";
                notify_victory("BSGS WIF FOUND",
                    "*WIF:*\n`" + found_wif + "`\n\n*Private key:*\n`" + hex_key_from_be32(key_be) + "`",
                    pubkey_str.empty() ? "" : ("*Target:* `" + pubkey_str + "`"));
                return true;
            }
        }
        return false;
    };

    BsgsCandidate zero_candidate = {};
    cerr = cudaMemcpy(buffers.d_candidate, &zero_candidate, sizeof(BsgsCandidate), cudaMemcpyHostToDevice);
    uint32_t zero_hits = 0;
    if (cerr == cudaSuccess)
        cerr = cudaMemcpy(buffers.d_hit_count, &zero_hits, sizeof(uint32_t), cudaMemcpyHostToDevice);

    std::cout << "[BSGS] Giant scheduler: radix-58 tiled Bloom-only ("
              << (BSGS_RADIX58_TILE * BSGS_RADIX58_ROWS_PER_THREAD)
              << " items/thread = 58x" << BSGS_RADIX58_ROWS_PER_THREAD
              << ", 128 threads/block)"
              << " + RAM " << host_entry_label << " resolve\n";
    const auto t_giant0 = ClockLocal::now();
    uint64_t giant_done = 0;
    if (resume && resume->active)
        giant_done = std::min<uint64_t>(resume->offset, plan.giant_count);
    const uint64_t giant_start = giant_done;
    const uint64_t giant_chunk = 1ULL << 26;
    bool victory = false;
    uint64_t total_bloom_hits = 0;
    auto t_progress_last = t_giant0;

    ResumeState rs = make_resume_state("wif_bsgs", wif_str, pubkey_str);
    rs.total = plan.giant_count;
    rs.dict_byte_offset = plan.baby_unknown;
    rs.bsgs_flags = (plan.baby_count > (uint64_t)UINT32_MAX + 1ULL) ? HYDRA_RESUME_BSGS_WIF_BABY55 : 0ULL;
    rs.offset = giant_done;
    rs.tested = giant_done;
    write_resume_snapshot(rs);

    while (cerr == cudaSuccess && !g_sigint && giant_done < plan.giant_count) {
        const uint64_t chunk = std::min<uint64_t>(giant_chunk, plan.giant_count - giant_done);
        zero_hits = 0;
        cerr = cudaMemcpy(buffers.d_hit_count, &zero_hits, sizeof(uint32_t), cudaMemcpyHostToDevice);
        if (cerr != cudaSuccess) break;

        cerr = bsgs_launch_giant_bloom_only_radix58_tiled(
            buffers, giant_done, chunk, plan.wif_shift);
        if (cerr != cudaSuccess) break;

        uint32_t hit_count = 0;
        cerr = cudaMemcpy(&hit_count, buffers.d_hit_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (cerr != cudaSuccess) break;
        total_bloom_hits += hit_count;
        if (validate_ram_hits(hit_count)) {
            victory = true;
            giant_done += chunk;
            break;
        }

        giant_done += chunk;
        rs.offset = giant_done;
        rs.tested = giant_done;
        write_resume_snapshot(rs);

        const auto now = ClockLocal::now();
        const double since = std::chrono::duration<double>(now - t_progress_last).count();
        if (plan.giant_count > giant_chunk && since >= 0.75) {
            const double elapsed = std::chrono::duration<double>(now - t_giant0).count();
            const uint64_t done_this_run = giant_done - giant_start;
            const double speed = elapsed > 0.0 ? (double)done_this_run / elapsed / 1e6 : 0.0;
            const double prog = 100.0 * (double)giant_done / (double)plan.giant_count;
            const double eta = (speed > 0.0)
                ? ((double)(plan.giant_count - giant_done) / (speed * 1e6)) : 0.0;
            const int eh = (int)(eta / 3600.0);
            const int em = (int)((eta - eh * 3600.0) / 60.0);
            const int es = (int)((long long)eta % 60LL);
            std::cout << "\r[BSGS] Giant " << std::fixed << std::setprecision(1)
                      << prog << "% | " << speed << " Msteps/s | ETA "
                      << std::setfill('0') << std::setw(2) << eh << ":"
                      << std::setw(2) << em << ":" << std::setw(2) << es
                      << std::setfill(' ') << std::flush;
            t_progress_last = now;
        }
    }
    const auto t_giant1 = ClockLocal::now();
    if (plan.giant_count > giant_chunk) std::cout << "\n";

    if (cerr != cudaSuccess) {
        std::cout << "[BSGS] Giant Bloom-only failed: " << cudaGetErrorString(cerr) << "\n";
    } else if (!victory && !g_sigint) {
        clear_resume_snapshot();
        std::cout << "[BSGS] Giant Bloom-only: no checksum-valid match\n";
    } else if (g_sigint) {
        rs.offset = giant_done;
        rs.tested = giant_done;
        write_resume_snapshot(rs);
        print_resume_hint();
    }

    const auto t_total1 = ClockLocal::now();
    std::cout << "[BSGS] Timings ms: baby_ram=" << elapsed_ms(t_baby0, t_baby1)
              << " count=" << elapsed_ms(t_baby0, t_count1)
              << " scatter=" << elapsed_ms(t_count1, t_baby1)
              << " giant=" << elapsed_ms(t_giant0, t_giant1)
              << " total=" << elapsed_ms(t_total0, t_total1)
              << " bloom_hits=" << total_bloom_hits << "\n";

    bsgs_free_buffers(buffers);
    return victory ? 0 : 1;
}

static int run_wif_bsgs_mode(
    const std::string& wif_str,
    const std::string& pubkey_str,
    const ResumeState* resume = nullptr,
    const TargetData* resolved_target = nullptr,
    BsgsBabyOverride baby_override = BsgsBabyOverride::AUTO)
{
    using Clock = std::chrono::steady_clock;
    auto elapsed_ms = [](Clock::time_point a, Clock::time_point b) -> double {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    hydra_platform::install_interrupt_handler(&g_sigint);
    const auto t_total0 = Clock::now();

    TargetData target = {};
    if (resolved_target) {
        target = *resolved_target;
        std::cout << "Mode : WIF BSGS + PubKey\n";
    } else if (!resolve_bsgs_pubkey_target(pubkey_str, target, "WIF")) {
        return 1;
    }

    int device_count = 0;
    cudaError_t cerr = cudaGetDeviceCount(&device_count);
    if (cerr != cudaSuccess || device_count <= 0) {
        std::cout << "[BSGS] CUDA unavailable for WIF BSGS run: "
                  << cudaGetErrorString(cerr) << "\n";
        return 0;
    }

    cudaSetDevice(0);
    size_t free_bytes = 0, total_bytes = 0;
    bool have_vram = (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess);
    const uint32_t total_unknown = bsgs_count_wif_unknowns(wif_str);
    const uint64_t vram_budget = have_vram ? hydra_vram_budget_bytes((uint64_t)free_bytes) : 0;
    const bool force_baby5_vram = baby_override == BsgsBabyOverride::WIF_BABY5_VRAM;
    uint32_t baby_unknown = choose_bsgs_wif_baby_split(total_unknown, vram_budget, have_vram);
    if (!force_baby5_vram)
        baby_unknown = bsgs_cap_auto_wif_baby_split(baby_unknown);
    const bool auto_ram_baby =
        bsgs_should_auto_use_wif_ram_baby(total_unknown, have_vram, vram_budget);
    bool force_baby55 = baby_override == BsgsBabyOverride::WIF_BABY55;
    bool use_ram_baby = auto_ram_baby || force_baby55;
    if (resume && resume->active && ((resume->bsgs_flags & HYDRA_RESUME_BSGS_WIF_BABY55) != 0)) {
        force_baby55 = true;
        use_ram_baby = true;
    }
    if (use_ram_baby && total_unknown > 1) {
        baby_unknown = std::min<uint32_t>(5u, total_unknown - 1u);
    }
    bool auto_rebalance_split = true;
    bool baby55_split = false;
    if (force_baby5_vram) {
        if (total_unknown <= 5u) {
            std::cerr << "Error: forced WIF baby5 VRAM requires more than 5 unknown characters.\n";
            return 1;
        }
        baby_unknown = std::min<uint32_t>(5u, total_unknown - 1u);
        use_ram_baby = false;
        auto_rebalance_split = false;
        std::cout << "[Scheduler] WIF baby5 VRAM beta forced by --baby=5-vram\n";
    } else if (force_baby55) {
        baby_unknown = 6u;
        baby55_split = true;
        auto_rebalance_split = false;
    }
    if (resume && resume->active && resume->dict_byte_offset > 0 &&
        resume->dict_byte_offset < (baby55_split ? (total_unknown + 1u) : total_unknown)) {
        baby_unknown = (uint32_t)resume->dict_byte_offset;
        if (!baby55_split)
            auto_rebalance_split = false;
    }
    if (use_ram_baby && !baby55_split && total_unknown > 1 && baby_unknown > 5u) {
        baby_unknown = 5u;
        auto_rebalance_split = true;
    }

    BsgsPlan plan = {};
    if (!build_bsgs_wif_plan_cpu(wif_str, target, baby_unknown, BsgsLookupBackend::BLOOM_BUCKETED, plan, auto_rebalance_split, baby55_split))
        return 1;
    print_bsgs_plan(plan);

    if (use_ram_baby) {
        if (baby55_split)
            std::cout << "[BSGS] WIF RAM baby5.5 mode enabled (forced/experimental)\n";
        else
            std::cout << "[BSGS] WIF RAM baby mode enabled (auto scheduler)\n";
        return run_wif_bsgs_ram_baby_mode(
            wif_str, pubkey_str, target, plan,
            (uint64_t)free_bytes, vram_budget, have_vram, resume, t_total0);
    }

    ResumeState rs = make_resume_state("wif_bsgs", wif_str, pubkey_str);
    rs.total = plan.giant_count;
    rs.dict_byte_offset = plan.baby_unknown;
    if (resume && resume->active) {
        rs.offset = std::min<uint64_t>(resume->offset, plan.giant_count);
        rs.tested = rs.offset;
        std::cout << "[Resume] WIF BSGS giant offset " << rs.offset
                  << " / " << plan.giant_count
                  << " | baby split " << plan.baby_unknown << "\n";
    }

    const uint64_t bloom_bits = bsgs_bloom_bits_for_baby_count(plan.baby_count);
    const uint64_t bloom_mb = bloom_bits / 8ULL / 1024ULL / 1024ULL;
    const uint32_t bucket_bits = bsgs_bucket_bits_for_baby_count(plan.baby_count);
    const uint32_t bucket_count = 1u << bucket_bits;
    const uint64_t hit_capacity = (plan.wif_shift != 0) ? (1ULL << 20) : 65536ULL;
    const uint64_t estimated_bytes = bsgs_estimate_bloom_bucketed_bytes(plan.baby_count);
    if (have_vram) {
        std::cout << "[BSGS] VRAM free: " << format_bytes_mb((uint64_t)free_bytes)
                  << " / budget: " << format_bytes_mb(vram_budget)
                  << " / estimated: " << format_bytes_mb(estimated_bytes) << "\n";
    }
    std::cout << "[BSGS] BloomBucketed filter: " << bloom_mb << " MB ("
              << bloom_bits << " bits, k=" << BLOOM_K_HASHES << ")\n";
    std::cout << "[BSGS] Buckets: " << bucket_count
              << " (2^" << bucket_bits << ", target ~16 entries/bucket)\n";

    BsgsGpuBuffers buffers = {};
    cerr = bsgs_alloc_bloom_bucketed_buffers(
        buffers, plan.baby_count, hit_capacity, bloom_bits, bucket_bits);
    if (cerr != cudaSuccess) {
        std::cout << "[BSGS] BloomBucketed allocation skipped: "
                  << cudaGetErrorString(cerr) << "\n";
        return 0;
    }

    std::cout << "[BSGS] Baby scheduler: radix-58 direct contribution scan\n";
    cerr = bsgs_upload_plan(plan, buffers);
    const auto t_baby0 = Clock::now();
    if (cerr == cudaSuccess)
        cerr = bsgs_launch_baby_bloom_bucket_count(buffers, plan.baby_count);
    const auto t_baby1 = Clock::now();
    if (cerr != cudaSuccess) {
        std::cout << "[BSGS] Baby kernel skipped/failed: "
                  << cudaGetErrorString(cerr) << "\n";
        bsgs_free_buffers(buffers);
        return 0;
    }

    std::vector<uint32_t> bucket_counts((size_t)bucket_count);
    const auto t_copy0 = Clock::now();
    cerr = cudaMemcpy(bucket_counts.data(), buffers.d_bucket_counts,
                      (size_t)bucket_count * sizeof(uint32_t),
                      cudaMemcpyDeviceToHost);
    const auto t_copy1 = Clock::now();
    std::vector<uint32_t> bucket_offsets((size_t)bucket_count + 1u, 0);
    const auto t_prefix0 = Clock::now();
    if (cerr == cudaSuccess) {
        std::cout << "[BSGS] Baby entries built: " << plan.baby_count << "\n";
        for (uint32_t i = 0; i < bucket_count; i++)
            bucket_offsets[(size_t)i + 1u] = bucket_offsets[i] + bucket_counts[i];
    }
    const auto t_prefix1 = Clock::now();
    const auto t_upload0 = Clock::now();
    if (cerr == cudaSuccess) {
        cerr = cudaMemcpy(buffers.d_bucket_offsets, bucket_offsets.data(),
                          ((size_t)bucket_count + 1u) * sizeof(uint32_t),
                          cudaMemcpyHostToDevice);
        if (cerr == cudaSuccess)
            cerr = cudaMemset(buffers.d_bucket_cursor, 0,
                              (size_t)bucket_count * sizeof(uint32_t));
    }
    const auto t_upload1 = Clock::now();
    const auto t_scatter0 = Clock::now();
    if (cerr == cudaSuccess)
        cerr = bsgs_launch_bucket_scatter(buffers, plan.baby_count);
    const auto t_scatter1 = Clock::now();
    if (cerr != cudaSuccess) {
        std::cout << "[BSGS] Baby BloomBucketed GPU bucketization failed: "
                  << cudaGetErrorString(cerr) << "\n";
        bsgs_free_buffers(buffers);
        return 0;
    }
    bsgs_release_raw_baby_entries(buffers);

    BsgsCandidate zero_candidate = {};
    cerr = cudaMemcpy(buffers.d_candidate, &zero_candidate, sizeof(BsgsCandidate), cudaMemcpyHostToDevice);
    std::cout << "[BSGS] Giant scheduler: radix-58 tiled ("
              << (BSGS_RADIX58_TILE * BSGS_RADIX58_ROWS_PER_THREAD)
              << " items/thread = 58x" << BSGS_RADIX58_ROWS_PER_THREAD
              << ", 128 threads/block)"
              << " + batch inversion + shift-template + exact low carry\n";

    bool victory = false;
    bool hit_buffer_full = false;
    uint64_t total_bloom_hits = 0;
    WifMask verify_mask = {};
    uint8_t verify_base_bytes[WIF_MAX_BYTES] = {};
    uint8_t verify_weights[WIF_MAX_UNKN][WIF_MAX_BYTES] = {};
    const bool verify_cache_ok =
        parse_wif_mask(wif_str, verify_mask, false) &&
        bsgs_wif_precompute_base_weights_cpu(verify_mask, verify_base_bytes, verify_weights);

    auto validate_wif_hits = [&](uint32_t hit_count) -> bool {
        const uint32_t copied_hit_count =
            (uint32_t)std::min<uint64_t>((uint64_t)hit_count, buffers.hit_capacity);
        if (copied_hit_count == 0)
            return false;

        std::vector<BsgsHit> hits(copied_hit_count);
        cudaError_t hit_copy_err = cudaMemcpy(hits.data(), buffers.d_hits,
                                              (size_t)copied_hit_count * sizeof(BsgsHit),
                                              cudaMemcpyDeviceToHost);
        if (hit_copy_err != cudaSuccess) {
            std::cout << "[BSGS] WIF hit copy failed: "
                      << cudaGetErrorString(hit_copy_err) << "\n";
            cerr = hit_copy_err;
            return false;
        }

        if ((uint64_t)hit_count > buffers.hit_capacity)
            hit_buffer_full = true;

        for (const BsgsHit& hit : hits) {
            const uint32_t low_carry = hit.carry & 0x0Fu;
            const uint32_t top_carry = (hit.carry >> 4) & 0x0Fu;

            uint8_t key_be[32] = {};
            std::string found_wif;
            const bool checksum_ok = verify_cache_ok && bsgs_reconstruct_wif_candidate_fast_cpu(
                verify_mask, verify_base_bytes, verify_weights,
                plan, hit.idx_a, hit.idx_b, key_be, found_wif);
            const bool pubkey_ok = checksum_ok && bsgs_verify_key_against_target_pubkey(key_be, target);
            if (!pubkey_ok)
                continue;

            clear_resume_snapshot();
            std::cout << "[BSGS] Giant BloomBucketed MATCH\n";
            std::cout << "  idx_a=" << hit.idx_a
                      << " idx_b=" << hit.idx_b
                      << " low_carry=" << low_carry
                      << " top_carry=" << top_carry
                      << " carry_raw=0x" << std::hex << (int)hit.carry << std::dec << "\n";
            std::cout << "\n======== BSGS WIF VICTORY ===========================\n";
            std::cout << "WIF         : " << found_wif << "\n";
            std::cout << "Private key : " << hex_key_from_be32(key_be) << "\n";
            std::cout << "idx_a       : " << hit.idx_a << "\n";
            std::cout << "idx_b       : " << hit.idx_b << "\n";
            std::cout << "low carry   : " << low_carry << "\n";
            std::cout << "top carry   : " << top_carry << "\n";
            std::cout << "carry raw   : 0x" << std::hex << (int)hit.carry << std::dec << "\n";
            std::cout << "======================================================\n";
            return true;
        }
        return false;
    };

    const auto t_giant0 = Clock::now();
    uint64_t giant_done = (resume && resume->active)
        ? std::min<uint64_t>(resume->offset, plan.giant_count) : 0;
    const uint64_t giant_start = giant_done;
    const uint64_t giant_chunk = 1ULL << 24;
    auto t_progress_last = t_giant0;
    bool printed_progress = false;
    if (cerr == cudaSuccess) {
        write_resume_snapshot(rs);
        while (!g_sigint && giant_done < plan.giant_count) {
            const uint64_t chunk = std::min<uint64_t>(giant_chunk, plan.giant_count - giant_done);
            cerr = bsgs_launch_giant_bloom_bucketed_radix58_tiled(
                buffers, giant_done, chunk, plan.wif_shift);
            if (cerr != cudaSuccess) break;

            giant_done += chunk;
            rs.offset = giant_done;
            rs.tested = giant_done;

            uint32_t chunk_hit_count = 0;
            cudaError_t hit_count_err = cudaMemcpy(&chunk_hit_count, buffers.d_hit_count,
                                                   sizeof(uint32_t), cudaMemcpyDeviceToHost);
            if (hit_count_err != cudaSuccess) {
                cerr = hit_count_err;
                break;
            }
            if (chunk_hit_count > 0) {
                total_bloom_hits += (uint64_t)chunk_hit_count;
                if (validate_wif_hits(chunk_hit_count)) {
                    victory = true;
                    break;
                }
                if (cerr != cudaSuccess)
                    break;
                cerr = cudaMemset(buffers.d_hit_count, 0, sizeof(uint32_t));
                if (cerr != cudaSuccess)
                    break;
            }

            BsgsCandidate partial = {};
            cudaError_t copy_err = cudaMemcpy(&partial, buffers.d_candidate,
                                              sizeof(BsgsCandidate), cudaMemcpyDeviceToHost);
            if (copy_err != cudaSuccess) {
                cerr = copy_err;
                break;
            }
            if (partial.found || g_sigint || giant_done >= plan.giant_count)
                write_resume_snapshot(rs);
            if (partial.found) break;
            if (victory) break;

            const auto now = Clock::now();
            const double since = std::chrono::duration<double>(now - t_progress_last).count();
            if (plan.giant_count > giant_chunk && since >= 0.75) {
                const double elapsed = std::chrono::duration<double>(now - t_giant0).count();
                const uint64_t done_this_run = giant_done - giant_start;
                const double speed = elapsed > 0.0 ? (double)done_this_run / elapsed / 1e6 : 0.0;
                const double prog = 100.0 * (double)giant_done / (double)plan.giant_count;
                const double eta = (speed > 0.0)
                    ? ((double)(plan.giant_count - giant_done) / (speed * 1e6)) : 0.0;
                const int eh = (int)(eta / 3600.0);
                const int em = (int)((eta - eh * 3600.0) / 60.0);
                const int es = (int)((long long)eta % 60LL);
                std::cout << "\r[BSGS] Giant " << std::fixed << std::setprecision(1)
                          << prog << "% | " << speed << " Msteps/s | ETA "
                          << std::setfill('0') << std::setw(2) << eh << ":"
                          << std::setw(2) << em << ":" << std::setw(2) << es
                          << std::setfill(' ') << std::flush;
                t_progress_last = now;
                printed_progress = true;
            }
        }
    }
    const auto t_giant1 = Clock::now();
    if (printed_progress) std::cout << "\n";

    if (cerr != cudaSuccess) {
        std::cout << "[BSGS] Giant BloomBucketed kernel skipped/failed: "
                  << cudaGetErrorString(cerr) << "\n";
        bsgs_free_buffers(buffers);
        return 0;
    }
    if (g_sigint) {
        write_resume_snapshot(rs);
        print_resume_hint();
        bsgs_free_buffers(buffers);
        return 0;
    }

    const auto t_total1 = Clock::now();
    std::cout << std::fixed << std::setprecision(3)
              << "[BSGS] Timings ms: baby=" << elapsed_ms(t_baby0, t_baby1)
              << " counts_d2h=" << elapsed_ms(t_copy0, t_copy1)
              << " prefix_cpu=" << elapsed_ms(t_prefix0, t_prefix1)
              << " offsets_h2d=" << elapsed_ms(t_upload0, t_upload1)
              << " scatter=" << elapsed_ms(t_scatter0, t_scatter1)
              << " giant=" << elapsed_ms(t_giant0, t_giant1)
              << " total=" << elapsed_ms(t_total0, t_total1)
              << " bloom_hits=" << total_bloom_hits
              << std::defaultfloat << "\n";

    if (!victory && hit_buffer_full) {
        std::cout << "[BSGS] WIF hit buffer full in at least one chunk; "
                  << "increase hit capacity if checksum-valid hits are missed\n";
    }
    if (victory) {
        bsgs_free_buffers(buffers);
        return 0;
    }

    BsgsCandidate candidate = {};
    cerr = cudaMemcpy(&candidate, buffers.d_candidate, sizeof(BsgsCandidate), cudaMemcpyDeviceToHost);
    if (cerr == cudaSuccess && candidate.found) {
        const uint32_t low_carry = candidate.carry & 0x0Fu;
        const uint32_t top_carry = (candidate.carry >> 4) & 0x0Fu;
        std::cout << "[BSGS] Giant BloomBucketed MATCH\n";
        std::cout << "  idx_a=" << candidate.idx_a
                  << " idx_b=" << candidate.idx_b
                  << " low_carry=" << low_carry
                  << " top_carry=" << top_carry
                  << " carry_raw=0x" << std::hex << (int)candidate.carry << std::dec << "\n";

        uint8_t key_be[32] = {};
        std::string found_wif;
        const bool checksum_ok = verify_cache_ok && bsgs_reconstruct_wif_candidate_fast_cpu(
            verify_mask, verify_base_bytes, verify_weights,
            plan, candidate.idx_a, candidate.idx_b, key_be, found_wif);
        const bool pubkey_ok = checksum_ok && bsgs_verify_key_against_target_pubkey(key_be, target);

        if (pubkey_ok) {
            clear_resume_snapshot();
            std::cout << "\n======== BSGS WIF VICTORY ===========================\n";
            std::cout << "WIF         : " << found_wif << "\n";
            std::cout << "Private key : " << hex_key_from_be32(key_be) << "\n";
            std::cout << "idx_a       : " << candidate.idx_a << "\n";
            std::cout << "idx_b       : " << candidate.idx_b << "\n";
            std::cout << "low carry   : " << low_carry << "\n";
            std::cout << "top carry   : " << top_carry << "\n";
            std::cout << "carry raw   : 0x" << std::hex << (int)candidate.carry << std::dec << "\n";
            std::cout << "======================================================\n";
        } else {
            std::cout << "[BSGS] Candidate failed CPU WIF checksum/pubkey verification\n";
            std::cout << "  checksum=" << (checksum_ok ? "ok" : "failed") << "\n";
        }
    } else if (cerr == cudaSuccess) {
        clear_resume_snapshot();
        if (total_bloom_hits == 0)
            std::cout << "[BSGS] Giant BloomBucketed: no match\n";
        else
            std::cout << "[BSGS] Giant BloomBucketed: no checksum-valid match\n";
    } else {
        std::cout << "[BSGS] Giant BloomBucketed candidate copy failed: "
                  << cudaGetErrorString(cerr) << "\n";
    }

    bsgs_free_buffers(buffers);
    return 0;
}

static uint64_t bsgs_hex_host_bloom_bits(uint64_t baby_count, uint64_t vram_budget)
{
    uint64_t bloom_bits = bsgs_bloom_bits_for_baby_count(baby_count);
    uint64_t requested_mb = 0;
    const uint64_t aux_reserve = 768ULL * 1024ULL * 1024ULL;
    if (vram_budget > aux_reserve) {
        const uint64_t usable = vram_budget - aux_reserve;
        if (usable >= (4ULL << 30)) requested_mb = 4096;
    }
    if (requested_mb != 0) {
        uint64_t bits = requested_mb * 1024ULL * 1024ULL * 8ULL;
        if ((bits & (bits - 1ULL)) != 0) {
            uint64_t pow2 = 1ULL;
            while ((pow2 << 1) && (pow2 << 1) <= bits) pow2 <<= 1;
            bits = pow2;
        }
        if (bits >= (1ULL << 20))
            bloom_bits = bits;
    }
    return bloom_bits;
}

static int run_hex_bsgs_host_baby_mode(
    const std::string& mask_str,
    const std::string& pubkey_str,
    const TargetData& target,
    const BsgsPlan& plan,
    uint64_t free_bytes,
    uint64_t vram_budget,
    bool have_vram,
    const ResumeState* resume,
    const std::chrono::steady_clock::time_point& t_total0,
    bool baby75_split = false,
    bool compact6_mode = true)
{
    using ClockLocal = std::chrono::steady_clock;
    auto elapsed_ms = [](ClockLocal::time_point a, ClockLocal::time_point b) -> double {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    if (plan.radix != 16u || plan.wif_shift != 0u ||
        (!compact6_mode && plan.baby_unknown != 8u) ||
        plan.baby_count > (uint64_t)UINT32_MAX + 1ULL) {
        std::cout << "[BSGS] HEX host backend requires radix16 baby side"
                  << (compact6_mode ? " within idx32 range\n" : " with 8 groups\n");
        return 1;
    }

    const uint64_t bloom_bits = bsgs_hex_host_bloom_bits(plan.baby_count, vram_budget);
    const uint64_t bloom_mb = bloom_bits / 8ULL / 1024ULL / 1024ULL;
    const uint32_t bucket_bits = 24u;
    const uint32_t bucket_count = 1u << bucket_bits;
    const uint32_t bucket_mask = bucket_count - 1u;
    const uint64_t hit_capacity = 1ULL << 22;
    const bool baby75 = baby75_split;
    const char* host_backend_label = compact6_mode
        ? (baby75 ? "RAM Compact6 baby7.5" : "RAM Compact6 Baby8")
        : (baby75 ? "RAM baby7.5" : "RAM Baby8");
    const char* host_table_label = compact6_mode
        ? (baby75 ? "Compact6 baby7.5" : "Compact6 Baby8")
        : (baby75 ? "Baby7.5" : "Baby8");
    const uint64_t host_entry_size = compact6_mode ? sizeof(BsgsHostHex6) : sizeof(BsgsHostHex8);

    uint64_t chunk_capacity = 1ULL << 22;
    if (chunk_capacity > plan.baby_count) chunk_capacity = plan.baby_count;

    const uint64_t estimated_vram =
        bloom_bits / 8ULL +
        chunk_capacity * sizeof(BsgsBabyEntry) +
        hit_capacity * sizeof(BsgsHit) +
        64ULL * 1024ULL * 1024ULL;
    const uint64_t estimated_host =
        ((uint64_t)bucket_count + 1ULL) * sizeof(uint64_t) +
        (uint64_t)bucket_count * (sizeof(uint32_t) + sizeof(uint64_t)) +
        chunk_capacity * sizeof(BsgsBabyEntry) +
        plan.baby_count * host_entry_size;
    const double bits_per_entry =
        plan.baby_count ? (double)bloom_bits / (double)plan.baby_count : 0.0;
    const double fpr_est =
        std::pow(1.0 - std::exp(-(double)BLOOM_K_HASHES / std::max(1.0, bits_per_entry)),
                 (double)BLOOM_K_HASHES);

    if (have_vram) {
        std::cout << "[BSGS] VRAM free: " << format_bytes_mb(free_bytes)
                  << " / budget: " << format_bytes_mb(vram_budget)
                  << " / host-baby estimated VRAM: " << format_bytes_mb(estimated_vram) << "\n";
    }
    std::cout << "[BSGS] " << host_backend_label
              << " table: " << host_table_label << " "
              << format_bytes_mb(plan.baby_count * host_entry_size)
              << " (" << host_entry_size << " bytes/entry, "
              << (compact6_mode ? "idx32 + fp16" : "fp32 + idx32") << ")\n";
    std::cout << "[BSGS] HEX " << host_backend_label << " enabled\n";
    std::cout << "[BSGS] Bloom-only GPU filter: " << bloom_mb << " MB ("
              << bloom_bits << " bits, k=" << BLOOM_K_HASHES
              << ", m/n=" << std::fixed << std::setprecision(1) << bits_per_entry
              << ", fpr~" << std::scientific << std::setprecision(2) << fpr_est
              << std::defaultfloat << ")\n";
    std::cout << "[BSGS] RAM buckets: " << bucket_count
              << " (2^" << bucket_bits << ", avg ~"
              << std::fixed << std::setprecision(1)
              << ((double)plan.baby_count / (double)bucket_count)
              << " entries/bucket)\n" << std::defaultfloat;
    std::cout << "[BSGS] Baby RAM chunk: " << chunk_capacity << " entries ("
              << format_bytes_mb(chunk_capacity * sizeof(BsgsBabyEntry)) << " GPU temp)\n";
    std::cout << "[BSGS] Host RAM side buffers: " << format_bytes_mb(estimated_host) << "\n";

    BsgsGpuBuffers buffers = {};
    cudaError_t cerr = bsgs_alloc_bloom_buffers(buffers, chunk_capacity, hit_capacity, bloom_bits);
    if (cerr != cudaSuccess) {
        std::cout << "[BSGS] Host-baby GPU allocation skipped: "
                  << cudaGetErrorString(cerr) << "\n";
        return 0;
    }
    cerr = bsgs_upload_plan(plan, buffers);
    if (cerr != cudaSuccess) {
        std::cout << "[BSGS] Plan upload failed: " << cudaGetErrorString(cerr) << "\n";
        bsgs_free_buffers(buffers);
        return 0;
    }

    std::vector<BsgsBabyEntry> chunk_entries((size_t)chunk_capacity);
    std::vector<uint32_t> bucket_counts((size_t)bucket_count, 0);
    auto emit_chunk = [&](uint64_t offset, uint64_t count, bool add_bloom) -> cudaError_t {
        cudaError_t e = bsgs_launch_baby_bloom_bucket_emit(
            buffers, offset, count, bucket_mask, add_bloom);
        if (e != cudaSuccess) return e;
        return cudaMemcpy(chunk_entries.data(), buffers.d_baby_entries,
                          (size_t)count * sizeof(BsgsBabyEntry),
                          cudaMemcpyDeviceToHost);
    };

    auto print_progress = [&](const char* label, uint64_t done, ClockLocal::time_point started) {
        const auto now = ClockLocal::now();
        const double elapsed = std::chrono::duration<double>(now - started).count();
        const double speed = elapsed > 0.0 ? (double)done / elapsed / 1e6 : 0.0;
        const double pct = plan.baby_count ? 100.0 * (double)done / (double)plan.baby_count : 100.0;
        const double eta = speed > 0.0 ? (double)(plan.baby_count - done) / (speed * 1e6) : 0.0;
        const int eh = (int)(eta / 3600.0);
        const int em = (int)((eta - eh * 3600.0) / 60.0);
        const int es = (int)((long long)eta % 60LL);
        std::cout << "\r[BSGS] Baby RAM"
                  << " " << label << " " << std::fixed << std::setprecision(1)
                  << pct << "% | " << speed << " Mentries/s | ETA "
                  << std::setfill('0') << std::setw(2) << eh << ":"
                  << std::setw(2) << em << ":" << std::setw(2) << es
                  << std::setfill(' ') << std::flush;
    };

    const auto t_baby0 = ClockLocal::now();
    std::cout << "[BSGS] Baby RAM"
              << " pass 1/2: Bloom + bucket counts\n" << std::flush;
    auto t_progress_last = t_baby0;
    for (uint64_t offset = 0; offset < plan.baby_count; offset += chunk_capacity) {
        const uint64_t count = std::min<uint64_t>(chunk_capacity, plan.baby_count - offset);
        cerr = emit_chunk(offset, count, true);
        if (cerr != cudaSuccess) break;
        for (uint64_t i = 0; i < count; i++)
            bucket_counts[(size_t)(chunk_entries[(size_t)i].bucket & bucket_mask)]++;
        const uint64_t done = offset + count;
        const auto now = ClockLocal::now();
        if (done == plan.baby_count || std::chrono::duration<double>(now - t_progress_last).count() >= 0.75) {
            print_progress("count", done, t_baby0);
            t_progress_last = now;
        }
    }
    std::cout << "\n";
    const auto t_count1 = ClockLocal::now();
    if (cerr != cudaSuccess) {
        std::cout << "[BSGS] Baby RAM count pass failed: " << cudaGetErrorString(cerr) << "\n";
        bsgs_free_buffers(buffers);
        return 0;
    }

    std::vector<uint64_t> bucket_offsets((size_t)bucket_count + 1u, 0);
    for (uint32_t i = 0; i < bucket_count; i++)
        bucket_offsets[(size_t)i + 1u] = bucket_offsets[i] + bucket_counts[i];
    if (bucket_offsets.back() != plan.baby_count) {
        std::cout << "[BSGS] RAM bucket count mismatch: "
                  << bucket_offsets.back() << " != " << plan.baby_count << "\n";
        bsgs_free_buffers(buffers);
        return 0;
    }

    std::vector<BsgsHostHex8> ram_table8;
    std::vector<BsgsHostHex6> ram_table6;
    BsgsHostHex8* baby_table8_data = nullptr;
    BsgsHostHex6* baby_table6_data = nullptr;
    try {
        if (compact6_mode) {
            ram_table6.resize((size_t)plan.baby_count);
            baby_table6_data = ram_table6.data();
        } else {
            ram_table8.resize((size_t)plan.baby_count);
            baby_table8_data = ram_table8.data();
        }
    } catch (const std::bad_alloc&) {
        std::cout << "[BSGS] Host RAM allocation failed for " << host_table_label << " table ("
                  << format_bytes_mb(plan.baby_count * host_entry_size) << ")\n";
        bsgs_free_buffers(buffers);
        return 0;
    }

    uint32_t bucket_size_max = 0;
    uint64_t nonempty_buckets = 0;
    for (uint32_t c : bucket_counts) {
        if (c) {
            nonempty_buckets++;
            if (c > bucket_size_max) bucket_size_max = c;
        }
    }

    std::vector<uint64_t> bucket_cursor = bucket_offsets;
    std::cout << "[BSGS] Baby RAM"
              << " pass 2/2: scatter "
              << host_table_label << " table"
              << "\n" << std::flush;
    t_progress_last = t_count1;
    for (uint64_t offset = 0; offset < plan.baby_count; offset += chunk_capacity) {
        const uint64_t count = std::min<uint64_t>(chunk_capacity, plan.baby_count - offset);
        cerr = emit_chunk(offset, count, false);
        if (cerr != cudaSuccess) break;
        for (uint64_t i = 0; i < count; i++) {
            const BsgsBabyEntry& e = chunk_entries[(size_t)i];
            const uint32_t bucket = e.bucket & bucket_mask;
            const uint64_t slot = bucket_cursor[(size_t)bucket]++;
            const uint32_t key_fp = (uint32_t)e.key_fp ^ (uint32_t)(e.key_fp >> 32) ^ e.key_fp_hi;
            if (compact6_mode) {
                baby_table6_data[slot] = { (uint32_t)e.idx_a, (uint16_t)key_fp };
            } else {
                baby_table8_data[slot] = {
                    key_fp,
                    (uint32_t)e.idx_a
                };
            }
        }
        const uint64_t done = offset + count;
        const auto now = ClockLocal::now();
        if (done == plan.baby_count || std::chrono::duration<double>(now - t_progress_last).count() >= 0.75) {
            print_progress("scatter", done, t_count1);
            t_progress_last = now;
        }
    }
    std::cout << "\n";
    const auto t_baby1 = ClockLocal::now();
    if (cerr != cudaSuccess) {
        std::cout << "[BSGS] Baby RAM scatter pass failed: " << cudaGetErrorString(cerr) << "\n";
        bsgs_free_buffers(buffers);
        return 0;
    }
    chunk_entries.clear();
    chunk_entries.shrink_to_fit();

    uint64_t host_bucket_scans = 0;
    uint64_t host_entries_scanned = 0;
    uint64_t host_fp_matches = 0;
    uint64_t host_verify_attempts = 0;
    uint64_t host_pubkey_fail = 0;
    double host_resolve_ms = 0.0;

    auto validate_hex_host_hits = [&](uint32_t hit_count, bool& hit_buffer_full) -> bool {
        const uint32_t copied = (uint32_t)std::min<uint64_t>((uint64_t)hit_count, buffers.hit_capacity);
        if ((uint64_t)hit_count > buffers.hit_capacity)
            hit_buffer_full = true;
        if (copied == 0) return false;
        const auto t_resolve0 = ClockLocal::now();
        std::vector<BsgsHit> hits(copied);
        cudaError_t e = cudaMemcpy(hits.data(), buffers.d_hits,
                                   (size_t)copied * sizeof(BsgsHit),
                                   cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            cerr = e;
            return false;
        }
        for (const BsgsHit& hit : hits) {
            const uint32_t key_fp = bsgs_host_hex8_fingerprint_key(hit.lookup_x, hit.lookup_y_parity, 0);
            const uint32_t bucket = bsgs_host_bucket_index_key(hit.lookup_x, hit.lookup_y_parity, 0, bucket_mask);
            const uint64_t begin = bucket_offsets[(size_t)bucket];
            const uint64_t end = bucket_offsets[(size_t)bucket + 1u];
            host_bucket_scans++;
            host_entries_scanned += (end - begin);
            for (uint64_t i = begin; i < end; i++) {
                uint32_t idx_a = 0;
                if (compact6_mode) {
                    const BsgsHostHex6& baby = baby_table6_data[i];
                    if (baby.key_fp16 != (uint16_t)key_fp) continue;
                    host_fp_matches++;
                    idx_a = baby.idx_a;
                } else {
                    const BsgsHostHex8& baby = baby_table8_data[i];
                    if (baby.key_fp != key_fp) continue;
                    host_fp_matches++;
                    idx_a = baby.idx_a;
                }

                uint8_t key_be[32] = {};
                bsgs_reconstruct_hex_key_cpu(plan, (uint64_t)idx_a, hit.idx_b, key_be);
                host_verify_attempts++;
                if (!bsgs_verify_key_against_target_pubkey(key_be, target)) {
                    host_pubkey_fail++;
                    continue;
                }

                clear_resume_snapshot();
                const std::string private_hex = hex_key_from_be32(key_be);
                std::cout << "[BSGS] Giant Bloom-only MATCH via " << host_backend_label << "\n";
                std::cout << "  idx_a=" << (uint64_t)idx_a
                          << " idx_b=" << hit.idx_b << "\n";
                std::cout << "\n======== BSGS VICTORY ===============================\n";
                std::cout << "Private key : " << private_hex << "\n";
                std::cout << "idx_a       : " << (uint64_t)idx_a << "\n";
                std::cout << "idx_b       : " << hit.idx_b << "\n";
                std::cout << "carry       : 0\n";
                std::cout << "======================================================\n";
                notify_victory("BSGS HEX FOUND",
                    "*Private key:*\n`" + private_hex + "`",
                    pubkey_str.empty() ? "" : ("*Target:* `" + pubkey_str + "`"));
                const auto t_resolve1 = ClockLocal::now();
                host_resolve_ms += elapsed_ms(t_resolve0, t_resolve1);
                return true;
            }
        }
        const auto t_resolve1 = ClockLocal::now();
        host_resolve_ms += elapsed_ms(t_resolve0, t_resolve1);
        return false;
    };

    uint32_t zero_hits = 0;
    cerr = cudaMemcpy(buffers.d_hit_count, &zero_hits, sizeof(uint32_t), cudaMemcpyHostToDevice);
    const bool use_tiled_giant = (plan.giant_low_bits == (uint32_t)LOW_BITS);
    std::cout << "[BSGS] Giant scheduler: HEX Bloom-only "
              << (use_tiled_giant ? "tiled Gray + " : "binary Gray + ")
              << host_backend_label << " resolve\n";

    const auto t_giant0 = ClockLocal::now();
    uint64_t giant_done = (resume && resume->active)
        ? std::min<uint64_t>(resume->offset, plan.giant_count) : 0;
    const uint64_t giant_start = giant_done;
    const uint64_t giant_chunk = use_tiled_giant ? (1ULL << 22) : (1ULL << 22);
    uint64_t total_bloom_hits = 0;
    bool victory = false;
    bool hit_buffer_full = false;
    t_progress_last = t_giant0;

    ResumeState rs = make_resume_state("hex_bsgs", mask_str, pubkey_str);
    rs.total = plan.giant_count;
    rs.dict_byte_offset = plan.baby_unknown;
    rs.bsgs_flags = HYDRA_RESUME_BSGS_HOST_BABY |
        (baby75_split ? HYDRA_RESUME_BSGS_HEX_BABY75 : 0ULL) |
        (compact6_mode ? HYDRA_RESUME_BSGS_HEX_COMPACT6 : 0ULL);
    rs.offset = giant_done;
    rs.tested = giant_done;
    write_resume_snapshot(rs);

    while (cerr == cudaSuccess && !g_sigint && giant_done < plan.giant_count) {
        const uint64_t chunk = std::min<uint64_t>(giant_chunk, plan.giant_count - giant_done);
        zero_hits = 0;
        cerr = cudaMemcpy(buffers.d_hit_count, &zero_hits, sizeof(uint32_t), cudaMemcpyHostToDevice);
        if (cerr != cudaSuccess) break;
        cerr = use_tiled_giant
            ? bsgs_launch_giant_bloom_only_hex_tiled(buffers, giant_done, chunk, 128)
            : bsgs_launch_giant_bloom_only_hex_gray(buffers, giant_done, chunk, 256);
        if (cerr != cudaSuccess) break;

        uint32_t hit_count = 0;
        cerr = cudaMemcpy(&hit_count, buffers.d_hit_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (cerr != cudaSuccess) break;
        total_bloom_hits += hit_count;
        if (validate_hex_host_hits(hit_count, hit_buffer_full)) {
            victory = true;
            giant_done += chunk;
            break;
        }

        giant_done += chunk;
        rs.offset = giant_done;
        rs.tested = giant_done;
        write_resume_snapshot(rs);

        const auto now = ClockLocal::now();
        const double since = std::chrono::duration<double>(now - t_progress_last).count();
        if (plan.giant_count > giant_chunk && since >= 0.75) {
            const double elapsed = std::chrono::duration<double>(now - t_giant0).count();
            const uint64_t done_this_run = giant_done - giant_start;
            const double speed = elapsed > 0.0 ? (double)done_this_run / elapsed / 1e6 : 0.0;
            const double prog = 100.0 * (double)giant_done / (double)plan.giant_count;
            const double eta = speed > 0.0
                ? ((double)(plan.giant_count - giant_done) / (speed * 1e6)) : 0.0;
            const int eh = (int)(eta / 3600.0);
            const int em = (int)((eta - eh * 3600.0) / 60.0);
            const int es = (int)((long long)eta % 60LL);
            std::cout << "\r[BSGS] Giant " << std::fixed << std::setprecision(1)
                      << prog << "% | " << speed << " Msteps/s | ETA "
                      << std::setfill('0') << std::setw(2) << eh << ":"
                      << std::setw(2) << em << ":" << std::setw(2) << es
                      << std::setfill(' ') << std::flush;
            t_progress_last = now;
        }
    }
    const auto t_giant1 = ClockLocal::now();
    if (plan.giant_count > giant_chunk) std::cout << "\n";

    if (cerr != cudaSuccess) {
        std::cout << "[BSGS] HEX host-baby giant failed: " << cudaGetErrorString(cerr) << "\n";
    } else if (g_sigint) {
        rs.offset = giant_done;
        rs.tested = giant_done;
        write_resume_snapshot(rs);
        print_resume_hint();
    } else if (!victory) {
        clear_resume_snapshot();
        std::cout << "[BSGS] HEX host-baby Bloom-only: no pubkey-valid match\n";
    }
    if (hit_buffer_full) {
        std::cout << "[BSGS] HEX host-baby hit buffer full in at least one chunk; "
                  << "use a larger Bloom or smaller giant chunk if matches are missed\n";
    }

    const auto t_total1 = ClockLocal::now();
    std::cout << "[BSGS] Timings ms: baby_host=" << elapsed_ms(t_baby0, t_baby1)
              << " count=" << elapsed_ms(t_baby0, t_count1)
              << " scatter=" << elapsed_ms(t_count1, t_baby1)
              << " giant=" << elapsed_ms(t_giant0, t_giant1)
              << " total=" << elapsed_ms(t_total0, t_total1)
              << " bloom_hits=" << total_bloom_hits << "\n";
    std::cout << "[BSGS][host] bucket_scans=" << host_bucket_scans
              << " entries_scanned=" << host_entries_scanned
              << " fp_matches=" << host_fp_matches
              << " verify_attempts=" << host_verify_attempts
              << " pubkey_fail=" << host_pubkey_fail
              << " resolve_ms=" << host_resolve_ms
              << " bucket_avg=" << std::fixed << std::setprecision(1)
              << ((double)plan.baby_count / (double)bucket_count)
              << " bucket_nonempty=" << nonempty_buckets
              << " bucket_max=" << bucket_size_max
              << std::defaultfloat << "\n";

    bsgs_free_buffers(buffers);
    return victory ? 0 : (g_sigint ? 0 : 1);
}

static int run_hex_bsgs_execute_mode(
    const std::string& mask_str,
    const std::string& pubkey_str,
    const ResumeState* resume = nullptr,
    const TargetData* resolved_target = nullptr,
    BsgsBabyOverride baby_override = BsgsBabyOverride::AUTO)
{
    using Clock = std::chrono::steady_clock;
    auto elapsed_ms = [](Clock::time_point a, Clock::time_point b) -> double {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    hydra_platform::install_interrupt_handler(&g_sigint);
    const auto t_total0 = Clock::now();

    TargetData target = {};
    if (resolved_target) {
        target = *resolved_target;
        std::cout << "Mode : HEX BSGS + PubKey\n";
    } else if (!resolve_bsgs_pubkey_target(pubkey_str, target)) {
        return 1;
    }

    int device_count = 0;
    cudaError_t cerr = cudaGetDeviceCount(&device_count);
    if (cerr != cudaSuccess || device_count <= 0) {
        std::cout << "[BSGS] CUDA unavailable for BloomBucketed run: "
                  << cudaGetErrorString(cerr) << "\n";
        return 0;
    }

    cudaSetDevice(0);
    size_t free_bytes = 0, total_bytes = 0;
    bool have_vram = (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess);
    const uint32_t total_unknown = bsgs_count_hex_unknowns(mask_str);
    const uint64_t vram_budget = have_vram ? hydra_vram_budget_bytes((uint64_t)free_bytes) : 0;
    bool use_host_baby_backend = false;
    bool use_baby75_split = false;
    bool auto_ram_baby75 = false;
    bool compact6_mode = false;

    uint32_t baby_unknown = choose_bsgs_hex_baby_split(total_unknown, vram_budget, have_vram);
    if (baby_unknown > 7u)
        baby_unknown = 7u; // baby8 is opt-in only; auto may upgrade to tested baby7.5 below.
    const bool resume_host_baby =
        resume && resume->active && ((resume->bsgs_flags & HYDRA_RESUME_BSGS_HOST_BABY) != 0);
    const bool resume_hex75_vram =
        resume && resume->active &&
        ((resume->bsgs_flags & HYDRA_RESUME_BSGS_HEX_BABY75) != 0) &&
        !resume_host_baby;
    const bool host_baby_scale = total_unknown >= 17u;
    if (baby_override == BsgsBabyOverride::HEX_BABY75_VRAM && total_unknown > 8u) {
        baby_unknown = 8u;
        use_baby75_split = true;
        std::cout << "[Scheduler] HEX baby7.5 VRAM beta forced by --baby=7.5-vram\n";
    } else if (baby_override == BsgsBabyOverride::HEX_BABY8 && total_unknown > 8u) {
        baby_unknown = 8u;
        use_host_baby_backend = true;
        compact6_mode = true;
        std::cout << "[Scheduler] HEX Compact6 Baby8 forced by --baby=8\n";
    } else if (resume_host_baby && total_unknown > 8u) {
        baby_unknown = 8u;
        use_host_baby_backend = true;
        use_baby75_split = (resume->bsgs_flags & HYDRA_RESUME_BSGS_HEX_BABY75) != 0;
        compact6_mode = true;
    } else if (!resume_hex75_vram && host_baby_scale && have_vram) {
        const uint64_t baby75_count = 1ULL << 30;
        const uint64_t baby75_vram_need = bsgs_estimate_hex_host_baby_vram_bytes(baby75_count, vram_budget);
        const uint64_t baby75_host_need = bsgs_estimate_hex_host_baby_host_bytes(baby75_count);
        if (baby75_vram_need <= vram_budget &&
            bsgs_host_ram_budget_allows(baby75_host_need)) {
            baby_unknown = 8u;
            use_host_baby_backend = true;
            use_baby75_split = true;
            auto_ram_baby75 = true;
            compact6_mode = true;
        }
    }
    if (baby_override == BsgsBabyOverride::HEX_BABY75_VRAM && !use_baby75_split) {
        std::cerr << "Error: forced HEX baby7.5 VRAM split requires more than 8 unknown nibbles.\n";
        return 1;
    }
    if (baby_override == BsgsBabyOverride::HEX_BABY8 && !use_host_baby_backend) {
        std::cerr << "Error: forced HEX baby split requires more than 8 unknown nibbles.\n";
        return 1;
    }

    if (resume && resume->active && !use_host_baby_backend &&
        resume->dict_byte_offset > 0 && resume->dict_byte_offset < total_unknown) {
        baby_unknown = (uint32_t)resume->dict_byte_offset;
        if ((resume->bsgs_flags & HYDRA_RESUME_BSGS_HEX_BABY75) != 0)
            use_baby75_split = true;
    }

    BsgsPlan plan = {};
    if (!build_bsgs_hex_plan_cpu(
            mask_str, target, baby_unknown, BsgsLookupBackend::BLOOM_BUCKETED,
            plan, use_baby75_split))
        return 1;

    print_bsgs_plan(plan);

    if (use_host_baby_backend && plan.baby_unknown == 8u) {
        if (resume_host_baby)
            std::cout << "[Resume] HEX host baby backend restored"
                      << (use_baby75_split ? " (baby7.5" : " (Baby8")
                      << (compact6_mode ? ", Compact6)\n" : ")\n");
        else if (auto_ram_baby75)
            std::cout << "[Scheduler] HEX Compact6 baby7.5 selected in host RAM (tested auto split)\n";
        return run_hex_bsgs_host_baby_mode(
            mask_str, pubkey_str, target, plan,
            (uint64_t)free_bytes, vram_budget, have_vram, resume, t_total0,
            use_baby75_split, compact6_mode);
    }

    ResumeState rs = make_resume_state("hex_bsgs", mask_str, pubkey_str);
    rs.total = plan.giant_count;
    rs.dict_byte_offset = plan.baby_unknown; // keep split stable across resume
    if (use_baby75_split && !use_host_baby_backend)
        rs.bsgs_flags |= HYDRA_RESUME_BSGS_HEX_BABY75;
    if (resume && resume->active) {
        rs.offset = std::min<uint64_t>(resume->offset, plan.giant_count);
        rs.tested = rs.offset;
        std::cout << "[Resume] BSGS giant offset " << rs.offset
                  << " / " << plan.giant_count
                  << " | baby split " << plan.baby_unknown << "\n";
    }

    const bool hidden_hex75_vram = use_baby75_split && !use_host_baby_backend;
    const uint64_t hidden_hex75_bloom_bits = 4096ULL * 1024ULL * 1024ULL * 8ULL;
    const uint64_t bloom_bits = hidden_hex75_vram
        ? hidden_hex75_bloom_bits
        : bsgs_bloom_bits_for_baby_count(plan.baby_count);
    const uint64_t bloom_mb = bloom_bits / 8ULL / 1024ULL / 1024ULL;
    const uint32_t bucket_bits = bsgs_bucket_bits_for_baby_count(plan.baby_count);
    const uint32_t bucket_count = 1u << bucket_bits;
    const uint64_t hit_capacity = 65536;
    const bool use_hex8 = (plan.radix == 16u && plan.wif_shift == 0u &&
                           plan.baby_count <= (uint64_t)UINT32_MAX + 1ULL);
    const uint64_t estimated_bytes = use_hex8
        ? bsgs_estimate_hex8_bucketed_bytes_with_bloom(plan.baby_count, bloom_bits)
        : bsgs_estimate_bloom_bucketed_bytes(plan.baby_count);
    if (have_vram) {
        std::cout << "[BSGS] VRAM free: " << format_bytes_mb((uint64_t)free_bytes)
                  << " / budget: " << format_bytes_mb(vram_budget)
                  << " / estimated: " << format_bytes_mb(estimated_bytes) << "\n";
    } else {
        std::cout << "[BSGS] VRAM estimate: " << format_bytes_mb(estimated_bytes) << "\n";
    }
    std::cout << "[BSGS] BloomBucketed filter: " << bloom_mb << " MB ("
              << bloom_bits << " bits, k=" << BLOOM_K_HASHES << ")\n";
    if (hidden_hex75_vram) {
        std::cout << "[BSGS] Hidden beta: HEX baby7.5 VRAM with reduced Bloom "
                  << "(expect more Bloom probes than default)\n";
    }
    if (use_hex8) {
        std::cout << "[BSGS] HEX Hex8 table: "
                  << format_bytes_mb(plan.baby_count * sizeof(BsgsHex8Entry))
                  << " (" << sizeof(BsgsHex8Entry)
                  << " bytes/entry, fp32 + idx32)\n";
    }
    std::cout << "[BSGS] Buckets: " << bucket_count
              << " (2^" << bucket_bits << ", target ~16 entries/bucket)\n";

    BsgsGpuBuffers buffers = {};
    if (use_hex8) {
        cerr = bsgs_alloc_hex8_bucketed_buffers(
            buffers, plan.baby_count, hit_capacity, bloom_bits, bucket_bits);
    } else {
        cerr = bsgs_alloc_bloom_bucketed_buffers(
            buffers, plan.baby_count, hit_capacity, bloom_bits, bucket_bits);
    }
    if (cerr != cudaSuccess) {
        std::cout << "[BSGS] BloomBucketed allocation skipped: "
                  << cudaGetErrorString(cerr) << "\n";
        return 0;
    }

    const bool use_tiled_baby = (plan.baby_low_bits == (uint32_t)LOW_BITS);
    std::cout << "[BSGS] Baby scheduler: ";
    if (use_hex8) {
        std::cout << "HEX Hex8 compact two-pass direct contribution scan\n";
    } else if (use_tiled_baby) {
        std::cout << "tiled + Montgomery batch inversion"
                  << " (low=" << plan.baby_low_bits << " bits)\n";
    } else {
        std::cout << "direct contribution scan\n";
    }

    cerr = bsgs_upload_plan(plan, buffers);
    const auto t_baby0 = Clock::now();
    if (cerr == cudaSuccess) {
        if (use_hex8)
            cerr = bsgs_launch_hex8_baby_count(buffers, plan.baby_count);
        else if (use_tiled_baby)
            cerr = bsgs_launch_baby_bloom_bucket_count_tiled(buffers, plan.baby_count);
        else
            cerr = bsgs_launch_baby_bloom_bucket_count(buffers, plan.baby_count);
    }
    const auto t_baby1 = Clock::now();

    if (cerr != cudaSuccess) {
        std::cout << "[BSGS] Baby kernel skipped/failed: "
                  << cudaGetErrorString(cerr) << "\n";
        bsgs_free_buffers(buffers);
        return 0;
    }

    std::vector<uint32_t> bucket_counts((size_t)bucket_count);
    const auto t_copy0 = Clock::now();
    cerr = cudaMemcpy(bucket_counts.data(), buffers.d_bucket_counts,
                      (size_t)bucket_count * sizeof(uint32_t),
                      cudaMemcpyDeviceToHost);
    const auto t_copy1 = Clock::now();
    auto t_prefix0 = Clock::now();
    auto t_prefix1 = t_prefix0;
    auto t_upload0 = t_prefix0;
    auto t_upload1 = t_prefix0;
    auto t_scatter0 = t_prefix0;
    auto t_scatter1 = t_prefix0;
    std::vector<uint32_t> bucket_offsets((size_t)bucket_count + 1u, 0);
    if (cerr == cudaSuccess) {
        std::cout << "[BSGS] Baby entries built: " << plan.baby_count << "\n";
        t_prefix0 = Clock::now();
        for (uint32_t i = 0; i < bucket_count; i++) {
            bucket_offsets[(size_t)i + 1u] = bucket_offsets[i] + bucket_counts[i];
        }
        t_prefix1 = Clock::now();
        t_upload0 = Clock::now();
        cerr = cudaMemcpy(buffers.d_bucket_offsets, bucket_offsets.data(),
                          ((size_t)bucket_count + 1u) * sizeof(uint32_t),
                          cudaMemcpyHostToDevice);
        if (cerr == cudaSuccess) {
            cerr = cudaMemset(buffers.d_bucket_cursor, 0,
                              (size_t)bucket_count * sizeof(uint32_t));
        }
        t_upload1 = Clock::now();
        t_scatter0 = Clock::now();
        if (cerr == cudaSuccess) {
            if (use_hex8)
                cerr = bsgs_launch_hex8_baby_scatter(buffers, plan.baby_count);
            else
                cerr = bsgs_launch_bucket_scatter(buffers, plan.baby_count);
        }
        t_scatter1 = Clock::now();
    }

    if (cerr != cudaSuccess) {
        std::cout << "[BSGS] Baby BloomBucketed GPU bucketization failed: "
                  << cudaGetErrorString(cerr) << "\n";
        bsgs_free_buffers(buffers);
        return 0;
    }
    if (!use_hex8)
        bsgs_release_raw_baby_entries(buffers);
    BsgsCandidate zero_candidate = {};
    cerr = cudaMemcpy(buffers.d_candidate, &zero_candidate, sizeof(BsgsCandidate), cudaMemcpyHostToDevice);
    const bool use_tiled_giant = (plan.giant_low_bits == (uint32_t)LOW_BITS);
    std::cout << "[BSGS] Giant scheduler: ";
    if (use_tiled_giant) {
        std::cout << "tiled Gray + Montgomery batch inversion"
                  << " (low=" << plan.giant_low_bits << " bits)\n";
    } else {
        std::cout << "binary Gray (" << BSGS_GRAY_ITEMS_PER_THREAD
                  << " items/thread)\n";
    }
    const auto t_giant0 = Clock::now();
    uint64_t giant_done = 0;
    if (resume && resume->active)
        giant_done = std::min<uint64_t>(resume->offset, plan.giant_count);
    const uint64_t giant_start = giant_done;
    const uint64_t giant_chunk = 1ULL << 26; // ~0.5s on RTX 5060 for current kernels
    auto t_progress_last = t_giant0;
    auto t_resume_last = t_giant0;
    bool printed_progress = false;
    if (cerr == cudaSuccess) {
        write_resume_snapshot(rs);
        while (!g_sigint && giant_done < plan.giant_count) {
            const uint64_t chunk = std::min<uint64_t>(giant_chunk, plan.giant_count - giant_done);
            if (use_tiled_giant) {
                cerr = use_hex8
                    ? bsgs_launch_giant_hex8_tiled(buffers, giant_done, chunk)
                    : bsgs_launch_giant_bloom_bucketed_tiled(buffers, giant_done, chunk);
            } else {
                cerr = use_hex8
                    ? bsgs_launch_giant_hex8_gray(buffers, giant_done, chunk)
                    : bsgs_launch_giant_bloom_bucketed_gray(buffers, giant_done, chunk);
            }
            if (cerr != cudaSuccess) break;

            giant_done += chunk;
            rs.offset = giant_done;
            rs.tested = giant_done;

            BsgsCandidate partial = {};
            cudaError_t copy_err = cudaMemcpy(&partial, buffers.d_candidate,
                                              sizeof(BsgsCandidate), cudaMemcpyDeviceToHost);
            if (copy_err != cudaSuccess) {
                cerr = copy_err;
                break;
            }
            const bool force_resume = partial.found || g_sigint || (giant_done >= plan.giant_count);
            const auto now_after_chunk = Clock::now();
            if (should_write_resume_snapshot(t_resume_last, now_after_chunk, force_resume)) {
                write_resume_snapshot(rs);
                t_resume_last = now_after_chunk;
            }
            if (partial.found) break;

            const auto now = now_after_chunk;
            const double since = std::chrono::duration<double>(now - t_progress_last).count();
            if (plan.giant_count > giant_chunk && since >= 0.75) {
                const double elapsed = std::chrono::duration<double>(now - t_giant0).count();
                const uint64_t giant_done_this_run = giant_done - giant_start;
                const double speed = elapsed > 0.0 ? (double)giant_done_this_run / elapsed / 1e6 : 0.0;
                const double prog = 100.0 * (double)giant_done / (double)plan.giant_count;
                const double eta = (speed > 0.0)
                    ? ((double)(plan.giant_count - giant_done) / (speed * 1e6)) : 0.0;
                const int eh = (int)(eta / 3600.0);
                const int em = (int)((eta - eh * 3600.0) / 60.0);
                const int es = (int)((long long)eta % 60LL);
                std::cout << "\r[BSGS] Giant " << std::fixed << std::setprecision(1)
                          << prog << "% | " << speed << " Msteps/s | ETA "
                          << std::setfill('0') << std::setw(2) << eh << ":"
                          << std::setw(2) << em << ":" << std::setw(2) << es
                          << std::setfill(' ') << std::flush;
                t_progress_last = now;
                printed_progress = true;
            }
        }
    }
    const auto t_giant1 = Clock::now();
    if (printed_progress) std::cout << "\n";

    if (cerr != cudaSuccess) {
        std::cout << "[BSGS] Giant BloomBucketed kernel skipped/failed: "
                  << cudaGetErrorString(cerr) << "\n";
        bsgs_free_buffers(buffers);
        return 0;
    }

    if (g_sigint) {
        write_resume_snapshot(rs);
        print_resume_hint();
        bsgs_free_buffers(buffers);
        return 0;
    }

    const auto t_total1 = Clock::now();
    uint32_t bloom_hit_count = 0;
    cudaMemcpy(&bloom_hit_count, buffers.d_hit_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::cout << std::fixed << std::setprecision(3)
              << "[BSGS] Timings ms: baby=" << elapsed_ms(t_baby0, t_baby1)
              << " counts_d2h=" << elapsed_ms(t_copy0, t_copy1)
              << " prefix_cpu=" << elapsed_ms(t_prefix0, t_prefix1)
              << " offsets_h2d=" << elapsed_ms(t_upload0, t_upload1)
              << " scatter=" << elapsed_ms(t_scatter0, t_scatter1)
              << " giant=" << elapsed_ms(t_giant0, t_giant1)
              << " total=" << elapsed_ms(t_total0, t_total1)
              << " bloom_hits=" << bloom_hit_count
              << std::defaultfloat << "\n";

    BsgsCandidate candidate = {};
    cerr = cudaMemcpy(&candidate, buffers.d_candidate, sizeof(BsgsCandidate), cudaMemcpyDeviceToHost);
    if (cerr == cudaSuccess && candidate.found) {
        std::cout << "[BSGS] Giant BloomBucketed MATCH\n";
        std::cout << "  idx_a=" << candidate.idx_a
                  << " idx_b=" << candidate.idx_b
                  << " carry=" << (int)candidate.carry << "\n";

        uint8_t key_be[32] = {};
        bsgs_reconstruct_hex_key_cpu(plan, candidate.idx_a, candidate.idx_b, key_be);
        const bool verified = bsgs_verify_key_against_target_pubkey(key_be, target);

        if (verified) {
            clear_resume_snapshot();
            const std::string private_hex = hex_key_from_be32(key_be);
            std::cout << "\n======== BSGS VICTORY ===============================\n";
            std::cout << "Private key : " << private_hex << "\n";
            std::cout << "idx_a       : " << candidate.idx_a << "\n";
            std::cout << "idx_b       : " << candidate.idx_b << "\n";
            std::cout << "carry       : " << (int)candidate.carry << "\n";
            std::cout << "======================================================\n";
            notify_victory("BSGS HEX FOUND",
                "*Private key:*\n`" + private_hex + "`",
                pubkey_str.empty() ? "" : ("*Target:* `" + pubkey_str + "`"));
        } else {
            std::cout << "[BSGS] Candidate failed CPU pubkey verification\n";
            std::cout << "  private key candidate: " << hex_key_from_be32(key_be) << "\n";
        }
    } else if (cerr == cudaSuccess) {
        clear_resume_snapshot();
        std::cout << "[BSGS] Giant BloomBucketed: no match\n";
    } else {
        std::cout << "[BSGS] Giant BloomBucketed candidate copy failed: "
                  << cudaGetErrorString(cerr) << "\n";
    }

    bsgs_free_buffers(buffers);
    return 0;
}

static int run_hex_bsgs_mode(
    const std::string& mask_str,
    const std::string& pubkey_str,
    const ResumeState* resume = nullptr,
    const TargetData* resolved_target = nullptr,
    BsgsBabyOverride baby_override = BsgsBabyOverride::AUTO)
{
    return run_hex_bsgs_execute_mode(mask_str, pubkey_str, resume, resolved_target, baby_override);
}

static int run_hex_mode(const std::string &mask_str, const std::string &addr_str, const ResumeState* resume = nullptr) {

    // --- Parse mask ---
    MaskParseResult mask_r = parseMask(mask_str);
    if (!mask_r.valid) return 1;

    // --- Decode target address ---
    TargetData target = {};
    if (is_bloom_arg(addr_str)) {
        target.type = get_bloom_type(addr_str);
        if (!load_bloom_to_target(target)) return 1;
        std::string mode_name = (target.type==TargetType::BLOOM_BTC) ? "HEX + Bloom BTC"
                              : (target.type==TargetType::BLOOM_ETH) ? "HEX + Bloom ETH"
                              : "HEX + Bloom";
        std::cout << "Mode : " << mode_name << "\n";
    } else if (parse_pubkey_target_arg(addr_str, target)) {
        std::cout << "Mode : HEX + PubKey (hash bypassed)\n";
    } else if (addr_str.size() >= 2 && addr_str[0]=='0' && (addr_str[1]=='x'||addr_str[1]=='X')) {
        target.type = TargetType::ETH;
        if (!ethAddrToBytes(addr_str, target.hash20)) {
            std::cerr << "Error: invalid ETH address.\n"; return 1;
        }
        // Attempt to fetch pubkey via Blockscout + ecrecover.
        // If successful, target.type becomes ETH_PUBKEY and keccak256 is bypassed.
        try_fetch_pubkey_eth(addr_str, target);
        std::cout << "Mode : " << (target.type == TargetType::ETH_PUBKEY
                                   ? "ETH (pubkey known -- keccak256 bypassed)"
                                   : "ETH") << "\n";
    } else {
        target.type = TargetType::BTC;
        if (!addrToHash160Any(addr_str, target.hash20)) {
            std::cerr << "Error: invalid BTC address.\n"; return 1;
        }
        // Attempt to fetch the known public key from blockchain API.
        // If successful, target.type becomes BTC_PUBKEY and SHA256+RIPEMD160
        // are bypassed on GPU (~25% faster). Falls back to BTC_EXACT silently.
        try_fetch_pubkey_btc(addr_str, target);
        std::cout << "Mode : " << (target.type == TargetType::BTC_PUBKEY
                                   ? "BTC (pubkey known -- hash160 bypassed)"
                                   : "BTC") << "\n";
    }

    // --- CPU ECC precomputation ---
    HydraData h_fd;
    if (!precompute_ecc(mask_r, h_fd)) return 1;

    // --- GPU init ---
    ResumeState rs = make_resume_state("hex", mask_str, addr_str);
    rs.total = h_fd.high_candidates;

    int device = 0;
    cudaError_t cuda_err = cudaSetDevice(device);
    if (cuda_err != cudaSuccess) {
        std::cout << "CUDA unavailable for HEX run: "
                  << cudaGetErrorString(cuda_err) << "\n";
        return 0;
    }
    cudaDeviceProp prop;
    cuda_err = cudaGetDeviceProperties(&prop, device);
    if (cuda_err != cudaSuccess) {
        std::cout << "CUDA unavailable for HEX run: "
                  << cudaGetErrorString(cuda_err) << "\n";
        return 0;
    }

    // --- Allocate GPU buffers ---
    HydraData   *d_fd     = nullptr;
    TargetData  *d_target = nullptr;
    HydraResult *d_result = nullptr;

    cudaMalloc(&d_fd,     sizeof(HydraData));
    cudaMalloc(&d_target, sizeof(TargetData));
    cudaMalloc(&d_result, sizeof(HydraResult));

    cudaMemcpy(d_target, &target, sizeof(TargetData), cudaMemcpyHostToDevice);

    HydraResult h_result = {0, 0, 0};
    cudaMemcpy(d_result, &h_result, sizeof(HydraResult), cudaMemcpyHostToDevice);

    // In V5, each thread handles ONE P_base -> LOW_SIZE candidates
    // wave_size = number of P_base per kernel launch
    const int threads   = 256;
    const int blocks    = prop.multiProcessorCount * 128;
    const int wave_size = (int)std::min(
        (uint64_t)(blocks * threads),
        h_fd.high_candidates);

    std::cout << "======== HYDRA V5 (AFFINE DICT) =================\n";
    std::cout << "GPU         : " << prop.name << " (" << prop.multiProcessorCount << " SM)\n";
    std::cout << "Blocks      : " << blocks << " x " << threads << " threads\n";
    std::cout << "Dict size   : " << LOW_SIZE << " (2^" << LOW_BITS << " bits bas)\n";
    std::cout << "Candidates  : " << format_pow2_candidate_count((int)h_fd.num_var_bits)
              << " (2^" << (int)h_fd.num_var_bits << ")\n";
    std::cout << "P_base pool : " << h_fd.high_candidates << " (2^" << (int)h_fd.num_high_bits << " bits hauts)\n";
    hydra_platform::install_interrupt_handler(&g_sigint);

    auto t0     = std::chrono::high_resolution_clock::now();
    auto t_last = t0;
    auto t_resume_last = t0;
    uint64_t offset = 0;       // in units of P_base (index in high bits)
    int found = 0;
    double keys_since_last = 0;

    if (resume && resume->active) {
        offset = std::min(resume->offset, h_fd.high_candidates);
        rs.offset = offset;
        std::cout << "[Resume] HEX at high-offset " << offset << " / " << h_fd.high_candidates << "\n";
    }
    write_resume_snapshot(rs);

    while (!g_sigint && found == 0 && offset < h_fd.high_candidates) {
        uint64_t remaining = h_fd.high_candidates - offset;
        int cur_wave = (int)std::min((uint64_t)wave_size, remaining);

        h_fd.gray_offset_start = offset;
        cudaMemcpy(d_fd, &h_fd, sizeof(HydraData), cudaMemcpyHostToDevice);

        launch_hydra_mega_kernel(d_fd, d_target, d_result, cur_wave, blocks, threads, target.type);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n";
            break;
        }

        cudaMemcpy(&found, &d_result->found, sizeof(int), cudaMemcpyDeviceToHost);
        offset += cur_wave;
        rs.offset = offset;
        // Count actual candidates tested (P_base * LOW_SIZE)
        keys_since_last += (double)cur_wave * LOW_SIZE;

        auto now = std::chrono::high_resolution_clock::now();
        const bool force_resume = found || g_sigint || (offset >= h_fd.high_candidates);
        if (should_write_resume_snapshot(t_resume_last, now, force_resume)) {
            write_resume_snapshot(rs);
            t_resume_last = now;
        }
        double dt = std::chrono::duration<double>(now - t_last).count();
        if (dt >= 1.0) {
            double speed   = keys_since_last / dt / 1e6;
            double elapsed = std::chrono::duration<double>(now - t0).count();
            // progress in terms of actual candidates tested
            const double total_candidates_d = pow2_as_double((int)h_fd.num_var_bits);
            double prog    = 100.0 * ((double)offset * (double)LOW_SIZE) / total_candidates_d;
            double total_keys_done = (double)offset * LOW_SIZE;
            double eta = (elapsed > 0 && total_keys_done > 0)
                ? (total_candidates_d - total_keys_done) / (total_keys_done / elapsed) : 0;
            int eh=(int)(eta/3600), em=(int)((eta-eh*3600)/60), es=(int)((long long)eta%60);
            std::cout << "\r[" << std::fixed << std::setprecision(1) << prog << "%] "
                      << std::setprecision(2) << speed << " MK/s"
                      << " | ETA " << std::setfill('0')
                      << std::setw(2) << eh << ":" << std::setw(2) << em << ":" << std::setw(2) << es
                      << std::flush;
            t_last = now;
            keys_since_last = 0;
        }

        // Bloom hit detected inside loop -> handle here so we can continue
        if (found && (target.type == TargetType::BLOOM || target.type == TargetType::BLOOM_BTC || target.type == TargetType::BLOOM_ETH)) {
            cudaMemcpy(&h_result, d_result, sizeof(HydraResult), cudaMemcpyDeviceToHost);

            uint64_t high_idx = h_result.index / (uint64_t)LOW_SIZE;
            uint64_t low_k    = h_result.index % (uint64_t)LOW_SIZE;
            uint8_t key[32];
            memcpy(key, mask_r.k_fixed, 32);
            uint64_t gray_high = high_idx ^ (high_idx >> 1);
            int high_bits = (int)h_fd.num_high_bits;
            int low_bits  = (int)h_fd.num_var_bits - high_bits;
            for (int i = 0; i < high_bits; i++) {
                if ((gray_high >> i) & 1ULL) {
                    int bit_pos  = mask_r.var_bit_positions[low_bits + i];
                    int byte_idx = 31 - (bit_pos / 8);
                    int bit_off  = bit_pos % 8;
                    key[byte_idx] |= (uint8_t)(1 << bit_off);
                }
            }
            for (int j = 0; j < low_bits; j++) {
                if ((low_k >> j) & 1ULL) {
                    int bit_pos  = mask_r.var_bit_positions[j];
                    int byte_idx = 31 - (bit_pos / 8);
                    int bit_off  = bit_pos % 8;
                    key[byte_idx] |= (uint8_t)(1 << bit_off);
                }
            }
            std::string addr_legacy, addr_segwit, addr_eth;
            key_to_addresses(key, addr_legacy, addr_segwit, addr_eth);

            bool victory = check_balances_and_notify(key, addr_legacy, addr_segwit, addr_eth);
            if (victory) {
                // Non-zero balance -> keep found=1, exit loop -> VICTORY displayed below
            } else {
                // False positive -> reset and continue
                int zero = 0;
                cudaMemcpy(&d_result->found, &zero, sizeof(int), cudaMemcpyHostToDevice);
                found = 0;
            }
        }
    }
    std::cout << "\n";

    if (found) {
        clear_resume_snapshot();
        cudaMemcpy(&h_result, d_result, sizeof(HydraResult), cudaMemcpyDeviceToHost);

        // In V5, result->index = high_idx * LOW_SIZE + low_k
        // Reconstruct : high bits = high_idx (Gray), low bits = low_k (direct binary)
        uint64_t high_idx = h_result.index / (uint64_t)LOW_SIZE;
        uint64_t low_k    = h_result.index % (uint64_t)LOW_SIZE;

        uint8_t key[32];
        memcpy(key, mask_r.k_fixed, 32);

        // High bits : decode Gray -> set in key
        uint64_t gray_high = high_idx ^ (high_idx >> 1);
        int high_bits = (int)h_fd.num_high_bits;
        int low_bits  = (int)h_fd.num_var_bits - high_bits;
        for (int i = 0; i < high_bits; i++) {
            if ((gray_high >> i) & 1ULL) {
                int bit_pos  = mask_r.var_bit_positions[low_bits + i];
                int byte_idx = 31 - (bit_pos / 8);
                int bit_off  = bit_pos % 8;
                key[byte_idx] |= (uint8_t)(1 << bit_off);
            }
        }
        // Low bits : low_k is direct binary (not Gray)
        for (int j = 0; j < low_bits; j++) {
            if ((low_k >> j) & 1ULL) {
                int bit_pos  = mask_r.var_bit_positions[j];
                int byte_idx = 31 - (bit_pos / 8);
                int bit_off  = bit_pos % 8;
                key[byte_idx] |= (uint8_t)(1 << bit_off);
            }
        }

        // Compute and display addresses
        std::string addr_legacy, addr_segwit, addr_eth;
        key_to_addresses(key, addr_legacy, addr_segwit, addr_eth);

        // Bloom mode : already handled in loop (API check + reset on false positive)
        // We only reach here if victory=true (balance > 0) or direct address mode
        std::cout << "\n======== VICTORY ! KEY FOUND ==========================\n";
        std::cout << "Private key : "; print_key(key);
        std::cout << "  BTC legacy  : " << addr_legacy << "\n";
        std::cout << "  BTC segwit  : " << addr_segwit << "\n";
        std::cout << "  ETH         : " << addr_eth << "\n";
        std::cout << "=======================================================\n";
        {
            std::ostringstream pk_ss;
            pk_ss << std::hex << std::setfill('0');
            for (int i = 0; i < 32; ++i) pk_ss << std::setw(2) << (int)key[i];
            std::string key_info = "*Private Key:*\n`" + pk_ss.str() + "`";
            // Relevant address = the one we were searching for
            std::string addr_info;
            if (target.type == TargetType::BTC_PUBKEY || target.type == TargetType::ETH_PUBKEY)
                addr_info = "*PubKey target matched*";
            else if (target.type == TargetType::ETH)
                addr_info = "*ETH:* `" + addr_eth + "`";
            else if (!addr_str.empty() && addr_str.substr(0,3) == "bc1")
                addr_info = "*BTC segwit:* `" + addr_segwit + "`";
            else
                addr_info = "*BTC legacy:* `" + addr_legacy + "`";
            notify_victory("KEY FOUND", key_info, addr_info);
        }
    } else if (!g_sigint) {
        clear_resume_snapshot();
        std::cout << "Not found in " << format_pow2_candidate_count((int)h_fd.num_var_bits)
                  << " candidates.\n";
    } else {
        write_resume_snapshot(rs);
        print_resume_hint();
    }
    print_search_summary(found != 0);
    if(target.d_bloom_filter && (target.type == TargetType::BLOOM || target.type == TargetType::BLOOM_BTC || target.type == TargetType::BLOOM_ETH)) cudaFree((void*)target.d_bloom_filter);
    cudaFree(d_fd); cudaFree(d_target); cudaFree(d_result);
    return found ? 0 : 2;
}

// =================================================================================
// 6. MODE DETECTION AND MAIN
// =================================================================================

// Detect if the argument looks like a hex mask (64 hex+# chars)
static bool looks_like_hex_mask(const std::string &s) {
    std::string t = s;
    if (t.size() >= 2 && t[0]=='0' && (t[1]=='x'||t[1]=='X')) t=t.substr(2);
    if (t.size() != 64) return false;
    for (char c : t) {
        if (!isxdigit(c) && c!='#') return false;
    }
    return true;
}

// Detect if this is a BIP39 phrase (contains spaces)
static bool looks_like_seed(const std::string &s) {
    return s.find(' ') != std::string::npos;
}

// BIP39 phrase with no '#' -> all words known -> passphrase mode
static bool is_passphrase_mode(const std::string &s) {
    return looks_like_seed(s) && s.find('#') == std::string::npos;
}

static bool looks_like_wif_mask(const std::string &s) {
    // Compressed WIF: 52 chars (K/L...), uncompressed WIF: 51 chars (5...)
    if(s.size() != WIF_COMPRESSED_LEN && s.size() != WIF_UNCOMPRESSED_LEN) return false;
    static const char* valid = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz#";
    for(char c : s) if(!strchr(valid, c)) return false;
    if(s.size() == WIF_COMPRESSED_LEN)
        return (s[0] == 'K' || s[0] == 'L' || s[0] == '#');
    return (s[0] == '5' || s[0] == '#');
}

static bool cpu_derive_key(const std::string& phrase, const std::string& passphrase,
                            uint32_t coin_type, uint8_t privkey[32]);
static bool cpu_derive_key_electrum(const std::string& phrase, uint8_t privkey[32]);


// =================================================================================
// 7. BIP39 PHRASE PARSER WITH UNKNOWN POSITIONS
// =================================================================================

// Lookup table : word -> BIP39 index  (built at runtime from BIP39_Dict.h)
static std::unordered_map<std::string, uint16_t> build_word_map() {
    std::unordered_map<std::string, uint16_t> m;
    for(int i=0;i<2048;i++){
        uint16_t off = h_BIP39_OFFS[i];
        uint8_t  len = h_BIP39_LENS[i];
        m[std::string((char*)h_BIP39_BLOB+off, len)] = (uint16_t)i;
    }
    return m;
}

static bool parse_seed_mask(const std::string &phrase,
                            SeedMask &mask,
                            const std::unordered_map<std::string,uint16_t> &wmap)
{
    // Tokenize on whitespace
    std::vector<std::string> tokens;
    std::istringstream ss(phrase);
    std::string tok;
    while(ss >> tok) tokens.push_back(tok);

    int n = (int)tokens.size();
    if(n!=12 && n!=15 && n!=18 && n!=21 && n!=24){
        std::cerr << "Error: phrase must have 12/15/18/21/24 words (got " << n << ")\n";
        return false;
    }
    mask.num_words   = (uint8_t)n;
    mask.num_unknown = 0;
    // checksum_bits : ENT/32 = (n*11/33)*8/32 = n*11/132 bits -> 4 for 12 words, 8 for 24, etc.
    mask.checksum_bits = (uint8_t)((n * 11) / 33);  // 12->4, 15->5, 18->6, 21->7, 24->8

    for(int i=0;i<n;i++){
        if(tokens[i]=="#"){
            if(mask.num_unknown >= SEED_MAX_X){
                std::cerr << "Error: too many unknown positions (max " << SEED_MAX_X << ")\n";
                return false;
            }
            mask.unknown_pos[mask.num_unknown++] = (uint8_t)i;
            mask.word_indices[i] = 0xFFFF;
        } else {
            auto it = wmap.find(tokens[i]);
            if(it==wmap.end()){
                std::cerr << "Error: word not found in BIP39 wordlist: \"" << tokens[i] << "\"\n";
                return false;
            }
            mask.word_indices[i] = it->second;
        }
    }

    // Compute total_candidates and optimized checksum fields
    const uint8_t cs_bits = mask.checksum_bits;
    const uint8_t cs_mask_val = (uint8_t)((1u << cs_bits) - 1u);

    mask.last_word_unknown = (mask.num_unknown > 0 &&
        mask.unknown_pos[mask.num_unknown-1] == (uint8_t)(n-1));

    if(mask.num_unknown == 0){
        mask.total_candidates  = 1;
        mask.required_checksum = 0xFF;
    } else if(mask.last_word_unknown){
        // Last word unknown : iterate over 2^(11-cs_bits) entropy values,
        // checksum will be forced by K1 -> always valid
        mask.total_candidates = 1;
        for(int i=0; i<mask.num_unknown-1; i++) mask.total_candidates *= 2048ULL;
        mask.total_candidates *= (1ULL << (11 - cs_bits));
        mask.required_checksum = 0xFF;
    } else {
        // Last word known -> fixed checksum, K1 filters ~1/2^cs_bits
        mask.total_candidates = 1;
        for(int i=0; i<mask.num_unknown; i++) mask.total_candidates *= 2048ULL;
        mask.required_checksum = (uint8_t)(mask.word_indices[n-1] & cs_mask_val);
    }
    return true;
}

// =================================================================================
// 8. RUN SEED MODE
// =================================================================================
static int run_seed_mode(const std::string &phrase, const std::string &addr_str, const ResumeState* resume = nullptr) {

    auto wmap = build_word_map();

    // Parse target address
    TargetData target = {};
    if(is_bloom_arg(addr_str)){
        target.type = get_bloom_type(addr_str);
        if(!load_bloom_to_target(target)) return 1;
        std::string _bmode=(target.type==TargetType::BLOOM_BTC)?"SEED + Bloom BTC":(target.type==TargetType::BLOOM_ETH)?"SEED + Bloom ETH":"SEED + Bloom";
        std::cout << "Mode : " << _bmode << "\n";
        if (target.type == TargetType::BLOOM)
            std::cout << "Note : 'bloom' in seed mode scans only BTC path (m/44'/0'/0'/0/0). Use 'bloometh' for ETH wallets.\n";
    } else if (try_resolve_any_pubkey_target_for_scheduler(addr_str, target)) {
        if (target.type == TargetType::ETH_PUBKEY) {
            std::cout << "Mode : SEED / ETH PubKey m/44'/60'/0'/0/0\n";
        } else {
            std::cout << "Mode : SEED / BTC PubKey m/44'/0'/0'/0/0\n";
        }
    } else if(addr_str.size()>=2 && addr_str[0]=='0' && (addr_str[1]=='x'||addr_str[1]=='X')){
        target.type = TargetType::ETH;
        if(!ethAddrToBytes(addr_str, target.hash20)){
            std::cerr << "Error: invalid ETH address.\n"; return 1;
        }
        std::cout << "Mode : SEED / ETH m/44'/60'/0'/0/0\n";
    } else {
        target.type = TargetType::BTC;
        if(!addrToHash160Any(addr_str, target.hash20)){
            std::cerr << "Error: invalid BTC address.\n"; return 1;
        }
        std::cout << "Mode : SEED / BTC m/44'/0'/0'/0/0\n";
    }

    // Parse mask
    SeedMask mask = {};
    if(!parse_seed_mask(phrase, mask, wmap)) return 1;

    // Display
    {
        const uint32_t cs_div = (1u << mask.checksum_bits);
        uint64_t effective = mask.last_word_unknown
            ? mask.total_candidates
            : mask.total_candidates / cs_div;
        std::cout << "Words       : " << (int)mask.num_words << "\n";
        std::cout << "Unknown pos : " << (int)mask.num_unknown << "\n";
        std::cout << "Candidates  : " << mask.total_candidates
                  << " raw -> ~" << effective
                  << " valid (BIP39 checksum ÷" << cs_div << ")\n";
    }

    ResumeState rs = make_resume_state("seed", phrase, addr_str);
    rs.total = mask.total_candidates;

    // Upload BIP39 dictionary to constant memory
    cudaMemcpyToSymbol(d_BIP39_BLOB, h_BIP39_BLOB, sizeof(h_BIP39_BLOB));
    cudaMemcpyToSymbol(d_BIP39_OFFS, h_BIP39_OFFS, sizeof(h_BIP39_OFFS));
    cudaMemcpyToSymbol(d_BIP39_LENS, h_BIP39_LENS, sizeof(h_BIP39_LENS));

    int device=0; cudaSetDevice(device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);

    SeedMask    *d_mask   = nullptr;
    TargetData  *d_target = nullptr;
    HydraResult *d_result = nullptr;
    cudaMalloc(&d_mask,   sizeof(SeedMask));
    cudaMalloc(&d_target, sizeof(TargetData));
    cudaMalloc(&d_result, sizeof(HydraResult));
    cudaMemcpy(d_mask,   &mask,   sizeof(SeedMask),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, &target, sizeof(TargetData), cudaMemcpyHostToDevice);
    HydraResult h_res={0,0,0};
    cudaMemcpy(d_result, &h_res, sizeof(HydraResult), cudaMemcpyHostToDevice);

    const int THREADS   = 256;
    // Large wave size : K1 filters 15/16, we want ~250K valid for K2
    // 250K * 16 = 4M raw -> GPU stays busy between waves
    const int wave_k1   = 4 * 1024 * 1024;  // 4M raw candidates per wave
    const int blocks_k1 = (wave_k1 + THREADS - 1) / THREADS;
    const int blocks_k2 = prop.multiProcessorCount * 64;

    uint64_t *d_valid_indices = nullptr;
    int      *d_valid_count   = nullptr;
    uint8_t  *d_seeds     = nullptr;  // K2a -> K2b : seeds [wave_k2 × 64]
    uint8_t  *d_intermed  = nullptr;  // K2b -> K2c : priv||chain after m/44'/coin'/0' [×64]
    uint8_t  *d_nodes_tmp = nullptr;
    EccAffinePoint* d_affine = nullptr;
    const int wave_k2 = wave_k1;
    const int seed_wcomb_w = 14;
    uint64_t* d_seed_comb_GX = nullptr;
    uint64_t* d_seed_comb_GY = nullptr;
    int seed_comb_cols = 0, seed_comb_stride = 0;
    if (!brain_gen_table(seed_wcomb_w, &d_seed_comb_GX, &d_seed_comb_GY,
                         &seed_comb_cols, &seed_comb_stride)) return 1;
    cudaMalloc(&d_valid_indices, (size_t)wave_k1 * sizeof(uint64_t));
    cudaMalloc(&d_valid_count,   sizeof(int));
    cudaMalloc(&d_seeds,         (size_t)wave_k2 * 64);
    cudaMalloc(&d_intermed,      (size_t)wave_k2 * 64);
    cudaMalloc(&d_nodes_tmp,     (size_t)wave_k2 * 64);
    cudaMalloc(&d_affine,        (size_t)wave_k2 * sizeof(EccAffinePoint));
    
    BrainJacobian* d_jac = nullptr;
    bw_u256* d_local_except = nullptr;
    bw_u256* d_block_prods = nullptr;
    bw_u256* d_block_inv = nullptr;
    uint8_t* d_z_zero = nullptr;
    
    cudaMalloc(&d_jac, (size_t)wave_k2 * sizeof(BrainJacobian));
    cudaMalloc(&d_local_except, (size_t)wave_k2 * sizeof(bw_u256));
    const int max_blk_inv = (wave_k2 + BW_THREADS_INV - 1) / BW_THREADS_INV;
    cudaMalloc(&d_block_prods, (size_t)max_blk_inv * sizeof(bw_u256));
    cudaMalloc(&d_block_inv, (size_t)max_blk_inv * sizeof(bw_u256));
    cudaMalloc(&d_z_zero, (size_t)wave_k2 * sizeof(uint8_t));

    std::cout << "======== HYDRA V5 (SEED MODE) ====================\n";
    std::cout << "GPU         : " << prop.name << " (" << prop.multiProcessorCount << " SM)\n";
    std::cout << "K1 checksum : " << blocks_k1 << " blocks x " << THREADS << " (40 reg, wave=" << wave_k1 << ")\n";
    std::cout << "K2a PBKDF2  : " << blocks_k2 << " blocks x " << THREADS << " (~106 reg)\n";
    std::cout << "K2b hardened: " << blocks_k2 << " blocks x " << THREADS << " (0 ECC, 0 spill)\n";
    std::cout << "K2c ECC     : wCOMB w=" << seed_wcomb_w
              << " cols=" << seed_comb_cols << " stride=" << seed_comb_stride
              << " | multi-kernel batch inverse\n";
    hydra_platform::install_interrupt_handler(&g_sigint);

    auto t0 = std::chrono::high_resolution_clock::now(), t_last = t0;
    auto t_resume_last = t0;
    uint64_t offset        = 0;
    uint64_t pbkdf2_done   = 0;   // for final summary only
    uint64_t scan_since_last = 0; // raw candidates since last display
    int      found         = 0;
    uint64_t prof_raw = 0, prof_valid = 0, prof_batches = 0;
    double prof_k1_ms = 0.0, prof_k2a_ms = 0.0, prof_k2b_ms = 0.0, prof_k2c_ms = 0.0;
    cudaEvent_t ev_k10, ev_k11, ev_k2a0, ev_k2a1, ev_k2b0, ev_k2b1, ev_k2c0, ev_k2c1;
    cudaEventCreate(&ev_k10);  cudaEventCreate(&ev_k11);
    cudaEventCreate(&ev_k2a0); cudaEventCreate(&ev_k2a1);
    cudaEventCreate(&ev_k2b0); cudaEventCreate(&ev_k2b1);
    cudaEventCreate(&ev_k2c0); cudaEventCreate(&ev_k2c1);

    if (resume && resume->active) {
        offset = std::min(resume->offset, mask.total_candidates);
        rs.offset = offset;
        std::cout << "[Resume] SEED at offset " << offset << " / " << mask.total_candidates << "\n";
    }
    write_resume_snapshot(rs);

    while(!g_sigint && found==0 && offset < mask.total_candidates){
        uint64_t remaining = mask.total_candidates - offset;
        int cur_wave = (int)std::min((uint64_t)wave_k1, remaining);

        // K1 : checksum filter (~40 reg, eliminates 15/16)
        {
            int zero = 0;
            cudaMemcpy(d_valid_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
            int blk = (cur_wave + THREADS - 1) / THREADS;
            cudaEventRecord(ev_k10);
            hydra_checksum_kernel<<<blk, THREADS>>>(d_mask, offset, cur_wave,
                                                     d_valid_indices, d_valid_count, cur_wave);
            cudaEventRecord(ev_k11);
            if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                std::cerr << "CUDA Error (checksum): " << cudaGetErrorString(cudaGetLastError()) << "\n"; break;
            }
            float ms = 0.0f; cudaEventElapsedTime(&ms, ev_k10, ev_k11); prof_k1_ms += ms;
        }

        int h_valid = 0;
        cudaMemcpy(&h_valid, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_valid > cur_wave) {
            std::cerr << "Warning: seed valid index buffer saturated (" << h_valid
                      << " > " << cur_wave << "); clamping this wave.\n";
            h_valid = cur_wave;
        }

        // K2a/K2b/K2c : pipeline 3 kernels
        if(h_valid > 0 && !found){
            int blk = (h_valid + THREADS - 1) / THREADS;

            // K2a : word_indices -> seed[64]
            cudaEventRecord(ev_k2a0);
            hydra_k2a_pbkdf2<<<blk, THREADS>>>(
                d_mask, d_valid_indices, h_valid, d_seeds);
            cudaEventRecord(ev_k2a1);

            std::vector<uint32_t> coins_to_check;
            if (target.type == TargetType::BLOOM) coins_to_check = {0u, 60u};
            else coins_to_check = {(target.type == TargetType::ETH || target.type == TargetType::BLOOM_ETH || target.type == TargetType::ETH_PUBKEY) ? 60u : 0u};

            for (uint32_t coin_type : coins_to_check) {
                // K2b : seed[64] -> priv||chain after m/44'/coin'/0' (0 ECC, 0 spill)
                cudaEventRecord(ev_k2b0);
                hydra_k2b_hardened<<<blk, THREADS>>>(
                    coin_type, d_seeds, d_intermed, h_valid);
                cudaEventRecord(ev_k2b1);

                // K2c : 2 niveaux normaux (ECC×2) + ECC final + Hash + compare/Bloom
                cudaEventRecord(ev_k2c0);
                {
                    const int blk_ecc = (h_valid + BW_THREADS_ECC - 1) / BW_THREADS_ECC;

                    auto nodes_to_affine = [&](const uint8_t* nodes) {
                        kernel_ecc_nodes_to_jac_t<14><<<blk_ecc, BW_THREADS_ECC>>>(
                            nodes, h_valid, d_seed_comb_GX, d_seed_comb_GY, d_jac);
                            
                        const int blk_inv = (h_valid + BW_THREADS_INV - 1) / BW_THREADS_INV;
                        const int blk_inv2 = (blk_inv + 127) / 128;
                        const size_t smem_local = 2 * BW_THREADS_INV * sizeof(bw_u256);
                        const size_t smem_inv = (128 + 4) * sizeof(bw_u256);
                        
                        kernel_bw_local_prod<<<blk_inv, BW_THREADS_INV, smem_local>>>(
                            d_jac, h_valid, d_local_except, d_block_prods, d_z_zero);
                        kernel_bw_invert_blocks<<<blk_inv2, 128, smem_inv>>>(
                            d_block_prods, d_block_inv, blk_inv);
                        kernel_ecc_affine_from_jac<<<blk_inv, BW_THREADS_INV>>>(
                            d_jac, d_local_except, d_block_inv, d_z_zero, h_valid, d_affine);
                    };

                    // account pub -> external chain m/.../0
                    nodes_to_affine(d_intermed);
                    kernel_bip32_derive_normal_from_affine<<<blk, THREADS>>>(
                        d_intermed, d_affine, 0, h_valid, d_nodes_tmp);

                    // external chain pub -> address index 0
                    nodes_to_affine(d_nodes_tmp);
                    kernel_bip32_derive_normal_from_affine<<<blk, THREADS>>>(
                        d_nodes_tmp, d_affine, 0, h_valid, d_intermed);

                    // final leaf pub -> hash/compare
                    {
                        kernel_ecc_nodes_to_jac_t<14><<<blk_ecc, BW_THREADS_ECC>>>(
                            d_intermed, h_valid, d_seed_comb_GX, d_seed_comb_GY, d_jac);
                            
                        const int blk_inv = (h_valid + BW_THREADS_INV - 1) / BW_THREADS_INV;
                        const int blk_inv2 = (blk_inv + 127) / 128;
                        const size_t smem_local = 2 * BW_THREADS_INV * sizeof(bw_u256);
                        const size_t smem_inv = (128 + 4) * sizeof(bw_u256);
                        
                        kernel_bw_local_prod<<<blk_inv, BW_THREADS_INV, smem_local>>>(
                            d_jac, h_valid, d_local_except, d_block_prods, d_z_zero);
                        kernel_bw_invert_blocks<<<blk_inv2, 128, smem_inv>>>(
                            d_block_prods, d_block_inv, blk_inv);
                        kernel_ecc_finalize_from_jac_index<<<blk_inv, BW_THREADS_INV>>>(
                            d_jac, d_local_except, d_block_inv, d_z_zero, d_valid_indices, h_valid, d_target, d_result);
                    }
                }
                cudaEventRecord(ev_k2c1);

                cudaError_t err = cudaDeviceSynchronize();
                if(err != cudaSuccess){
                    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n"; break;
                }
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev_k2b0, ev_k2b1); prof_k2b_ms += ms;
                cudaEventElapsedTime(&ms, ev_k2c0, ev_k2c1); prof_k2c_ms += ms;
                cudaMemcpy(&found, &d_result->found, sizeof(int), cudaMemcpyDeviceToHost);
                if (found) break;
            }

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_k2a0, ev_k2a1); prof_k2a_ms += ms;

            // Bloom mode : bloom hit in loop -> verify balance
            if(found && (target.type==TargetType::BLOOM||target.type==TargetType::BLOOM_BTC||target.type==TargetType::BLOOM_ETH)) {
                cudaMemcpy(&h_res, d_result, sizeof(HydraResult), cudaMemcpyDeviceToHost);

                // Reconstruire la phrase du hit
                uint16_t hit_words[SEED_MAX_WORDS];
                for(int w=0;w<mask.num_words;w++) hit_words[w]=mask.word_indices[w];
                uint64_t hidx=h_res.index;
                for(int x=(int)mask.num_unknown-1;x>=0;x--){
                    hit_words[mask.unknown_pos[x]]=(uint16_t)(hidx%2048); hidx/=2048;
                }
                std::string hit_phrase;
                for(int w=0;w<mask.num_words;w++){
                    uint16_t off=h_BIP39_OFFS[hit_words[w]]; uint8_t len=h_BIP39_LENS[hit_words[w]];
                    if(w>0) hit_phrase+=" ";
                    hit_phrase+=std::string((char*)h_BIP39_BLOB+off,len);
                }

                std::cout << "\n!!! BLOOM HIT !!!\n  Phrase : " << hit_phrase << "\n";
                uint32_t coin = (target.type == TargetType::BLOOM_ETH) ? 60u : 0u;
                uint8_t privkey[32];
                cpu_derive_key(hit_phrase, "", coin, privkey);

                std::string addr_legacy, addr_segwit, addr_eth;
                key_to_addresses(privkey, addr_legacy, addr_segwit, addr_eth);
                bool victory = check_balances_and_notify(privkey, addr_legacy, addr_segwit, addr_eth);
                if (!victory) {
                    int zero = 0;
                    cudaMemcpy(&d_result->found, &zero, sizeof(int), cudaMemcpyHostToDevice);
                    found = 0;
                }
            }
        }

        offset           += cur_wave;
        rs.offset = offset;
        pbkdf2_done      += h_valid;
        prof_raw         += cur_wave;
        prof_valid       += (uint64_t)h_valid;
        prof_batches++;
        scan_since_last  += cur_wave;  // espace de recherche parcouru (brut)

        auto now = std::chrono::high_resolution_clock::now();
        const bool force_resume = found || g_sigint || (offset >= mask.total_candidates);
        if (should_write_resume_snapshot(t_resume_last, now, force_resume)) {
            write_resume_snapshot(rs);
            t_resume_last = now;
        }
        double dt = std::chrono::duration<double>(now - t_last).count();
        if(dt >= 1.0){
            // Speed = search space covered (K1 included in denominator)
            double speed   = scan_since_last / dt / 1e6;
            double elapsed = std::chrono::duration<double>(now - t0).count();
            double prog    = 100.0 * (double)offset / (double)mask.total_candidates;
            double eta     = (elapsed>0 && offset>0)
                ? (double)(mask.total_candidates - offset) / ((double)offset / elapsed) : 0;
            int eh=(int)(eta/3600), em=(int)((eta-eh*3600)/60), es=(int)((long long)eta%60);
            std::cout << "\r[" << std::fixed << std::setprecision(1) << prog << "%] "
                      << std::setprecision(2) << speed << " MKey/s"
                      << " | ETA " << std::setfill('0')
                      << std::setw(2)<<eh<<":"<<std::setw(2)<<em<<":"<<std::setw(2)<<es
                      << std::flush;
            t_last = now; scan_since_last = 0;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(t_end - t0).count();
    double avg_speed = (total_elapsed > 0) ? (double)offset / total_elapsed / 1e6 : 0;
    std::cout << "\nTime    : " << std::fixed << std::setprecision(2) << total_elapsed << " s"
              << " | " << std::setprecision(2) << avg_speed << " MKey/s avg"
              << " | PBKDF2 done : " << pbkdf2_done << "\n";
    if (prof_batches > 0) {
        auto raw_rate = [&](double ms) -> double {
            return (ms > 0.0) ? ((double)prof_raw / ms / 1000.0) : 0.0;
        };
        auto valid_rate = [&](double ms) -> double {
            return (ms > 0.0) ? ((double)prof_valid / ms / 1000.0) : 0.0;
        };
        const double k2_ms = prof_k2a_ms + prof_k2b_ms + prof_k2c_ms;
        std::cout << "[Profile] batches=" << prof_batches
                  << " raw=" << prof_raw
                  << " valid=" << prof_valid
                  << " k2=" << std::fixed << std::setprecision(2) << k2_ms << " ms\n";
        std::cout << "  K1 checksum : " << std::setw(9) << prof_k1_ms
                  << " ms | " << std::setw(7) << raw_rate(prof_k1_ms) << " Mraw/s\n";
        std::cout << "  K2a PBKDF2  : " << std::setw(9) << prof_k2a_ms
                  << " ms | " << std::setw(7) << valid_rate(prof_k2a_ms) << " Mcand/s\n";
        std::cout << "  K2b hardened: " << std::setw(9) << prof_k2b_ms
                  << " ms | " << std::setw(7) << valid_rate(prof_k2b_ms) << " Mcand/s\n";
        std::cout << "  K2c ECC     : " << std::setw(9) << prof_k2c_ms
                  << " ms | " << std::setw(7) << valid_rate(prof_k2c_ms) << " Mcand/s\n";
    }

    cudaEventDestroy(ev_k10);  cudaEventDestroy(ev_k11);
    cudaEventDestroy(ev_k2a0); cudaEventDestroy(ev_k2a1);
    cudaEventDestroy(ev_k2b0); cudaEventDestroy(ev_k2b1);
    cudaEventDestroy(ev_k2c0); cudaEventDestroy(ev_k2c1);
    cudaFree(d_valid_indices); cudaFree(d_valid_count);
    cudaFree(d_seeds); cudaFree(d_intermed); cudaFree(d_nodes_tmp); cudaFree(d_affine);
    cudaFree(d_jac); cudaFree(d_local_except); cudaFree(d_block_prods); cudaFree(d_block_inv); cudaFree(d_z_zero);
    cudaFree(d_seed_comb_GX); cudaFree(d_seed_comb_GY);

    if(found){
        clear_resume_snapshot();
        cudaMemcpy(&h_res,d_result,sizeof(HydraResult),cudaMemcpyDeviceToHost);
        // Reconstruit la combinaison gagnante
        uint16_t win_words[SEED_MAX_WORDS];
        for(int w=0;w<mask.num_words;w++) win_words[w]=mask.word_indices[w];
        uint64_t idx=h_res.index;
        for(int x=(int)mask.num_unknown-1;x>=0;x--){
            win_words[mask.unknown_pos[x]]=(uint16_t)(idx%2048);
            idx/=2048;
        }

        // Reconstruire la phrase
        std::string phrase;
        for(int w=0;w<mask.num_words;w++){
            uint16_t off=h_BIP39_OFFS[win_words[w]];
            uint8_t  len=h_BIP39_LENS[win_words[w]];
            if(w>0) phrase += " ";
            phrase += std::string((char*)h_BIP39_BLOB+off,len);
        }

        std::cout<<"\n======== VICTORY ! SEED FOUND =========================\n";
        std::cout<<"Phrase : " << phrase << "\n";
        std::cout<<"=======================================================\n";
        {
            std::string key_info = "*Phrase:*\n`" + phrase + "`";
            std::string addr_info = addr_str.empty() ? "" : ("*Adresse:* `" + addr_str + "`");
            notify_victory("SEED FOUND \xF0\x9F\x8C\xB1", key_info, addr_info);
        }
    } else if(!g_sigint){
        clear_resume_snapshot();
        std::cout<<"Not found in "<<mask.total_candidates<<" candidates.\n";
    } else {
        write_resume_snapshot(rs);
        print_resume_hint();
    }
    print_search_summary(found != 0);

    if(target.d_bloom_filter && (target.type == TargetType::BLOOM || target.type == TargetType::BLOOM_BTC || target.type == TargetType::BLOOM_ETH)) cudaFree((void*)target.d_bloom_filter);
    cudaFree(d_mask); cudaFree(d_target); cudaFree(d_result);
    return found ? 0 : 2;
}

// =================================================================================
// 8b. RUN ELECTRUM V2 SEED MODE
// Checksum : HMAC-SHA512("Seed version", mnemonic) → préfixe "01" (standard)
//            ou "100" (segwit, auto-détecté depuis bc1...)
// PBKDF2   : salt = "electrum"
// Dérivation: m/0/0 (normal, non hardened)
// =================================================================================
static int run_electrumv2_mode(const std::string &phrase, const std::string &addr_str, const ResumeState* resume = nullptr) {

    auto wmap = build_word_map();

    // Parse target address (même logique que run_seed_mode)
    TargetData target = {};
    uint8_t electrum_mode = 1;  // standard "01" par défaut

    if(is_bloom_arg(addr_str)){
        target.type = get_bloom_type(addr_str);
        if (target.type == TargetType::BLOOM_ETH) {
            std::cerr << "Error: Electrum V2 only supports BTC targets. Use 'bloom' or 'bloombtc'.\n";
            return 1;
        }
        if (target.type == TargetType::BLOOM) target.type = TargetType::BLOOM_BTC;
        if(!load_bloom_to_target(target)) return 1;
        std::string _bmode=(target.type==TargetType::BLOOM_BTC)?"ELECTRUM V2 + Bloom BTC":(target.type==TargetType::BLOOM_ETH)?"ELECTRUM V2 + Bloom ETH":"ELECTRUM V2 + Bloom";
        std::cout << "Mode : " << _bmode << " (standard, prefix '01')\n";
        electrum_mode = 1;
    } else if(addr_str.size()>=2 && addr_str[0]=='0' && (addr_str[1]=='x'||addr_str[1]=='X')){
        std::cerr << "Error: Electrum V2 only supports BTC targets.\n";
        return 1;
    } else if (try_resolve_btc_pubkey_target_for_scheduler(addr_str, target)) {
        if(addr_str.size() >= 3 && addr_str.substr(0,3) == "bc1"){
            electrum_mode = 2;
            std::cout << "Mode : ELECTRUM V2 / BTC PubKey segwit m/0/0 (prefix '100')\n";
        } else {
            electrum_mode = 1;
            std::cout << "Mode : ELECTRUM V2 / BTC PubKey standard m/0/0 (prefix '01')\n";
        }
    } else {
        target.type = TargetType::BTC;
        if(!addrToHash160Any(addr_str, target.hash20)){
            std::cerr << "Error: invalid BTC address.\n"; return 1;
        }
        // Segwit natif (bc1...) → préfixe "100", sinon standard "01"
        if(addr_str.size() >= 3 && addr_str.substr(0,3) == "bc1"){
            electrum_mode = 2;
            std::cout << "Mode : ELECTRUM V2 / BTC segwit m/0/0 (prefix '100')\n";
        } else {
            electrum_mode = 1;
            std::cout << "Mode : ELECTRUM V2 / BTC standard m/0/0 (prefix '01')\n";
        }
    }

    // Parse masque (même wordlist BIP39)
    SeedMask mask = {};
    if(!parse_seed_mask(phrase, mask, wmap)) return 1;

    // Override pour mode Electrum : pas d'optimisation checksum
    mask.electrum_mode    = electrum_mode;
    mask.last_word_unknown = false;
    mask.required_checksum = 0xFF;
    if(mask.num_unknown > 0){
        mask.total_candidates = 1;
        for(int i = 0; i < (int)mask.num_unknown; i++) mask.total_candidates *= 2048ULL;
    }

    // Affichage
    {
        const uint32_t cs_div = (electrum_mode == 2) ? 4096u : 256u;
        uint64_t effective = mask.total_candidates / cs_div;
        std::cout << "Words       : " << (int)mask.num_words << "\n";
        std::cout << "Unknown pos : " << (int)mask.num_unknown << "\n";
        std::cout << "Candidates  : " << mask.total_candidates
                  << " raw -> ~" << effective
                  << " valid (Electrum checksum ÷" << cs_div << ")\n";
    }

    ResumeState rs = make_resume_state("electrumv2", phrase, addr_str);
    rs.total = mask.total_candidates;

    // Upload dictionnaire BIP39
    cudaMemcpyToSymbol(d_BIP39_BLOB, h_BIP39_BLOB, sizeof(h_BIP39_BLOB));
    cudaMemcpyToSymbol(d_BIP39_OFFS, h_BIP39_OFFS, sizeof(h_BIP39_OFFS));
    cudaMemcpyToSymbol(d_BIP39_LENS, h_BIP39_LENS, sizeof(h_BIP39_LENS));

    int device=0; cudaSetDevice(device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);

    // Setup kernel : précalcul H_ipad/H_opad pour "Seed version"
    hydra_electrum_setup<<<1,1>>>();
    cudaDeviceSynchronize();

    SeedMask    *d_mask   = nullptr;
    TargetData  *d_target = nullptr;
    HydraResult *d_result = nullptr;
    cudaMalloc(&d_mask,   sizeof(SeedMask));
    cudaMalloc(&d_target, sizeof(TargetData));
    cudaMalloc(&d_result, sizeof(HydraResult));
    cudaMemcpy(d_mask,   &mask,   sizeof(SeedMask),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, &target, sizeof(TargetData), cudaMemcpyHostToDevice);
    HydraResult h_res={0,0,0};
    cudaMemcpy(d_result, &h_res, sizeof(HydraResult), cudaMemcpyHostToDevice);

    const int THREADS   = 256;
    const int wave_k1   = 4 * 1024 * 1024;
    const int blocks_k1 = (wave_k1 + THREADS - 1) / THREADS;
    const int blocks_k2 = prop.multiProcessorCount * 64;

    uint64_t *d_valid_indices = nullptr;
    int      *d_valid_count   = nullptr;
    uint8_t  *d_seeds     = nullptr;
    uint8_t  *d_intermed  = nullptr;
    uint8_t  *d_nodes_tmp = nullptr;
    EccAffinePoint* d_affine = nullptr;
    const int wave_k2 = wave_k1;
    const int ev2_wcomb_w = 14;
    uint64_t* d_comb_GX = nullptr;
    uint64_t* d_comb_GY = nullptr;
    int ev2_comb_cols = 0, ev2_comb_stride = 0;
    if (!brain_gen_table(ev2_wcomb_w, &d_comb_GX, &d_comb_GY, &ev2_comb_cols, &ev2_comb_stride)) return 1;
    cudaMalloc(&d_valid_indices, (size_t)wave_k1 * sizeof(uint64_t));
    cudaMalloc(&d_valid_count,   sizeof(int));
    cudaMalloc(&d_seeds,         (size_t)wave_k2 * 64);
    cudaMalloc(&d_intermed,      (size_t)wave_k2 * 64);
    cudaMalloc(&d_nodes_tmp,     (size_t)wave_k2 * 64);
    cudaMalloc(&d_affine,        (size_t)wave_k2 * sizeof(EccAffinePoint));
    
    BrainJacobian* d_jac = nullptr;
    bw_u256* d_local_except = nullptr;
    bw_u256* d_block_prods = nullptr;
    bw_u256* d_block_inv = nullptr;
    uint8_t* d_z_zero = nullptr;
    
    cudaMalloc(&d_jac, (size_t)wave_k2 * sizeof(BrainJacobian));
    cudaMalloc(&d_local_except, (size_t)wave_k2 * sizeof(bw_u256));
    const int max_blk_inv = (wave_k2 + BW_THREADS_INV - 1) / BW_THREADS_INV;
    cudaMalloc(&d_block_prods, (size_t)max_blk_inv * sizeof(bw_u256));
    cudaMalloc(&d_block_inv, (size_t)max_blk_inv * sizeof(bw_u256));
    cudaMalloc(&d_z_zero, (size_t)wave_k2 * sizeof(uint8_t));

    std::cout << "======== HYDRA (ELECTRUM V2 MODE) ==================\n";
    std::cout << "GPU         : " << prop.name << " (" << prop.multiProcessorCount << " SM)\n";
    std::cout << "K1 Elec     : " << blocks_k1 << " blocks x " << THREADS << " (HMAC-SHA512 filter)\n";
    std::cout << "K2a PBKDF2  : " << blocks_k2 << " blocks x " << THREADS << " (salt='electrum')\n";
    std::cout << "K2b master  : " << blocks_k2 << " blocks x " << THREADS << " (BIP32 master key)\n";
    std::cout << "K2c ECC m/0/0: wCOMB w=" << ev2_wcomb_w
              << " cols=" << ev2_comb_cols << " stride=" << ev2_comb_stride
              << " | multi-kernel batch inverse\n";
    hydra_platform::install_interrupt_handler(&g_sigint);

    auto t0 = std::chrono::high_resolution_clock::now(), t_last = t0;
    auto t_resume_last = t0;
    uint64_t offset        = 0;
    uint64_t pbkdf2_done   = 0;
    uint64_t scan_since_last = 0;
    int      found         = 0;
    uint64_t prof_raw = 0, prof_valid = 0, prof_batches = 0;
    double prof_k1_ms = 0.0, prof_k2a_ms = 0.0, prof_k2b_ms = 0.0, prof_k2c_ms = 0.0;
    cudaEvent_t ev_k10, ev_k11, ev_k2a0, ev_k2a1, ev_k2b0, ev_k2b1, ev_k2c0, ev_k2c1;
    cudaEventCreate(&ev_k10);  cudaEventCreate(&ev_k11);
    cudaEventCreate(&ev_k2a0); cudaEventCreate(&ev_k2a1);
    cudaEventCreate(&ev_k2b0); cudaEventCreate(&ev_k2b1);
    cudaEventCreate(&ev_k2c0); cudaEventCreate(&ev_k2c1);

    if(resume && resume->active){
        offset = std::min(resume->offset, mask.total_candidates);
        rs.offset = offset;
        std::cout << "[Resume] ELECTRUM V2 at offset " << offset << " / " << mask.total_candidates << "\n";
    }
    write_resume_snapshot(rs);

    while(!g_sigint && found==0 && offset < mask.total_candidates){
        uint64_t remaining = mask.total_candidates - offset;
        int cur_wave = (int)std::min((uint64_t)wave_k1, remaining);

        // K1 : filtre Electrum (HMAC-SHA512 + vérif préfixe)
        {
            int zero = 0;
            cudaMemcpy(d_valid_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
            int blk = (cur_wave + THREADS - 1) / THREADS;
            cudaEventRecord(ev_k10);
            hydra_electrum_filter<<<blk, THREADS>>>(d_mask, offset, cur_wave,
                                                     d_valid_indices, d_valid_count, cur_wave);
            cudaEventRecord(ev_k11);
            if(cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess){
                std::cerr << "CUDA Error (electrum filter): " << cudaGetErrorString(cudaGetLastError()) << "\n"; break;
            }
            float ms = 0.0f; cudaEventElapsedTime(&ms, ev_k10, ev_k11); prof_k1_ms += ms;
        }

        int h_valid = 0;
        cudaMemcpy(&h_valid, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_valid > cur_wave) {
            std::cerr << "Warning: Electrum V2 valid index buffer saturated (" << h_valid
                      << " > " << cur_wave << "); clamping this wave.\n";
            h_valid = cur_wave;
        }

        if(h_valid > 0 && !found){
            int blk = (h_valid + THREADS - 1) / THREADS;

            // K2a : PBKDF2 avec salt "electrum"
            cudaEventRecord(ev_k2a0);
            hydra_electrum_k2a_pbkdf2<<<blk, THREADS>>>(
                d_mask, d_valid_indices, h_valid, d_seeds);
            cudaEventRecord(ev_k2a1);

            // K2b : BIP32 master key seulement
            cudaEventRecord(ev_k2b0);
            hydra_electrum_k2b<<<blk, THREADS>>>(d_seeds, d_intermed, h_valid);
            cudaEventRecord(ev_k2b1);

            // K2c : m/0/0 via common wCOMB affine helpers.
            cudaEventRecord(ev_k2c0);
            {
                const int blk_ecc  = (h_valid + BW_THREADS_ECC - 1) / BW_THREADS_ECC;

                auto nodes_to_affine = [&](const uint8_t* nodes) {
                    kernel_ecc_nodes_to_jac_t<14><<<blk_ecc, BW_THREADS_ECC>>>(
                        nodes, h_valid, d_comb_GX, d_comb_GY, d_jac);
                        
                    const int blk_inv = (h_valid + BW_THREADS_INV - 1) / BW_THREADS_INV;
                    const int blk_inv2 = (blk_inv + 127) / 128;
                    const size_t smem_local = 2 * BW_THREADS_INV * sizeof(bw_u256);
                    const size_t smem_inv = (128 + 4) * sizeof(bw_u256);
                    
                    kernel_bw_local_prod<<<blk_inv, BW_THREADS_INV, smem_local>>>(
                        d_jac, h_valid, d_local_except, d_block_prods, d_z_zero);
                    kernel_bw_invert_blocks<<<blk_inv2, 128, smem_inv>>>(
                        d_block_prods, d_block_inv, blk_inv);
                    kernel_ecc_affine_from_jac<<<blk_inv, BW_THREADS_INV>>>(
                        d_jac, d_local_except, d_block_inv, d_z_zero, h_valid, d_affine);
                };

                // master pub -> child m/0
                nodes_to_affine(d_intermed);
                kernel_bip32_derive_normal_from_affine<<<blk, THREADS>>>(
                    d_intermed, d_affine, 0, h_valid, d_nodes_tmp);

                // m/0 pub -> child m/0/0
                nodes_to_affine(d_nodes_tmp);
                kernel_bip32_derive_normal_from_affine<<<blk, THREADS>>>(
                    d_nodes_tmp, d_affine, 0, h_valid, d_intermed);

                // final leaf pub -> hash/compare
                {
                    kernel_ecc_nodes_to_jac_t<14><<<blk_ecc, BW_THREADS_ECC>>>(
                        d_intermed, h_valid, d_comb_GX, d_comb_GY, d_jac);
                        
                    const int blk_inv = (h_valid + BW_THREADS_INV - 1) / BW_THREADS_INV;
                    const int blk_inv2 = (blk_inv + 127) / 128;
                    const size_t smem_local = 2 * BW_THREADS_INV * sizeof(bw_u256);
                    const size_t smem_inv = (128 + 4) * sizeof(bw_u256);
                    
                    kernel_bw_local_prod<<<blk_inv, BW_THREADS_INV, smem_local>>>(
                        d_jac, h_valid, d_local_except, d_block_prods, d_z_zero);
                    kernel_bw_invert_blocks<<<blk_inv2, 128, smem_inv>>>(
                        d_block_prods, d_block_inv, blk_inv);
                    kernel_ecc_finalize_from_jac_index<<<blk_inv, BW_THREADS_INV>>>(
                        d_jac, d_local_except, d_block_inv, d_z_zero, d_valid_indices, h_valid, d_target, d_result);
                }
            }
            cudaEventRecord(ev_k2c1);

            cudaError_t err = cudaDeviceSynchronize();
            if(err != cudaSuccess){
                std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n"; break;
            }
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_k2a0, ev_k2a1); prof_k2a_ms += ms;
            cudaEventElapsedTime(&ms, ev_k2b0, ev_k2b1); prof_k2b_ms += ms;
            cudaEventElapsedTime(&ms, ev_k2c0, ev_k2c1); prof_k2c_ms += ms;
            cudaMemcpy(&found, &d_result->found, sizeof(int), cudaMemcpyDeviceToHost);

            // Bloom hit → vérification balance
            if(found && (target.type==TargetType::BLOOM||target.type==TargetType::BLOOM_BTC||target.type==TargetType::BLOOM_ETH)){
                cudaMemcpy(&h_res, d_result, sizeof(HydraResult), cudaMemcpyDeviceToHost);

                uint16_t hit_words[SEED_MAX_WORDS];
                for(int w=0;w<mask.num_words;w++) hit_words[w]=mask.word_indices[w];
                uint64_t hidx=h_res.index;
                for(int x=(int)mask.num_unknown-1;x>=0;x--){
                    hit_words[mask.unknown_pos[x]]=(uint16_t)(hidx%2048); hidx/=2048;
                }
                std::string hit_phrase;
                for(int w=0;w<mask.num_words;w++){
                    uint16_t off=h_BIP39_OFFS[hit_words[w]]; uint8_t len=h_BIP39_LENS[hit_words[w]];
                    if(w>0) hit_phrase+=" ";
                    hit_phrase+=std::string((char*)h_BIP39_BLOB+off,len);
                }

                std::cout << "\n!!! BLOOM HIT !!!\n  Phrase : " << hit_phrase << "\n";
                uint8_t privkey[32];
                cpu_derive_key_electrum(hit_phrase, privkey);

                std::string addr_legacy, addr_segwit, addr_eth;
                key_to_addresses(privkey, addr_legacy, addr_segwit, addr_eth);
                bool victory = check_balances_and_notify(privkey, addr_legacy, addr_segwit, addr_eth);
                if(!victory){
                    int zero = 0;
                    cudaMemcpy(&d_result->found, &zero, sizeof(int), cudaMemcpyHostToDevice);
                    found = 0;
                }
            }
        }

        offset         += cur_wave;
        rs.offset = offset;
        pbkdf2_done    += h_valid;
        prof_raw       += cur_wave;
        prof_valid     += (uint64_t)h_valid;
        prof_batches++;
        scan_since_last += cur_wave;

        auto now = std::chrono::high_resolution_clock::now();
        const bool force_resume = found || g_sigint || (offset >= mask.total_candidates);
        if(should_write_resume_snapshot(t_resume_last, now, force_resume)){
            write_resume_snapshot(rs);
            t_resume_last = now;
        }
        double dt = std::chrono::duration<double>(now - t_last).count();
        if(dt >= 1.0){
            double speed   = scan_since_last / dt / 1e6;
            double elapsed = std::chrono::duration<double>(now - t0).count();
            double prog    = 100.0 * (double)offset / (double)mask.total_candidates;
            double eta     = (elapsed>0 && offset>0)
                ? (double)(mask.total_candidates - offset) / ((double)offset / elapsed) : 0;
            int eh=(int)(eta/3600), em=(int)((eta-eh*3600)/60), es=(int)((long long)eta%60);
            std::cout << "\r[" << std::fixed << std::setprecision(1) << prog << "%] "
                      << std::setprecision(2) << speed << " MKey/s"
                      << " | ETA " << std::setfill('0')
                      << std::setw(2)<<eh<<":"<<std::setw(2)<<em<<":"<<std::setw(2)<<es
                      << std::flush;
            t_last = now; scan_since_last = 0;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(t_end - t0).count();
    double avg_speed = (total_elapsed > 0) ? (double)offset / total_elapsed / 1e6 : 0;
    std::cout << "\nTime    : " << std::fixed << std::setprecision(2) << total_elapsed << " s"
              << " | " << std::setprecision(2) << avg_speed << " MKey/s avg"
              << " | PBKDF2 done : " << pbkdf2_done << "\n";
    if (prof_batches > 0) {
        auto raw_rate = [&](double ms) -> double {
            return (ms > 0.0) ? ((double)prof_raw / ms / 1000.0) : 0.0;
        };
        auto valid_rate = [&](double ms) -> double {
            return (ms > 0.0) ? ((double)prof_valid / ms / 1000.0) : 0.0;
        };
        const double k2_ms = prof_k2a_ms + prof_k2b_ms + prof_k2c_ms;
        std::cout << "[Profile] batches=" << prof_batches
                  << " raw=" << prof_raw
                  << " valid=" << prof_valid
                  << " k2=" << std::fixed << std::setprecision(2) << k2_ms << " ms\n";
        std::cout << "  K1 electrum: " << std::setw(9) << prof_k1_ms
                  << " ms | " << std::setw(7) << raw_rate(prof_k1_ms) << " Mraw/s\n";
        std::cout << "  K2a PBKDF2 : " << std::setw(9) << prof_k2a_ms
                  << " ms | " << std::setw(7) << valid_rate(prof_k2a_ms) << " Mcand/s\n";
        std::cout << "  K2b master : " << std::setw(9) << prof_k2b_ms
                  << " ms | " << std::setw(7) << valid_rate(prof_k2b_ms) << " Mcand/s\n";
        std::cout << "  K2c ECC    : " << std::setw(9) << prof_k2c_ms
                  << " ms | " << std::setw(7) << valid_rate(prof_k2c_ms) << " Mcand/s\n";
    }

    cudaEventDestroy(ev_k10);  cudaEventDestroy(ev_k11);
    cudaEventDestroy(ev_k2a0); cudaEventDestroy(ev_k2a1);
    cudaEventDestroy(ev_k2b0); cudaEventDestroy(ev_k2b1);
    cudaEventDestroy(ev_k2c0); cudaEventDestroy(ev_k2c1);
    cudaFree(d_valid_indices); cudaFree(d_valid_count);
    cudaFree(d_seeds); cudaFree(d_intermed);
    cudaFree(d_nodes_tmp);
    cudaFree(d_affine);
    cudaFree(d_comb_GX); cudaFree(d_comb_GY);

    if(found){
        clear_resume_snapshot();
        cudaMemcpy(&h_res, d_result, sizeof(HydraResult), cudaMemcpyDeviceToHost);
        uint16_t win_words[SEED_MAX_WORDS];
        for(int w=0;w<mask.num_words;w++) win_words[w]=mask.word_indices[w];
        uint64_t idx=h_res.index;
        for(int x=(int)mask.num_unknown-1;x>=0;x--){
            win_words[mask.unknown_pos[x]]=(uint16_t)(idx%2048);
            idx/=2048;
        }

        std::string found_phrase;
        for(int w=0;w<mask.num_words;w++){
            uint16_t off=h_BIP39_OFFS[win_words[w]];
            uint8_t  len=h_BIP39_LENS[win_words[w]];
            if(w>0) found_phrase += " ";
            found_phrase += std::string((char*)h_BIP39_BLOB+off,len);
        }

        std::cout<<"\n======== VICTORY ! ELECTRUM V2 SEED FOUND =============\n";
        std::cout<<"Phrase : " << found_phrase << "\n";
        std::cout<<"=======================================================\n";
        {
            std::string key_info = "*Electrum V2 Phrase:*\n`" + found_phrase + "`";
            std::string addr_info = addr_str.empty() ? "" : ("*Adresse:* `" + addr_str + "`");
            notify_victory("ELECTRUM V2 SEED FOUND \xF0\x9F\x8C\xB1", key_info, addr_info);
        }
    } else if(!g_sigint){
        clear_resume_snapshot();
        std::cout<<"Not found in "<<mask.total_candidates<<" candidates.\n";
    } else {
        write_resume_snapshot(rs);
        print_resume_hint();
    }
    print_search_summary(found != 0);

    if(target.d_bloom_filter && (target.type==TargetType::BLOOM||target.type==TargetType::BLOOM_BTC||target.type==TargetType::BLOOM_ETH)) cudaFree((void*)target.d_bloom_filter);
    cudaFree(d_mask); cudaFree(d_target); cudaFree(d_result);
    return found ? 0 : 2;
}

// =================================================================================
// 8c. RUN ELECTRUM V1 SEED MODE
// Legacy Electrum: 1626-word mnemonic, 100000x SHA256 stretch, old m/change/index.
// =================================================================================
static int run_electrumv1_mode(
    const std::string &phrase,
    const std::string &addr_str,
    const ResumeState* resume = nullptr,
    const std::vector<std::string>& cli_opts = {})
{
    TargetData target = {};
    if (is_bloom_arg(addr_str)) {
        target.type = get_bloom_type(addr_str);
        if (target.type == TargetType::BLOOM_ETH) {
            std::cerr << "Error: Electrum V1 only supports BTC legacy targets.\n";
            return 1;
        }
        if (!load_bloom_to_target(target)) return 1;
        if (target.type == TargetType::BLOOM) target.type = TargetType::BLOOM_BTC;
        std::cout << "Mode : ELECTRUM V1 + Bloom BTC\n";
    } else {
        target.type = TargetType::BTC;
        if (!addrToHash160(addr_str, target.hash20)) {
            std::cerr << "Error: Electrum V1 expects a BTC legacy P2PKH address.\n";
            return 1;
        }
        std::cout << "Mode : ELECTRUM V1 / BTC legacy\n";
    }

    ElectrumV1Mask mask = {};
    if (!parse_electrumv1_mask(phrase, mask)) return 1;
    if (resume && resume->active) {
        apply_electrumv1_resume_flags(mask, resume->bsgs_flags);
    } else if (!apply_electrumv1_cli_options(mask, cli_opts)) {
        return 1;
    }
    std::cout << "Words       : 12 old Electrum words\n";
    std::cout << "Unknown pos : " << (int)mask.num_unknown << "\n";
    std::cout << "Candidates  : " << mask.total_candidates << "\n";
    if (mask.single_path) {
        std::cout << "Path        : m/" << mask.path_change << "/" << mask.path_index << "\n";
    } else {
        std::cout << "Lookahead   : m/0/0.." << (mask.lookahead - 1)
                  << " + m/1/0.." << (mask.lookahead - 1) << "\n";
    }

    int device = 0;
    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    ElectrumV1Mask*   d_mask = nullptr;
    TargetData*       d_target = nullptr;
    ElectrumV1Result* d_result = nullptr;
    cudaMalloc(&d_mask, sizeof(ElectrumV1Mask));
    cudaMalloc(&d_target, sizeof(TargetData));
    cudaMalloc(&d_result, sizeof(ElectrumV1Result));
    cudaMemcpy(d_mask, &mask, sizeof(ElectrumV1Mask), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, &target, sizeof(TargetData), cudaMemcpyHostToDevice);
    ElectrumV1Result h_res = {};
    cudaMemcpy(d_result, &h_res, sizeof(ElectrumV1Result), cudaMemcpyHostToDevice);

    const int THREADS = ELECTRUM_V1_THREADS;
    const int wave = std::max(1024, prop.multiProcessorCount * 256);
    std::cout << "======== HYDRA (ELECTRUM V1 MODE) ==================\n";
    std::cout << "GPU       : " << prop.name << " (" << prop.multiProcessorCount << " SM)\n";
    std::cout << "Kernel    : " << THREADS << "t | wave=" << wave
              << " | stretch=100000 SHA256\n";

    ResumeState rs = make_resume_state("electrumv1", phrase, addr_str);
    rs.total = mask.total_candidates;
    rs.bsgs_flags = pack_electrumv1_resume_flags(mask);
    hydra_platform::install_interrupt_handler(&g_sigint);

    uint64_t offset = 0;
    int found = 0;
    if (resume && resume->active) {
        offset = std::min(resume->offset, mask.total_candidates);
        rs.offset = offset;
        std::cout << "[Resume] ELECTRUM V1 at offset " << offset
                  << " / " << mask.total_candidates << "\n";
    }
    write_resume_snapshot(rs);

    cudaStream_t s0, s1;
    cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);

    uint8_t *d_master_priv0 = nullptr, *d_master_priv1 = nullptr;
    cudaMalloc(&d_master_priv0, (size_t)wave * 32);
    cudaMalloc(&d_master_priv1, (size_t)wave * 32);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto t_last = t0;
    auto t_resume_last = t0;
    uint64_t since_last = 0;
    uint64_t prof_batches = 0;
    uint64_t prof_candidates = 0;
    double prof_kernel_ms = 0.0;
    cudaEvent_t ev_k0, ev_k1;
    cudaEventCreate(&ev_k0);
    cudaEventCreate(&ev_k1);

    uint64_t offset0 = offset;
    int wave0 = (int)std::min<uint64_t>((uint64_t)wave, mask.total_candidates - offset0);
    int blk0 = (wave0 + THREADS - 1) / THREADS;
    if (wave0 > 0) {
        cudaEventRecord(ev_k0, s0);
        hydra_electrumv1_stretch_kernel<<<blk0, THREADS, 0, s0>>>(d_mask, offset0, wave0, d_master_priv0);
    }

    while (!g_sigint && found == 0 && offset0 < mask.total_candidates) {
        uint64_t offset1 = offset0 + wave0;
        int wave1 = (int)std::min<uint64_t>((uint64_t)wave, mask.total_candidates - offset1);
        int blk1 = (wave1 + THREADS - 1) / THREADS;

        if (wave1 > 0) {
            hydra_electrumv1_stretch_kernel<<<blk1, THREADS, 0, s1>>>(d_mask, offset1, wave1, d_master_priv1);
        }

        hydra_electrumv1_scan_kernel<<<blk0, THREADS, 0, s0>>>(d_mask, d_target, d_master_priv0, offset0, wave0, d_result);
        cudaEventRecord(ev_k1, s0);
        
        cudaError_t err = cudaStreamSynchronize(s0);
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error (electrum v1): " << cudaGetErrorString(err) << "\n";
            break;
        }

        float stage_ms = 0.0f;
        cudaEventElapsedTime(&stage_ms, ev_k0, ev_k1);
        prof_kernel_ms += stage_ms;
        prof_batches++;
        prof_candidates += (uint64_t)wave0;
        
        cudaMemcpyAsync(&h_res, d_result, sizeof(ElectrumV1Result), cudaMemcpyDeviceToHost, s0);
        cudaStreamSynchronize(s0);
        found = h_res.found;

        if (found && (target.type == TargetType::BLOOM || target.type == TargetType::BLOOM_BTC)) {
            uint16_t hit_words[ELECTRUM_V1_WORDS];
            electrumv1_words_for_index(mask, h_res.index, hit_words);
            std::string hit_phrase = electrumv1_phrase_from_words(hit_words);
            uint8_t privkey[32];
            std::cout << "\n!!! BLOOM HIT !!!\n  Phrase : " << hit_phrase
                      << "\n  Path   : m/" << h_res.change << "/" << h_res.address_index << "\n";
            bool victory = false;
            if (cpu_derive_key_electrumv1_path(hit_phrase, h_res.change, h_res.address_index, privkey)) {
                std::string al, as_, ae;
                key_to_wif_addresses(privkey, false, al, as_, ae);
                victory = check_balances_and_notify(privkey, al, as_, ae);
            }
            if (!victory) {
                ElectrumV1Result zero = {};
                cudaMemcpyAsync(d_result, &zero, sizeof(ElectrumV1Result), cudaMemcpyHostToDevice, s0);
                cudaStreamSynchronize(s0);
                h_res = {};
                found = 0;
            }
        }

        offset += wave0;
        rs.offset = offset;
        since_last += wave0;
        auto now = std::chrono::high_resolution_clock::now();
        const bool force_resume = found || g_sigint || (offset >= mask.total_candidates);
        if (should_write_resume_snapshot(t_resume_last, now, force_resume)) {
            write_resume_snapshot(rs);
            t_resume_last = now;
        }
        double dt = std::chrono::duration<double>(now - t_last).count();
        if (dt >= 1.0) {
            double speed = since_last / dt;
            double elapsed = std::chrono::duration<double>(now - t0).count();
            double prog = 100.0 * (double)offset / (double)mask.total_candidates;
            double eta = (elapsed > 0 && offset > 0)
                ? (double)(mask.total_candidates - offset) / ((double)offset / elapsed) : 0;
            int eh=(int)(eta/3600), em=(int)((eta-eh*3600)/60), es=(int)((long long)eta%60);
            std::cout << "\r[" << std::fixed << std::setprecision(1) << prog << "%] "
                      << std::setprecision(2) << speed << " cand/s"
                      << " | ETA " << std::setfill('0')
                      << std::setw(2)<<eh<<":"<<std::setw(2)<<em<<":"<<std::setw(2)<<es
                      << std::flush;
            t_last = now;
            since_last = 0;
        }
        
        offset0 = offset1;
        wave0 = wave1;
        blk0 = blk1;
        std::swap(s0, s1);
        std::swap(d_master_priv0, d_master_priv1);
        cudaEventRecord(ev_k0, s0);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t0).count();
    double avg = elapsed > 0 ? (double)offset / elapsed : 0.0;
    std::cout << "\nTime    : " << std::fixed << std::setprecision(2) << elapsed << " s"
              << " | " << std::setprecision(2) << avg << " cand/s avg"
              << " | Tested  : " << offset << "\n";
    if (prof_candidates != 0) {
        double kernel_rate = prof_kernel_ms > 0.0 ? (double)prof_candidates / (prof_kernel_ms / 1000.0) : 0.0;
        std::cout << "[Profile] batches=" << prof_batches << " candidates=" << prof_candidates << "\n";
        std::cout << "  kernel     : " << std::setw(9) << std::setprecision(2) << prof_kernel_ms
                  << " ms | " << std::setw(10) << std::setprecision(2) << kernel_rate << " cand/s\n";
    }

    if (found) {
        clear_resume_snapshot();
        cudaMemcpy(&h_res, d_result, sizeof(ElectrumV1Result), cudaMemcpyDeviceToHost);
        uint16_t win_words[ELECTRUM_V1_WORDS];
        electrumv1_words_for_index(mask, h_res.index, win_words);
        std::string found_phrase = electrumv1_phrase_from_words(win_words);
        uint8_t privkey[32];
        std::cout << "\n======== VICTORY ! ELECTRUM V1 SEED FOUND ==========\n";
        std::cout << "Phrase : " << found_phrase << "\n";
        std::cout << "Path   : m/" << h_res.change << "/" << h_res.address_index << "\n";
        if (cpu_derive_key_electrumv1_path(found_phrase, h_res.change, h_res.address_index, privkey)) {
            std::cout << "PrivKey: ";
            print_key(privkey);
            std::string al, as_, ae;
            key_to_wif_addresses(privkey, false, al, as_, ae);
            std::cout << "BTC legacy : " << al << "\n";
        }
        std::cout << "====================================================\n";
        notify_victory("ELECTRUM V1 SEED FOUND",
                       "*Electrum V1 Phrase:*\n`" + found_phrase + "`",
                       "*Path:* `m/" + std::to_string(h_res.change) + "/" +
                           std::to_string(h_res.address_index) + "`");
    } else if (!g_sigint) {
        clear_resume_snapshot();
        std::cout << "Not found in " << mask.total_candidates << " candidates.\n";
    } else {
        write_resume_snapshot(rs);
        print_resume_hint();
    }

    print_search_summary(found != 0);
    if (target.d_bloom_filter && (target.type == TargetType::BLOOM || target.type == TargetType::BLOOM_BTC))
        cudaFree((void*)target.d_bloom_filter);
    cudaDeviceSynchronize();
    cudaStreamDestroy(s0);
    cudaStreamDestroy(s1);
    cudaFree(d_master_priv0);
    cudaFree(d_master_priv1);

    cudaFree(d_mask);
    cudaFree(d_target);
    cudaFree(d_result);
    cudaEventDestroy(ev_k0);
    cudaEventDestroy(ev_k1);
    return found ? 0 : 2;
}

// =================================================================================
// WIF MODE -- parse_wif_mask
// =================================================================================
static bool parse_wif_mask(const std::string &wif_str, WifMask &mask, bool verbose)
{
    if(wif_str.size() != WIF_COMPRESSED_LEN && wif_str.size() != WIF_UNCOMPRESSED_LEN){
        std::cerr << "Error: WIF mask must be " << WIF_COMPRESSED_LEN
                  << " chars compressed or " << WIF_UNCOMPRESSED_LEN
                  << " chars uncompressed (got " << wif_str.size() << ")\n";
        return false;
    }

    static const char* alpha = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    mask.num_chars   = (uint8_t)wif_str.size();
    mask.num_unknown = 0;
    mask.is_compressed = (wif_str.size() == WIF_COMPRESSED_LEN) ? 1 : 0;
    mask.decoded_bytes = mask.is_compressed ? WIF_COMPRESSED_BYTES : WIF_UNCOMPRESSED_BYTES;
    mask.payload_len = mask.is_compressed ? 34 : 33;
    mask.checksum_offset = mask.payload_len;
    mask.total_candidates = 1;

    for(int i = 0; i < mask.num_chars; i++){
        char c = wif_str[i];
        if(c == '#'){
            if(mask.num_unknown >= WIF_MAX_UNKN){
                std::cerr << "Error: too many unknown positions (max " << WIF_MAX_UNKN << ")\n";
                return false;
            }
            mask.unknown_pos[mask.num_unknown++] = (uint8_t)i;
            mask.known_b58[i] = 0xFF;
            if (mask.total_candidates <= UINT64_MAX / 58ULL)
                mask.total_candidates *= 58ULL;
            else
                mask.total_candidates = UINT64_MAX;
        } else {
            const char *p = strchr(alpha, c);
            if(!p){
                std::cerr << "Error: invalid character '" << c << "' (not Base58 or #)\n";
                return false;
            }
            mask.known_b58[i] = (uint8_t)(p - alpha);
        }
    }

    if(mask.num_unknown == 0){
        std::cerr << "Error: no unknown position (#) in WIF mask.\n";
        return false;
    }

    if (verbose) {
        std::cout << "Positions $ : " << (int)mask.num_unknown << "\n";
        std::cout << "Format      : " << (mask.is_compressed ? "compressed" : "uncompressed") << "\n";
        std::cout << "Candidates  : ";
        if (mask.total_candidates == UINT64_MAX)
            std::cout << ">=2^64";
        else
            std::cout << mask.total_candidates;
        std::cout << " (58^" << (int)mask.num_unknown << ")"
                  << " -> ~1 valid after SHA256x2 checksum\n";
    }
    return true;
}

// =================================================================================
// 8b. RUN WIF MODE
// =================================================================================
static int run_wif_mode(const std::string &wif_str, const std::string &addr_str, const ResumeState* resume = nullptr)
{
    // Parse adresse cible (WIF = BTC uniquement, pas d'ETH)
    TargetData target = {};
    if(is_bloom_arg(addr_str)){
        // WIF is a Bitcoin-only format -- bloom and bloombtc are equivalent here
        std::string l = addr_str;
        for (auto& ch : l) ch = tolower(ch);
        if (l == "bloometh") {
            std::cerr << "Error: WIF mode is Bitcoin-only. Use 'bloom' or 'bloombtc'.\n";
            return 1;
        }
        // bloom and bloombtc both map to BLOOM_BTC in WIF mode
        target.type = TargetType::BLOOM_BTC;
        if(!load_bloom_to_target(target)) return 1;
        std::cout << "Mode : WIF + Bloom BTC\n";
    } else if (parse_pubkey_target_arg(addr_str, target)) {
        std::cout << "Mode : WIF + PubKey (hash bypassed)\n";
    } else {
        target.type = TargetType::BTC;
        if(!addrToHash160Any(addr_str, target.hash20)){
            std::cerr << "Error: invalid BTC/SegWit address.\n"; return 1;
        }
        std::cout << "Mode : WIF / BTC\n";
    }

    WifMask mask = {};
    if(!parse_wif_mask(wif_str, mask)) return 1;

    ResumeState rs = make_resume_state("wif", wif_str, addr_str);
    rs.total = mask.total_candidates;

    // Précompute base + poids B58 -> constant memory GPU
    precompute_wif_b58(mask);

    // Alloue GPU
    int device = 0; cudaSetDevice(device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);

    WifMask    *d_mask   = nullptr;
    TargetData *d_target = nullptr;
    HydraResult*d_result = nullptr;
    cudaMalloc(&d_mask,   sizeof(WifMask));
    cudaMalloc(&d_target, sizeof(TargetData));
    cudaMalloc(&d_result, sizeof(HydraResult));
    cudaMemcpy(d_mask,   &mask,   sizeof(WifMask),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, &target, sizeof(TargetData), cudaMemcpyHostToDevice);
    HydraResult h_res = {0, 0, 0};
    cudaMemcpy(d_result, &h_res, sizeof(HydraResult), cudaMemcpyHostToDevice);

    const int THREADS   = 256;
    const int blocks_k1 = prop.multiProcessorCount * 64;
    const int wave_k1   = blocks_k1 * THREADS;
    const int blocks_k2 = prop.multiProcessorCount * 32;

    uint64_t *d_valid_indices = nullptr;
    int      *d_valid_count   = nullptr;
    cudaMalloc(&d_valid_indices, (size_t)wave_k1 * sizeof(uint64_t));
    cudaMalloc(&d_valid_count,   sizeof(int));

    std::cout << "======== HYDRA V5 (WIF MODE) =====================\n";
    std::cout << "GPU         : " << prop.name << " (" << prop.multiProcessorCount << " SM)\n";
    std::cout << "K1 checksum : " << blocks_k1 << " blocks × " << THREADS << " (~40 reg)\n";
    std::cout << "K2 ecc      : " << blocks_k2 << " blocks × " << THREADS << "\n";
    hydra_platform::install_interrupt_handler(&g_sigint);

    auto t0 = std::chrono::high_resolution_clock::now(), t_last = t0;
    auto t_resume_last = t0;
    uint64_t offset        = 0;
    uint64_t checked_total = 0;
    double   checked_since = 0;
    int      found         = 0;

    if (resume && resume->active) {
        offset = std::min(resume->offset, mask.total_candidates);
        checked_total = offset;
        rs.offset = offset;
        rs.tested = checked_total;
        std::cout << "[Resume] WIF at offset " << offset << " / " << mask.total_candidates << "\n";
    }
    write_resume_snapshot(rs);

    while(!g_sigint && !found && offset < mask.total_candidates){
        uint64_t remaining = mask.total_candidates - offset;
        int cur_wave = (int)std::min((uint64_t)wave_k1, remaining);

        // K1 : SHA256x2 checksum filter
        {
            int zero = 0;
            cudaMemcpy(d_valid_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
            int blk = (cur_wave + THREADS - 1) / THREADS;
            if(mask.is_compressed) {
                hydra_wif_checksum_kernel_t<true><<<blk, THREADS>>>(
                    d_mask, offset, cur_wave, d_valid_indices, d_valid_count, cur_wave);
            } else {
                hydra_wif_checksum_kernel_t<false><<<blk, THREADS>>>(
                    d_mask, offset, cur_wave, d_valid_indices, d_valid_count, cur_wave);
            }
            if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                std::cerr << "CUDA Error (wif checksum): " << cudaGetErrorString(cudaGetLastError()) << "\n"; break;
            }
        }

        int h_valid = 0;
        cudaMemcpy(&h_valid, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_valid > cur_wave) {
            std::cerr << "Warning: WIF valid index buffer saturated (" << h_valid
                      << " > " << cur_wave << "); clamping this wave.\n";
            h_valid = cur_wave;
        }

        // K2 : ECC + compare (rarely reached thanks to ÷2^32 filter)
        if(h_valid > 0 && !found){
            int blk = (h_valid + THREADS - 1) / THREADS;
            hydra_wif_ecc_kernel<<<blk, THREADS>>>(
                d_mask, d_target, d_result, d_valid_indices, h_valid);
            cudaError_t err = cudaDeviceSynchronize();
            if(err != cudaSuccess){
                std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n"; break;
            }
            cudaMemcpy(&found, &d_result->found, sizeof(int), cudaMemcpyDeviceToHost);

            if(found && (target.type == TargetType::BLOOM || target.type == TargetType::BLOOM_BTC || target.type == TargetType::BLOOM_ETH)) {
                cudaMemcpy(&h_res, d_result, sizeof(HydraResult), cudaMemcpyDeviceToHost);

                uint8_t b58vals[WIF_MAX_LEN];
                for(int i = 0; i < mask.num_chars; i++) b58vals[i] = mask.known_b58[i];
                uint64_t idx = h_res.index;
                for(int x = (int)mask.num_unknown - 1; x >= 0; x--){
                    b58vals[mask.unknown_pos[x]] = (uint8_t)(idx % 58);
                    idx /= 58;
                }
                static const char* alpha = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
                std::string wif_hit(mask.num_chars, ' ');
                for(int i = 0; i < mask.num_chars; i++) wif_hit[i] = alpha[b58vals[i]];

                static const char* WIF_ALPHA = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
                BIGNUM* bn = BN_new(); BN_zero(bn);
                BN_CTX* bctx = BN_CTX_new();
                for(char c : wif_hit){
                    const char* p = strchr(WIF_ALPHA, c);
                    int val = p ? (int)(p - WIF_ALPHA) : 0;
                    BN_mul_word(bn, 58);
                    BN_add_word(bn, (unsigned long)val);
                }
                uint8_t raw[WIF_MAX_BYTES]={};
                BN_bn2binpad(bn, raw, mask.decoded_bytes);
                BN_free(bn); BN_CTX_free(bctx);
                uint8_t privkey_wif[32];
                memcpy(privkey_wif, raw+1, 32);

                std::string addr_legacy, addr_segwit, addr_eth;
                key_to_wif_addresses(privkey_wif, mask.is_compressed != 0, addr_legacy, addr_segwit, addr_eth);
                bool victory = check_balances_and_notify(privkey_wif, addr_legacy, addr_segwit, addr_eth);
                if (!victory) {
                    int zero = 0;
                    cudaMemcpy(&d_result->found, &zero, sizeof(int), cudaMemcpyHostToDevice);
                    found = 0;
                }
            }
        }

        offset          += cur_wave;
        checked_total   += cur_wave;
        rs.offset = offset;
        rs.tested = checked_total;
        checked_since   += cur_wave;

        auto now = std::chrono::high_resolution_clock::now();
        const bool force_resume = found || g_sigint || (offset >= mask.total_candidates);
        if (should_write_resume_snapshot(t_resume_last, now, force_resume)) {
            write_resume_snapshot(rs);
            t_resume_last = now;
        }
        double dt = std::chrono::duration<double>(now - t_last).count();
        if(dt >= 1.0){
            double speed   = checked_since / dt / 1e6;  // M/s (checksum SHA256 est rapide)
            double elapsed = std::chrono::duration<double>(now - t0).count();
            double prog    = 100.0 * (double)offset / (double)mask.total_candidates;
            double eta     = (elapsed > 0 && offset > 0)
                ? (double)(mask.total_candidates - offset) / ((double)offset / elapsed) : 0;
            int eh = (int)(eta/3600), em = (int)((eta - eh*3600)/60), es = (int)((long long)eta%60);
            std::cout << "\r[" << std::fixed << std::setprecision(1) << prog << "%] "
                      << std::setprecision(1) << speed << " MKey/s"
                      << " | ETA " << std::setfill('0')
                      << std::setw(2) << eh << ":" << std::setw(2) << em << ":" << std::setw(2) << es
                      << std::flush;
            t_last = now; checked_since = 0;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(t_end - t0).count();
    double avg_speed = (total_elapsed > 0) ? checked_total / total_elapsed / 1e6 : 0;
    std::cout << "\nTime  : " << std::fixed << std::setprecision(2) << total_elapsed << " s"
              << " | Avg speed : " << std::setprecision(1) << avg_speed << " M/s"
              << " | Tested : " << checked_total << "\n";

    if(found){
        clear_resume_snapshot();
        cudaMemcpy(&h_res, d_result, sizeof(HydraResult), cudaMemcpyDeviceToHost);

        // Reconstruct the found WIF
        uint8_t b58vals[WIF_MAX_LEN];
        for(int i = 0; i < mask.num_chars; i++) b58vals[i] = mask.known_b58[i];
        uint64_t idx = h_res.index;
        for(int x = (int)mask.num_unknown - 1; x >= 0; x--){
            b58vals[mask.unknown_pos[x]] = (uint8_t)(idx % 58);
            idx /= 58;
        }
        static const char* alpha = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
        std::string wif_found(mask.num_chars, ' ');
        for(int i = 0; i < mask.num_chars; i++) wif_found[i] = alpha[b58vals[i]];

        // Decoder WIF -> cle privee -> adresses
        std::string addr_legacy, addr_segwit, addr_eth;
        {
            static const char* WIF_ALPHA = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
            BIGNUM* bn = BN_new(); BN_zero(bn);
            BN_CTX* bctx = BN_CTX_new();
            for(char c : wif_found){
                const char* p = strchr(WIF_ALPHA, c);
                int val = p ? (int)(p - WIF_ALPHA) : 0;
                BN_mul_word(bn, 58);
                BN_add_word(bn, (unsigned long)val);
            }
            uint8_t raw[WIF_MAX_BYTES]={};
            BN_bn2binpad(bn, raw, mask.decoded_bytes);
            BN_free(bn); BN_CTX_free(bctx);
            uint8_t privkey[32];
            memcpy(privkey, raw+1, 32);
            key_to_wif_addresses(privkey, mask.is_compressed != 0, addr_legacy, addr_segwit, addr_eth);
        }
        std::cout << "\n======== VICTORY ! WIF FOUND ==========================\n";
        std::cout << "WIF         : " << wif_found << "\n";
        std::cout << "BTC legacy  : " << addr_legacy << "\n";
        std::cout << "BTC segwit  : " << addr_segwit << "\n";
        std::cout << "ETH         : " << addr_eth << "\n";
        std::cout << "=======================================================\n";
        {
            std::string key_info = "*WIF:*\n`" + wif_found + "`";
            std::string addr_info;
            if (target.type == TargetType::BTC_PUBKEY)
                addr_info = "*PubKey target matched*";
            else if (!addr_str.empty() && addr_str.substr(0,3) == "bc1")
                addr_info = "*BTC segwit:* `" + addr_segwit + "`";
            else
                addr_info = "*BTC legacy:* `" + addr_legacy + "`";
            notify_victory("WIF FOUND", key_info, addr_info);
        }

    } else if(!g_sigint){
        clear_resume_snapshot();
        std::cout << "Not found in " << checked_total << " candidates.\n";
    } else {
        write_resume_snapshot(rs);
        print_resume_hint();
    }
    print_search_summary(found != 0);

    if(target.d_bloom_filter && (target.type == TargetType::BLOOM || target.type == TargetType::BLOOM_BTC || target.type == TargetType::BLOOM_ETH)) cudaFree((void*)target.d_bloom_filter);
    cudaFree(d_mask); cudaFree(d_target); cudaFree(d_result);
    cudaFree(d_valid_indices); cudaFree(d_valid_count);
    return found ? 0 : 2;
}

// =================================================================================
// 9. RUN PASSPHRASE MODE
// All words are known, brute-force the BIP39 passphrase from dictionary.txt
// =================================================================================
#define DICT_FILE      "resources/dictionary.txt"
#define PASS_BATCH     491520    // 480K passphrases par batch (MAX_PASS_LEN=96 -> ~45 MB)

struct TextBatchReader {
    FILE* f = nullptr;
    std::vector<uint8_t> buf;
    size_t buf_pos = 0;
    size_t buf_len = 0;
    uint64_t logical_offset = 0;
    bool eof = false;

    TextBatchReader(FILE* file, size_t buf_size = 16 * 1024 * 1024)
        : f(file), buf(buf_size) {
        int64_t p = hydra_platform::file_tell(f);
        logical_offset = (p >= 0) ? (uint64_t)p : 0;
    }

    bool seek(uint64_t offset) {
        if (!fseek(f, offset, SEEK_SET)) return false;
        buf_pos = 0;
        buf_len = 0;
        eof = false;
        logical_offset = offset;
        return true;
    }

    int fill_batch(uint8_t* out_chunk, uint32_t* out_offsets, int max_count, size_t max_chunk_size,
                   uint64_t& batch_start, uint64_t& batch_end, bool& hit_eof, size_t& out_chunk_size) {
        batch_start = logical_offset;
        hit_eof = false;
        
        int count = 0;
        uint32_t out_pos = 0; // Position in the binary formatted out_chunk

        // Shift remaining bytes to start and fill buffer if we are low on data or if buffer was completely full
        size_t remain = buf_len - buf_pos;
        if (remain < 1024 * 1024 || buf_len == buf.size()) {
            if (remain > 0 && buf_pos > 0) {
                memmove(buf.data(), buf.data() + buf_pos, remain);
            }
            buf_pos = 0;
            size_t read_bytes = fread(buf.data() + remain, 1, buf.size() - remain, f);
            buf_len = remain + read_bytes;
            if (read_bytes == 0 && buf_len == 0) {
                hit_eof = true; eof = true;
                out_chunk_size = 0;
                return 0;
            }
        }

        uint32_t scan_pos = buf_pos;
        const uint8_t* data = buf.data();

        while (count < max_count && scan_pos < buf_len) {
            // Find next newline
            const uint8_t* nl = (const uint8_t*)memchr(data + scan_pos, '\n', buf_len - scan_pos);
            uint32_t line_end = nl ? (nl - data) : buf_len;

            // If we didn't find a newline and the buffer is full, we must break to refill
            if (!nl && buf_len == buf.size()) {
                if (scan_pos == 0) {
                    // Word is larger than 16MB! Force truncate it to prevent infinite loop.
                    line_end = buf_len;
                } else {
                    break; 
                }
            }

            // Extract line
            uint32_t word_len = line_end - scan_pos;
            
            // Trim carriage return if present
            if (word_len > 0 && data[scan_pos + word_len - 1] == '\r') {
                word_len--;
            }

            // Truncate to 255 if too long
            if (word_len > 255) word_len = 255;

            // Check if it fits in out_chunk
            if (out_pos + 1 + word_len > max_chunk_size) {
                break;
            }

            // Write to out_chunk in binary format: [len][chars]
            out_offsets[count++] = out_pos;
            out_chunk[out_pos++] = (uint8_t)word_len;
            if (word_len > 0) {
                memcpy(out_chunk + out_pos, data + scan_pos, word_len);
                out_pos += word_len;
            }

            scan_pos = line_end;
            if (nl) scan_pos++; // Skip the newline character
            else break; // End of buffer without newline means EOF if buf_len < buf.size()
        }

        // Update logical offset with the number of raw bytes consumed
        size_t consumed_raw_bytes = scan_pos - buf_pos;
        logical_offset += consumed_raw_bytes;
        buf_pos = scan_pos;
        
        batch_end = logical_offset;
        out_chunk_size = out_pos;
        
        if (scan_pos == buf_len && buf_len < buf.size()) {
            hit_eof = true; eof = true;
        }

        return count;
    }
};

// =================================================================================
// CPU BIP32/44 DERIVATION
// seed + passphrase -> privkey (m/44'/coin'/0'/0/0)
// Used to verify bloom hits in passphrase mode (GPU does not output privkey)
// =================================================================================
// HMAC-SHA512 wrapper (OpenSSL)
static void cpu_hmac_sha512(const uint8_t* key, size_t key_len,
                             const uint8_t* msg, size_t msg_len,
                             uint8_t out[64])
{
    unsigned int out_len = 64;
    HMAC(EVP_sha512(), key, (int)key_len, msg, (int)msg_len, out, &out_len);
}

// (tweak + parent) mod secp256k1_n, big-endian byte[32]
static void cpu_add_mod_n(const uint8_t tweak[32], const uint8_t parent[32], uint8_t out[32])
{
    static const uint8_t N[32] = {
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
    };
    uint32_t carry = 0;
    uint8_t tmp[32];
    for (int i = 31; i >= 0; i--) {
        uint32_t s = (uint32_t)tweak[i] + (uint32_t)parent[i] + carry;
        tmp[i] = (uint8_t)(s & 0xFF);
        carry  = s >> 8;
    }
    // if tmp >= N, subtract N
    bool ge = (carry > 0);
    if (!ge) for (int i = 0; i < 32; i++) {
        if (tmp[i] > N[i]) { ge = true; break; }
        if (tmp[i] < N[i]) break;
    }
    if (ge) {
        uint32_t borrow = 0;
        for (int i = 31; i >= 0; i--) {
            int32_t s = (int32_t)tmp[i] - (int32_t)N[i] - (int32_t)borrow;
            out[i] = (uint8_t)((s + 256) & 0xFF);
            borrow = (s < 0) ? 1 : 0;
        }
    } else {
        memcpy(out, tmp, 32);
    }
}

// BIP32 hardened child derivation (CPU)
static void cpu_bip32_hardened(const uint8_t par_priv[32], const uint8_t par_chain[32],
                                uint32_t index, uint8_t ch_priv[32], uint8_t ch_chain[32])
{
    uint8_t data[37];
    data[0] = 0x00;
    memcpy(data + 1, par_priv, 32);
    data[33] = (uint8_t)(index >> 24); data[34] = (uint8_t)(index >> 16);
    data[35] = (uint8_t)(index >>  8); data[36] = (uint8_t)(index);
    uint8_t out[64];
    cpu_hmac_sha512(par_chain, 32, data, 37, out);
    cpu_add_mod_n(out, par_priv, ch_priv);
    memcpy(ch_chain, out + 32, 32);
}

// BIP32 normal child derivation -- needs compressed pubkey (CPU via OpenSSL 3.0 API)
static void cpu_bip32_normal(const uint8_t par_priv[32], const uint8_t par_chain[32],
                              uint32_t index, uint8_t ch_priv[32], uint8_t ch_chain[32])
{
    // Use EC_GROUP/EC_POINT directly (EC_KEY API deprecated since OpenSSL 3.0)
    EC_GROUP* grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BIGNUM*   bn  = BN_bin2bn(par_priv, 32, nullptr);
    EC_POINT* pub = EC_POINT_new(grp);
    EC_POINT_mul(grp, pub, bn, nullptr, nullptr, nullptr);

    uint8_t pubkey[33];
    EC_POINT_point2oct(grp, pub, POINT_CONVERSION_COMPRESSED, pubkey, 33, nullptr);
    EC_POINT_free(pub); BN_free(bn); EC_GROUP_free(grp);

    uint8_t data[37];
    memcpy(data, pubkey, 33);
    data[33] = (uint8_t)(index >> 24); data[34] = (uint8_t)(index >> 16);
    data[35] = (uint8_t)(index >>  8); data[36] = (uint8_t)(index);
    uint8_t out[64];
    cpu_hmac_sha512(par_chain, 32, data, 37, out);
    cpu_add_mod_n(out, par_priv, ch_priv);
    memcpy(ch_chain, out + 32, 32);
}

// Full pipeline: mnemonic phrase + passphrase -> privkey via BIP44 m/44'/coin'/0'/0/0
// coin_type: 0=BTC, 60=ETH
// Returns true on success
static bool cpu_derive_key(const std::string& phrase, const std::string& passphrase,
                            uint32_t coin_type, uint8_t privkey[32])
{
    // 1. PBKDF2-HMAC-SHA512 : mnemonic -> seed[64]
    const std::string salt = "mnemonic" + passphrase;
    uint8_t seed[64];
    PKCS5_PBKDF2_HMAC(phrase.c_str(), (int)phrase.size(),
                      (const uint8_t*)salt.c_str(), (int)salt.size(),
                      2048, EVP_sha512(), 64, seed);

    // 2. BIP32 master key : HMAC-SHA512("Bitcoin seed", seed)
    static const uint8_t BITCOIN_SEED[] = "Bitcoin seed";
    uint8_t master[64];
    cpu_hmac_sha512(BITCOIN_SEED, 12, seed, 64, master);
    uint8_t mpriv[32], mchain[32];
    memcpy(mpriv,  master,      32);
    memcpy(mchain, master + 32, 32);

    // 3. BIP44 : m/44'/coin'/0'/0/0
    uint8_t k0[32],c0[32], k1[32],c1[32], k2[32],c2[32], k3[32],c3[32];
    cpu_bip32_hardened(mpriv,  mchain,  0x8000002C,           k0, c0);  // 44'
    cpu_bip32_hardened(k0, c0, 0x80000000 | coin_type,        k1, c1);  // coin'
    cpu_bip32_hardened(k1, c1, 0x80000000,                    k2, c2);  // 0'
    cpu_bip32_normal  (k2, c2, 0,                             k3, c3);  // 0
    cpu_bip32_normal  (k3, c3, 0,                        privkey, c0);  // 0

    return true;
}

static bool cpu_derive_key_electrum(const std::string& phrase, uint8_t privkey[32])
{
    // PBKDF2-HMAC-SHA512 avec salt "electrum" (pas "mnemonic")
    static const uint8_t ELECTRUM_SALT[] = "electrum";
    uint8_t seed[64];
    PKCS5_PBKDF2_HMAC(phrase.c_str(), (int)phrase.size(),
                      ELECTRUM_SALT, 8,
                      2048, EVP_sha512(), 64, seed);

    // BIP32 master key
    static const uint8_t BITCOIN_SEED[] = "Bitcoin seed";
    uint8_t master[64];
    cpu_hmac_sha512(BITCOIN_SEED, 12, seed, 64, master);
    uint8_t mpriv[32], mchain[32];
    memcpy(mpriv,  master,      32);
    memcpy(mchain, master + 32, 32);

    // Electrum : m/0/0 (normal, non hardened)
    uint8_t k0[32], c0[32];
    cpu_bip32_normal(mpriv,  mchain, 0, k0, c0);   // m/0
    cpu_bip32_normal(k0, c0,         0, privkey, c0);  // m/0/0

    return true;
}

static int run_passphrase_mode(const std::string &phrase, const std::string &addr_str, const std::string& rules_file = "", const ResumeState* resume = nullptr) {


    // ── Parse target address ─────────────────────────────────────────────────
    TargetData target = {};
    if(is_bloom_arg(addr_str)){
        target.type = get_bloom_type(addr_str);
        if(!load_bloom_to_target(target)) return 1;
        std::string _bmode=(target.type==TargetType::BLOOM_BTC)?"PASSPHRASE + Bloom BTC":(target.type==TargetType::BLOOM_ETH)?"PASSPHRASE + Bloom ETH":"PASSPHRASE + Bloom";
        std::cout << "Mode : " << _bmode << "\n";
        if (target.type == TargetType::BLOOM)
            std::cout << "Note : 'bloom' in passphrase mode scans only BTC path (m/44'/0'/0'/0/0). Use 'bloometh' for ETH wallets.\n";
    } else if (try_resolve_any_pubkey_target_for_scheduler(addr_str, target)) {
        if (target.type == TargetType::ETH_PUBKEY) {
            std::cout << "Mode : PASSPHRASE / ETH PubKey m/44'/60'/0'/0/0\n";
        } else {
            std::cout << "Mode : PASSPHRASE / BTC PubKey m/44'/0'/0'/0/0\n";
        }
    } else if(addr_str.size()>=2 && addr_str[0]=='0' && (addr_str[1]=='x'||addr_str[1]=='X')){
        target.type = TargetType::ETH;
        if(!ethAddrToBytes(addr_str, target.hash20)){
            std::cerr << "Error: invalid ETH address.\n"; return 1;
        }
        std::cout << "Mode : PASSPHRASE / ETH m/44'/60'/0'/0/0\n";
    } else {
        target.type = TargetType::BTC;
        if(!addrToHash160Any(addr_str, target.hash20)){
            std::cerr << "Error: invalid BTC address.\n"; return 1;
        }
        std::cout << "Mode : PASSPHRASE / BTC m/44'/0'/0'/0/0\n";
    }

    // ── Build mnemonic string from phrase ────────────────────────────────────
    // phrase = "word1 word2 ... word12" (all known, no #)
    auto wmap = build_word_map();
    SeedMask mask = {};
    if(!parse_seed_mask(phrase, mask, wmap)) return 1;

    // Rebuild mnemonic string (PBKDF2 password)
    // Using CPU-side build_mnemonic (simple CPU version)
    std::string mnemonic_str;
    {
        std::istringstream ss(phrase);
        std::string tok;
        bool first = true;
        while(ss >> tok) {
            if(!first) mnemonic_str += ' ';
            mnemonic_str += tok;
            first = false;
        }
    }
    const uint32_t mnemonic_len = (uint32_t)mnemonic_str.size();

    ResumeState rs = make_resume_state("passphrase", phrase, addr_str);

    // ── Open dictionary.txt ──────────────────────────────────────────────
    FILE* dict = fopen(DICT_FILE, "rb");
    if(!dict){
        std::cerr << "Error: cannot open " << DICT_FILE << "\n";
        return 1;
    }

    // No line count needed for text stream. We will estimate ETA by bytes.
    fseek(dict, 0, SEEK_END);
    uint64_t total_bytes = (uint64_t)ftell(dict);
    rewind(dict);
    std::cout << "Dictionary : " << DICT_FILE << " (" << total_bytes << " bytes)\n";
    TextBatchReader dict_reader(dict);

    std::vector<GpuRule> h_rules;
    std::vector<std::string> h_rule_strings;
    if (!rules_file.empty()) {
        if (!RuleParser::parse_rule_file(rules_file, h_rules, &h_rule_strings)) return 1;
    }
    int num_rules = (int)h_rules.size();
    GpuRule* d_rules = nullptr;
    if (num_rules > 0) {
        cudaError_t err = cudaMalloc(&d_rules, num_rules * sizeof(GpuRule));
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: cudaMalloc failed for rules: " << cudaGetErrorString(err) << "\n";
            return 1;
        }
        err = cudaMemcpy(d_rules, h_rules.data(), num_rules * sizeof(GpuRule), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: cudaMemcpy failed for rules: " << cudaGetErrorString(err) << "\n";
            cudaFree(d_rules);
            return 1;
        }
    }

    // ── GPU setup ────────────────────────────────────────────────────────────
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    const int THREADS = 256;
    const int blocks_pass = (prop.multiProcessorCount * 2048) / THREADS; // ~occupancy max
    
    size_t free_vram, total_vram;
    cudaMemGetInfo(&free_vram, &total_vram);
    size_t usable_vram = (free_vram > 512*1024*1024) ? (free_vram - 512*1024*1024) : 0;
    size_t max_items_from_vram = usable_vram / 128; // ~128 bytes per item
    
    int max_allowed_items = std::max((int)PASS_BATCH, 8388608);
    if ((size_t)max_allowed_items > max_items_from_vram) {
        max_allowed_items = (int)max_items_from_vram;
    }
    
    int effective_wave = max_allowed_items / std::max(1, num_rules);
    if (effective_wave < 1) effective_wave = 1;
    
    if (effective_wave < 256) {
        int target_items = 256 * std::max(1, num_rules);
        if ((size_t)target_items <= max_items_from_vram) {
            effective_wave = 256;
        } else {
            effective_wave = std::max(1, (int)(max_items_from_vram / std::max(1, num_rules)));
        }
    }
    const int max_items = effective_wave * std::max(1, num_rules);

    const size_t max_chunk_size = 16 * 1024 * 1024; // 16 MB chunk

    // Allocate device buffers
    uint8_t  *d_mnemonic    = nullptr;
    uint8_t  *d_chunk       = nullptr;
    uint32_t *d_offsets     = nullptr;
    uint8_t  *d_seeds       = nullptr;
    uint8_t  *d_intermed    = nullptr;
    uint8_t  *d_rejected    = nullptr;
    TargetData *d_target    = nullptr;
    HydraResult *d_result   = nullptr;

    cudaMalloc(&d_mnemonic,    mnemonic_len);
    cudaMalloc(&d_chunk,       max_chunk_size);
    cudaMalloc(&d_offsets,     (size_t)effective_wave * sizeof(uint32_t));
    cudaMalloc(&d_seeds,       (size_t)max_items * 64);
    cudaMalloc(&d_intermed,    (size_t)max_items * 64);
    cudaMalloc(&d_rejected,    (size_t)max_items * sizeof(uint8_t));
    cudaMalloc(&d_target,      sizeof(TargetData));
    cudaMalloc(&d_result,      sizeof(HydraResult));

    const int pass_wcomb_w = 14;
    uint64_t* d_pass_comb_GX = nullptr;
    uint64_t* d_pass_comb_GY = nullptr;
    int pass_comb_cols = 0, pass_comb_stride = 0;
    if (!brain_gen_table(pass_wcomb_w, &d_pass_comb_GX, &d_pass_comb_GY, &pass_comb_cols, &pass_comb_stride)) return 1;

    uint8_t  *d_nodes_tmp   = nullptr;
    EccAffinePoint* d_affine = nullptr;
    BrainJacobian* d_jac = nullptr;
    bw_u256* d_local_except = nullptr;
    bw_u256* d_block_prods = nullptr;
    bw_u256* d_block_inv = nullptr;
    uint8_t* d_z_zero = nullptr;

    cudaMalloc(&d_nodes_tmp, (size_t)max_items * 64);
    cudaMalloc(&d_affine, (size_t)max_items * sizeof(EccAffinePoint));
    cudaMalloc(&d_jac, (size_t)max_items * sizeof(BrainJacobian));
    cudaMalloc(&d_local_except, (size_t)max_items * sizeof(bw_u256));
    const int max_blk_inv = (max_items + BW_THREADS_INV - 1) / BW_THREADS_INV;
    cudaMalloc(&d_block_prods, (size_t)max_blk_inv * sizeof(bw_u256));
    cudaMalloc(&d_block_inv, (size_t)max_blk_inv * sizeof(bw_u256));
    cudaMalloc(&d_z_zero, (size_t)max_items * sizeof(uint8_t));

    // Upload mnemonic + target
    cudaMemcpy(d_mnemonic, mnemonic_str.c_str(), mnemonic_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, &target, sizeof(TargetData), cudaMemcpyHostToDevice);

    // Initialize BIP39 dictionary in constant memory
    cudaMemcpyToSymbol(d_BIP39_BLOB, h_BIP39_BLOB, sizeof(h_BIP39_BLOB));
    cudaMemcpyToSymbol(d_BIP39_OFFS, h_BIP39_OFFS, sizeof(h_BIP39_OFFS));
    cudaMemcpyToSymbol(d_BIP39_LENS, h_BIP39_LENS, sizeof(h_BIP39_LENS));

    // Precompute H_ipad/H_opad from mnemonic (1 thread, once)
    hydra_passphrase_setup<<<1, 1>>>(d_mnemonic, mnemonic_len);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "CUDA Error (passphrase setup): " << cudaGetErrorString(cudaGetLastError()) << "\n";
        return 1;
    }

    HydraResult h_res = {};
    cudaMemcpy(d_result, &h_res, sizeof(HydraResult), cudaMemcpyHostToDevice);

    std::cout << "======== HYDRA V5 (PASSPHRASE MODE) ====================\n";
    std::cout << "GPU          : " << prop.name << " (" << prop.multiProcessorCount << " SM)\n";
    std::cout << "Mnemonic     : " << mnemonic_str << "\n";
    std::cout << "K2a PBKDF2   : " << blocks_pass << " blocks x " << THREADS << "\n";
    std::cout << "K2b hardened : " << blocks_pass << " blocks x " << THREADS << "\n";
    std::cout << "K2c ECC      : wCOMB w=" << pass_wcomb_w 
              << " cols=" << pass_comb_cols << " stride=" << pass_comb_stride 
              << " | multi-kernel batch inverse\n";

    // ── Host buffers for dictionary reading ──────────────────────────────────
    uint8_t*  h_chunk[2]   = {nullptr, nullptr};
    uint32_t* h_offsets[2] = {nullptr, nullptr};
    const size_t h_chunk_bytes   = max_chunk_size;
    const size_t h_offsets_bytes = (size_t)effective_wave * sizeof(uint32_t);
    for (int b = 0; b < 2; ++b) {
        cudaError_t e1 = cudaMallocHost((void**)&h_chunk[b], h_chunk_bytes);
        cudaError_t e2 = cudaMallocHost((void**)&h_offsets[b], h_offsets_bytes);
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
            std::cerr << "CUDA Error: cudaMallocHost failed for passphrase buffers\n";
            for (int i = 0; i < 2; ++i) {
                if (h_chunk[i]) cudaFreeHost(h_chunk[i]);
                if (h_offsets[i]) cudaFreeHost(h_offsets[i]);
            }
            fclose(dict);
            cudaFree(d_mnemonic); cudaFree(d_chunk); cudaFree(d_offsets);
            cudaFree(d_seeds); cudaFree(d_intermed); cudaFree(d_rejected);
            cudaFree(d_nodes_tmp); cudaFree(d_affine); cudaFree(d_jac); cudaFree(d_local_except);
            cudaFree(d_block_prods); cudaFree(d_block_inv); cudaFree(d_z_zero); cudaFree(d_pass_comb_GX); cudaFree(d_pass_comb_GY);
            if(target.d_bloom_filter && (target.type == TargetType::BLOOM || target.type == TargetType::BLOOM_BTC || target.type == TargetType::BLOOM_ETH)) cudaFree((void*)target.d_bloom_filter);
            cudaFree(d_target); cudaFree(d_result);
            return 1;
        }
    }
    int      h_count[2] = {0, 0};
    uint64_t h_start_pos[2] = {0, 0};
    uint64_t h_end_pos[2] = {0, 0};
    size_t   h_chunk_size[2] = {0, 0};
    bool     h_eof[2] = {false, false};

    auto fill_pass_batch = [&](int slot) {
        bool hit_eof = false;
        int count = dict_reader.fill_batch(h_chunk[slot], h_offsets[slot], effective_wave, max_chunk_size,
                                           h_start_pos[slot], h_end_pos[slot], hit_eof, h_chunk_size[slot]);
        h_count[slot] = count;
        h_end_pos[slot] = h_end_pos[slot];
        h_eof[slot] = hit_eof;
    };

    // ── Main loop ────────────────────────────────────────────────────────────
    auto t0     = std::chrono::high_resolution_clock::now();
    auto t_last = t0;
    auto t_resume_last = t0;
    uint64_t total_tested      = 0;  // cumulative (includes previous runs)
    uint64_t tested_this_run   = 0;  // only this run (for accurate speed/ETA)
    uint64_t since_last        = 0;
    int      found         = 0;
    [[maybe_unused]] bool eof = false;
    int      last_batch_slot = 0;
    int      last_batch_count = 0;
    cudaStream_t pass_stream = nullptr;
    cudaStreamCreateWithFlags(&pass_stream, cudaStreamNonBlocking);
    uint64_t prof_candidates = 0, prof_batches = 0;
    double prof_h2d_ms = 0.0, prof_k2a_ms = 0.0, prof_k2b_ms = 0.0, prof_k2c_ms = 0.0;
    cudaEvent_t ev_h2d0, ev_h2d1, ev_k2a0, ev_k2a1, ev_k2b0, ev_k2b1, ev_k2c0, ev_k2c1;
    cudaEventCreate(&ev_h2d0); cudaEventCreate(&ev_h2d1);
    cudaEventCreate(&ev_k2a0); cudaEventCreate(&ev_k2a1);
    cudaEventCreate(&ev_k2b0); cudaEventCreate(&ev_k2b1);
    cudaEventCreate(&ev_k2c0); cudaEventCreate(&ev_k2c1);

    if (resume && resume->active) {
        total_tested = resume->tested;
        rs.tested = total_tested;
        rs.dict_byte_offset = resume->dict_byte_offset;
        if (!dict_reader.seek(resume->dict_byte_offset)) {
            std::cerr << "[Resume] Cannot seek in dictionary, restarting from beginning.\n";
            dict_reader.seek(0);
            total_tested = 0;
            rs.tested = 0;
            rs.dict_byte_offset = 0;
        } else {
            std::cout << "[Resume] PASSPHRASE at dictionary byte " << resume->dict_byte_offset
                      << " | tested " << total_tested << " candidates\n";
        }
    }
    write_resume_snapshot(rs);

    fill_pass_batch(0);
    int cur_slot = 0;
    int next_slot = 1;
    int batch_count = h_count[cur_slot];
    while(!g_sigint && !found && batch_count > 0){
        last_batch_slot = cur_slot;
        last_batch_count = batch_count;
        const uint64_t batch_end_pos = h_end_pos[cur_slot];

        std::thread reader_thread;
        bool reader_started = false;
        if (!h_eof[cur_slot]) {
            reader_started = true;
            reader_thread = std::thread(fill_pass_batch, next_slot);
        } else {
            eof = true;
        }

        // Upload batch
        cudaEventRecord(ev_h2d0, pass_stream);
        cudaMemcpyAsync(d_chunk, h_chunk[cur_slot],
                        h_chunk_size[cur_slot], cudaMemcpyHostToDevice, pass_stream);
        cudaMemcpyAsync(d_offsets, h_offsets[cur_slot],
                        (size_t)batch_count * sizeof(uint32_t), cudaMemcpyHostToDevice, pass_stream);
        cudaEventRecord(ev_h2d1, pass_stream);

        int total_batch_count = batch_count * std::max(1, num_rules);
        int blk_total = (total_batch_count + THREADS - 1) / THREADS;

        // K2a : passphrase -> seed[64]
        cudaEventRecord(ev_k2a0, pass_stream);
        hydra_k2a_passphrase<<<blk_total, THREADS, 0, pass_stream>>>(
            d_chunk, d_offsets, batch_count, d_rules, num_rules, d_rejected, d_seeds);
        cudaEventRecord(ev_k2a1, pass_stream);

        std::vector<uint32_t> coins_to_check;
        if (target.type == TargetType::BLOOM) coins_to_check = {0u, 60u};
        else coins_to_check = {(target.type == TargetType::ETH || target.type == TargetType::BLOOM_ETH || target.type == TargetType::ETH_PUBKEY) ? 60u : 0u};

        for (uint32_t coin_type : coins_to_check) {
            // K2b : seed -> priv||chain after m/44'/coin'/0'
            cudaEventRecord(ev_k2b0, pass_stream);
            hydra_k2b_hardened<<<blk_total, THREADS, 0, pass_stream>>>(
                coin_type, d_seeds, d_intermed, total_batch_count);
            cudaEventRecord(ev_k2b1, pass_stream);

            // K2c : normal derivations + ECC + Hash + compare
            cudaEventRecord(ev_k2c0, pass_stream);
            {
                const int blk_ecc = (total_batch_count + BW_THREADS_ECC - 1) / BW_THREADS_ECC;
                
                auto nodes_to_affine = [&](const uint8_t* nodes) {
                    kernel_ecc_nodes_to_jac_t<14><<<blk_ecc, BW_THREADS_ECC, 0, pass_stream>>>(
                        nodes, total_batch_count, d_pass_comb_GX, d_pass_comb_GY, d_jac);
                    hydra_mark_rejected_jac<<<blk_total, THREADS, 0, pass_stream>>>(
                        d_rejected, d_jac, total_batch_count);
                        
                    const int blk_inv = (total_batch_count + BW_THREADS_INV - 1) / BW_THREADS_INV;
                    const int blk_inv2 = (blk_inv + 127) / 128;
                    const size_t smem_local = 2 * BW_THREADS_INV * sizeof(bw_u256);
                    const size_t smem_inv = (128 + 4) * sizeof(bw_u256);
                    
                    kernel_bw_local_prod<<<blk_inv, BW_THREADS_INV, smem_local, pass_stream>>>(
                        d_jac, total_batch_count, d_local_except, d_block_prods, d_z_zero);
                    kernel_bw_invert_blocks<<<blk_inv2, 128, smem_inv, pass_stream>>>(
                        d_block_prods, d_block_inv, blk_inv);
                    kernel_ecc_affine_from_jac<<<blk_inv, BW_THREADS_INV, 0, pass_stream>>>(
                        d_jac, d_local_except, d_block_inv, d_z_zero, total_batch_count, d_affine);
                };

                nodes_to_affine(d_intermed);
                kernel_bip32_derive_normal_from_affine<<<blk_total, THREADS, 0, pass_stream>>>(
                    d_intermed, d_affine, 0, total_batch_count, d_nodes_tmp);
                    
                nodes_to_affine(d_nodes_tmp);
                kernel_bip32_derive_normal_from_affine<<<blk_total, THREADS, 0, pass_stream>>>(
                    d_nodes_tmp, d_affine, 0, total_batch_count, d_intermed);
                    
                {
                    kernel_ecc_nodes_to_jac_t<14><<<blk_ecc, BW_THREADS_ECC, 0, pass_stream>>>(
                        d_intermed, total_batch_count, d_pass_comb_GX, d_pass_comb_GY, d_jac);
                    hydra_mark_rejected_jac<<<blk_total, THREADS, 0, pass_stream>>>(
                        d_rejected, d_jac, total_batch_count);
                        
                    const int blk_inv = (total_batch_count + BW_THREADS_INV - 1) / BW_THREADS_INV;
                    const int blk_inv2 = (blk_inv + 127) / 128;
                    const size_t smem_local = 2 * BW_THREADS_INV * sizeof(bw_u256);
                    const size_t smem_inv = (128 + 4) * sizeof(bw_u256);
                    
                    kernel_bw_local_prod<<<blk_inv, BW_THREADS_INV, smem_local, pass_stream>>>(
                        d_jac, total_batch_count, d_local_except, d_block_prods, d_z_zero);
                    kernel_bw_invert_blocks<<<blk_inv2, 128, smem_inv, pass_stream>>>(
                        d_block_prods, d_block_inv, blk_inv);
                    kernel_ecc_finalize_from_jac_index<<<blk_inv, BW_THREADS_INV, 0, pass_stream>>>(
                        d_jac, d_local_except, d_block_inv, d_z_zero, nullptr, total_batch_count, d_target, d_result);
                }
            }
            cudaEventRecord(ev_k2c1, pass_stream);

            cudaError_t err_last = cudaGetLastError();
            cudaError_t err_sync = cudaStreamSynchronize(pass_stream);
            if (err_last != cudaSuccess || err_sync != cudaSuccess) {
                cudaError_t err = (err_last != cudaSuccess) ? err_last : err_sync;
                std::cerr << "CUDA Error (k2c_ecc_pass): " << cudaGetErrorString(err) << "\n";
                if (reader_started) reader_thread.join();
                break;
            }
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev_k2b0, ev_k2b1); prof_k2b_ms += ms;
            cudaEventElapsedTime(&ms, ev_k2c0, ev_k2c1); prof_k2c_ms += ms;
            cudaMemcpyAsync(&found, &d_result->found, sizeof(int), cudaMemcpyDeviceToHost, pass_stream);
            cudaStreamSynchronize(pass_stream);
            if (found) break;
        }

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_h2d0, ev_h2d1); prof_h2d_ms += ms;
        cudaEventElapsedTime(&ms, ev_k2a0, ev_k2a1); prof_k2a_ms += ms;
        prof_candidates += (uint64_t)total_batch_count;
        prof_batches++;

        // Bloom mode : passphrase hit -> derive privkey CPU -> check balance
        if(found && (target.type==TargetType::BLOOM||target.type==TargetType::BLOOM_BTC||target.type==TargetType::BLOOM_ETH)) {
            cudaMemcpyAsync(&h_res, d_result, sizeof(HydraResult), cudaMemcpyDeviceToHost, pass_stream);
            cudaStreamSynchronize(pass_stream);
            uint64_t local_idx = h_res.index;
            std::string hit_pass;
            std::string hit_rule;
            if(local_idx < (uint64_t)total_batch_count){
                int rule_idx = num_rules > 0 ? (local_idx % num_rules) : 0;
                int word_idx = num_rules > 0 ? (local_idx / num_rules) : local_idx;
                uint32_t offset = h_offsets[cur_slot][word_idx];
                uint32_t pass_len = h_chunk[cur_slot][offset];
                const uint8_t* p = h_chunk[cur_slot] + offset + 1;
                
                if (num_rules > 0) {
                    uint8_t local_pass[MAX_PASS_LEN];
                    uint32_t out_len = 0;
                    bool valid_rule = apply_rule(p, pass_len, local_pass, out_len, &h_rules[rule_idx], MAX_PASS_LEN);
                    hit_pass = valid_rule ? std::string((const char*)local_pass, out_len) : std::string("<rejected>");
                    hit_rule = h_rule_strings[rule_idx];
                } else {
                    hit_pass = std::string((const char*)p, pass_len);
                }
            }
            std::cout << "\n!!! BLOOM HIT !!!\n  Mnemonic   : " << mnemonic_str
                      << "\n  Passphrase : \"" << hit_pass << "\"\n";
            if (!hit_rule.empty()) std::cout << "  Rule       : " << hit_rule << "\n";
            std::cout << "  [CPU] BIP44 derivation...\n";

            // Derive privkey CPU (BTC coin=0, ETH coin=60)
            // For BLOOM_BTC -> BTC derivation only
            // For BLOOM_ETH -> ETH derivation only
            // For BLOOM    -> try BTC first, then ETH if no balance
            bool victory = false;

            if (target.type != TargetType::BLOOM_ETH) {
                uint8_t privkey_btc[32];
                cpu_derive_key(mnemonic_str, hit_pass, 0, privkey_btc);
                std::string addr_legacy, addr_segwit, addr_eth;
                key_to_addresses(privkey_btc, addr_legacy, addr_segwit, addr_eth);
                victory = check_balances_and_notify(privkey_btc, addr_legacy, addr_segwit, addr_eth);
            }
            if (!victory && target.type != TargetType::BLOOM_BTC) {
                uint8_t privkey_eth[32];
                cpu_derive_key(mnemonic_str, hit_pass, 60, privkey_eth);
                std::string addr_legacy, addr_segwit, addr_eth;
                key_to_addresses(privkey_eth, addr_legacy, addr_segwit, addr_eth);
                victory = check_balances_and_notify(privkey_eth, addr_legacy, addr_segwit, addr_eth);
            }
            if (!victory) {
                // False positive -> reset and continue
                int zero = 0;
                cudaMemcpyAsync(&d_result->found, &zero, sizeof(int), cudaMemcpyHostToDevice, pass_stream);
                cudaStreamSynchronize(pass_stream);
                found = 0;
            }
            // If victory=true -> found stays 1 -> exits loop -> VICTORY block below
        }

        total_tested      += total_batch_count;
        tested_this_run   += total_batch_count;
        rs.tested = total_tested;
        rs.dict_byte_offset = batch_end_pos;
        since_last   += total_batch_count;

        auto now = std::chrono::high_resolution_clock::now();
        const bool force_resume = found || g_sigint || batch_count == 0;
        if (should_write_resume_snapshot(t_resume_last, now, force_resume)) {
            double prog = total_bytes > 0 ? 100.0 * (double)rs.dict_byte_offset / (double)total_bytes : 0.0;
            double elapsed = std::chrono::duration<double>(now - t0).count();
            double eta = (elapsed > 0 && rs.dict_byte_offset > 0)
                ? (double)(total_bytes - rs.dict_byte_offset) / ((double)rs.dict_byte_offset / elapsed) : 0;
            write_resume_snapshot(rs);
            t_resume_last = now;
        }
        double dt = std::chrono::duration<double>(now - t_last).count();
        if(dt >= 1.0){
            double speed   = since_last / dt / 1e6;
            double elapsed = std::chrono::duration<double>(now - t0).count();
            double prog    = (total_bytes > 0) ? 100.0 * (double)rs.dict_byte_offset / (double)total_bytes : 0.0;
            // ETA uses dict_byte_offset/elapsed for accurate byte processing speed
            double bytes_per_sec = (elapsed > 0) ? (double)rs.dict_byte_offset / elapsed : 0;
            double eta     = (bytes_per_sec > 0 && total_bytes > rs.dict_byte_offset)
                ? (double)(total_bytes - rs.dict_byte_offset) / bytes_per_sec : 0;
            int eh=(int)(eta/3600), em=(int)((eta-eh*3600)/60), es=(int)((long long)eta%60);
            std::cout << "\r[" << std::fixed << std::setprecision(1) << prog << "%] "
                      << std::setprecision(2) << speed << " MKey/s"
                      << " | ETA " << std::setfill('0')
                      << std::setw(2)<<eh<<":"<<std::setw(2)<<em<<":"<<std::setw(2)<<es
                      << std::flush;
            t_last = now; since_last = 0;
        }

        if (reader_started) {
            reader_thread.join();
            if (!found && !g_sigint) {
                cur_slot = next_slot;
                next_slot = 1 - cur_slot;
                batch_count = h_count[cur_slot];
                if (batch_count == 0) eof = true;
            }
        } else {
            eof = true;
            batch_count = 0;
        }
    }
    fclose(dict);

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(t_end - t0).count();
    double avg_speed = (total_elapsed > 0) ? (double)total_tested / total_elapsed / 1e6 : 0;
    std::cout << "\nTime    : " << std::fixed << std::setprecision(2) << total_elapsed << " s"
              << " | " << std::setprecision(2) << avg_speed << " MKey/s avg"
              << " | Tested  : " << total_tested << "\n";
    if (prof_batches > 0 && prof_candidates > 0) {
        auto rate = [&](double ms) -> double {
            return (ms > 0.0) ? ((double)prof_candidates / ms / 1000.0) : 0.0;
        };
        const double gpu_ms = prof_h2d_ms + prof_k2a_ms + prof_k2b_ms + prof_k2c_ms;
        std::cout << "[Profile] batches=" << prof_batches
                  << " candidates=" << prof_candidates
                  << " gpu+copy=" << std::fixed << std::setprecision(2) << gpu_ms << " ms\n";
        std::cout << "  H2D copy    : " << std::setw(9) << prof_h2d_ms
                  << " ms | " << std::setw(7) << rate(prof_h2d_ms) << " Mpass/s\n";
        std::cout << "  K2a PBKDF2  : " << std::setw(9) << prof_k2a_ms
                  << " ms | " << std::setw(7) << rate(prof_k2a_ms) << " Mpass/s\n";
        std::cout << "  K2b hardened: " << std::setw(9) << prof_k2b_ms
                  << " ms | " << std::setw(7) << rate(prof_k2b_ms) << " Mpass/s\n";
        std::cout << "  K2c ECC     : " << std::setw(9) << prof_k2c_ms
                  << " ms | " << std::setw(7) << rate(prof_k2c_ms) << " Mpass/s\n";
    }

    if(found){
        clear_resume_snapshot();
        cudaMemcpyAsync(&h_res, d_result, sizeof(HydraResult), cudaMemcpyDeviceToHost, pass_stream);
        cudaStreamSynchronize(pass_stream);
        // result->index = position within current batch
        // Retrieve passphrase from host buffer (still in memory)
        uint64_t local_idx = h_res.index;
        std::string found_pass;
        std::string hit_rule;
        if(local_idx < (uint64_t)(last_batch_count * std::max(1, num_rules))){
            int rule_idx = num_rules > 0 ? (local_idx % num_rules) : 0;
            int word_idx = num_rules > 0 ? (local_idx / num_rules) : local_idx;
            uint32_t offset = h_offsets[last_batch_slot][word_idx];
            uint32_t pass_len = h_chunk[last_batch_slot][offset];
            const uint8_t* p = h_chunk[last_batch_slot] + offset + 1;
            
            if (num_rules > 0) {
                uint8_t local_pass[MAX_PASS_LEN];
                uint32_t out_len = 0;
                bool valid_rule = apply_rule(p, pass_len, local_pass, out_len, &h_rules[rule_idx], MAX_PASS_LEN);
                found_pass = valid_rule ? std::string((const char*)local_pass, out_len) : std::string("<rejected>");
                hit_rule = h_rule_strings[rule_idx];
            } else {
                found_pass = std::string((const char*)p, pass_len);
            }
        }
        std::cout << "\n======== VICTORY ! PASSPHRASE FOUND ==================\n";
        std::cout << "Mnemonic   : " << mnemonic_str << "\n";
        std::cout << "Passphrase : \"" << found_pass << "\"\n";
        if (!hit_rule.empty()) std::cout << "Rule       : " << hit_rule << "\n";
        std::cout << "=======================================================\n";
        {
            std::string key_info = "*Mnemonic:*\n`" + mnemonic_str + "`\n\n"
                                 + "*Passphrase:*\n`" + found_pass + "`";
            // Bloom mode : addr_str = "bloom/bloombtc/bloometh" -> not useful to display
            // Derive the real address CPU-side for the Telegram message
            std::string addr_info;
            if (is_bloom_arg(addr_str)) {
                // Derive the real address from the recovered key
                uint32_t coin = (target.type == TargetType::BLOOM_ETH) ? 60u : 0u;
                uint8_t privkey[32];
                cpu_derive_key(mnemonic_str, found_pass, coin, privkey);
                std::string al, as_, ae;
                key_to_addresses(privkey, al, as_, ae);
                if (target.type == TargetType::BLOOM_ETH)
                    addr_info = "*ETH:* `" + ae + "`";
                else
                    addr_info = "*BTC legacy:* `" + al + "`\n*BTC segwit:* `" + as_ + "`";
            } else {
                addr_info = addr_str.empty() ? "" : ("*Address:* `" + addr_str + "`");
            }
            notify_victory("PASSPHRASE FOUND \xF0\x9F\x94\x91", key_info, addr_info);
        }
    } else if (!g_sigint) {
        clear_resume_snapshot();
        std::cout << "Passphrase not found in " << DICT_FILE << "\n";
    } else {
        write_resume_snapshot(rs);
        print_resume_hint();
    }
    print_search_summary(found != 0);

    for(int i=0;i<2;++i){ cudaFreeHost(h_chunk[i]); cudaFreeHost(h_offsets[i]); }
    cudaFree(d_mnemonic); cudaFree(d_chunk); cudaFree(d_offsets);
    cudaFree(d_seeds); cudaFree(d_intermed); cudaFree(d_rejected);
    cudaFree(d_nodes_tmp); cudaFree(d_affine); cudaFree(d_jac); cudaFree(d_local_except);
    cudaFree(d_block_prods); cudaFree(d_block_inv); cudaFree(d_z_zero); cudaFree(d_pass_comb_GX); cudaFree(d_pass_comb_GY);
    cudaFree(d_rules);
    if(target.d_bloom_filter && (target.type == TargetType::BLOOM || target.type == TargetType::BLOOM_BTC || target.type == TargetType::BLOOM_ETH)) cudaFree((void*)target.d_bloom_filter);
    cudaFree(d_target); cudaFree(d_result);
    cudaEventDestroy(ev_h2d0); cudaEventDestroy(ev_h2d1);
    cudaEventDestroy(ev_k2a0); cudaEventDestroy(ev_k2a1);
    cudaEventDestroy(ev_k2b0); cudaEventDestroy(ev_k2b1);
    cudaEventDestroy(ev_k2c0); cudaEventDestroy(ev_k2c1);
    cudaStreamDestroy(pass_stream);
    return found ? 0 : 1;
}

// =================================================================================
// BRAINWALLET MODE
// =================================================================================

#define BRAIN_FILE "resources/brainwallet.txt"

// Choose the largest wCOMB window that fits in VRAM budget
static int brain_choose_w() {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        return 17;
    }
    const uint64_t budget = hydra_vram_budget_bytes(free_b);
    const uint64_t host_avail = hydra_host_available_ram_bytes();
    // Brainwallet keys are SHA256(passphrase) — fully random. Each wCOMB column
    // lookup is a guaranteed DRAM miss (no correlation between threads → no L2 reuse).
    // Keep only the first window of each useful column-count plateau.
    for (int w : {22, 19}) {
        const int cols   = (256 + w - 1) / w;
        const int stride = 1 << (w - 1);
        const uint64_t bytes = 2ULL * cols * stride * 4 * 8;
        const bool fits_vram = bytes <= budget;
        const bool fits_host = (host_avail == 0) || (bytes <= (host_avail * 3ULL) / 4ULL);
        if (fits_vram && fits_host) return w;
    }
    return 17;
}

// CPU table generator using OpenSSL — fills d_gx_out/d_gy_out on GPU
static bool brain_gen_table(int w,
    uint64_t** d_gx_out, uint64_t** d_gy_out,
    int* cols_out, int* stride_out)
{
    const int cols   = (256 + w - 1) / w;
    const int stride = 1 << (w - 1);
    const size_t total = (size_t)cols * stride;
    const size_t plane_bytes = total * 4 * sizeof(uint64_t);

    std::vector<uint64_t> h_gx(total * 4);
    std::vector<uint64_t> h_gy(total * 4);

    char cache_file[64];
    snprintf(cache_file, sizeof(cache_file), "resources/wcomb_table_w%d.bin", w);
    
    bool loaded_from_cache = false;
    FILE* f_cache = fopen(cache_file, "rb");
    if (f_cache) {
        fseek(f_cache, 0, SEEK_END);
        size_t fsize = ftell(f_cache);
        fseek(f_cache, 0, SEEK_SET);
        if (fsize == 2 * plane_bytes) {
            std::cout << "[wCOMB] Loading cached table (w=" << w
                      << ", cols=" << cols << ", stride=" << stride
                      << ", " << (2 * plane_bytes / 1024 / 1024) << " MB)... " << std::flush;
            auto t0 = std::chrono::high_resolution_clock::now();
            if (fread(h_gx.data(), 1, plane_bytes, f_cache) == plane_bytes &&
                fread(h_gy.data(), 1, plane_bytes, f_cache) == plane_bytes) {
                loaded_from_cache = true;
            }
            auto dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
            if (loaded_from_cache) {
                std::cout << "done (" << std::fixed << std::setprecision(2) << dt << " s)\n";
            } else {
                std::cout << "failed to read. Rebuilding...\n";
            }
        }
        fclose(f_cache);
    }

    if (!loaded_from_cache) {
        std::cout << "[wCOMB] Building table (w=" << w
                  << ", cols=" << cols << ", stride=" << stride
                  << ", " << (2 * plane_bytes / 1024 / 1024) << " MB)... " << std::flush;
        auto t0 = std::chrono::high_resolution_clock::now();

    EC_GROUP* grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX*   ctx = BN_CTX_new();

    // Lambda: big-endian BIGNUM → LE uint64_t[4] at position dst*4 in plane
    BIGNUM* bx = BN_new(); BIGNUM* by = BN_new();
    auto store_point = [&](const EC_POINT* P, size_t dst) {
        EC_POINT_get_affine_coordinates(grp, P, bx, by, ctx);
        uint8_t xb[32]={}, yb[32]={};
        BN_bn2binpad(bx, xb, 32);
        BN_bn2binpad(by, yb, 32);
        for (int lim=0; lim<4; lim++) {
            const int bs = (3-lim)*8;  // MSB first: lim0→bytes24-31, lim3→bytes0-7
            uint64_t vx=0, vy=0;
            for (int b=0;b<8;b++) {
                vx |= (uint64_t)xb[bs+b] << (56-b*8);
                vy |= (uint64_t)yb[bs+b] << (56-b*8);
            }
            h_gx[dst*4+lim] = vx;
            h_gy[dst*4+lim] = vy;
        }
    };

    // Batch size for EC_POINTs_make_affine (limits peak memory)
    const int BATCH = 8192;
    std::vector<EC_POINT*> bpts(BATCH);
    for (int i=0;i<BATCH;i++) bpts[i] = EC_POINT_new(grp);
    EC_POINT* base   = EC_POINT_new(grp);
    EC_POINT* cursor = EC_POINT_new(grp);

    // Start base = G (generator)
    EC_POINT_copy(base, EC_GROUP_get0_generator(grp));

    for (int col=0; col<cols; col++) {
        EC_POINT_copy(cursor, base);   // cursor = 1 * base_col

        for (int bstart=0; bstart<stride; bstart+=BATCH) {
            const int bsz = std::min(BATCH, stride - bstart);

            // Fill batch: bpts[i] = (bstart+i+1)*base_col
            EC_POINT_copy(bpts[0], cursor);
            for (int i=1;i<bsz;i++)
                EC_POINT_add(grp, bpts[i], bpts[i-1], base, ctx);

            // Advance cursor to start of next batch (before making affine)
            EC_POINT_add(grp, cursor, bpts[bsz-1], base, ctx);

            // Batch affine conversion (1 modular inverse for the whole batch)
#if defined(__GNUC__) && !defined(_MSC_VER)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
            EC_POINTs_make_affine(grp, bsz, bpts.data(), ctx);
#if defined(__GNUC__) && !defined(_MSC_VER)
#pragma GCC diagnostic pop
#endif

            // Extract coordinates
            for (int i=0;i<bsz;i++)
                store_point(bpts[i], (size_t)col*stride + bstart + i);
        }

        // Advance base to next column: base *= 2^w
        for (int d=0;d<w;d++) EC_POINT_dbl(grp, base, base, ctx);
    }

        BN_free(bx); BN_free(by);
        for (int i=0;i<BATCH;i++) EC_POINT_free(bpts[i]);
        EC_POINT_free(base); EC_POINT_free(cursor);
        BN_CTX_free(ctx); EC_GROUP_free(grp);

        auto dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
        std::cout << "done (" << std::fixed << std::setprecision(1) << dt << " s)\n";

        // Save to cache
        f_cache = fopen(cache_file, "wb");
        if (f_cache) {
            fwrite(h_gx.data(), 1, plane_bytes, f_cache);
            fwrite(h_gy.data(), 1, plane_bytes, f_cache);
            fclose(f_cache);
        }
    }

    // Upload to GPU
    if (cudaMalloc(d_gx_out, plane_bytes) != cudaSuccess ||
        cudaMalloc(d_gy_out, plane_bytes) != cudaSuccess) {
        std::cerr << "\n[wCOMB] cudaMalloc failed for table (" << (plane_bytes/1024/1024) << " MB per plane)\n";
        return false;
    }
    cudaMemcpy(*d_gx_out, h_gx.data(), plane_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(*d_gy_out, h_gy.data(), plane_bytes, cudaMemcpyHostToDevice);

    *cols_out   = cols;
    *stride_out = stride;
    return true;
}

// CPU-side privkey derivation for brainwallet (for bloom-hit verification)
static bool cpu_brain_privkey(const char* pass, uint32_t len, uint8_t privkey[32]) {
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    unsigned int olen = 32;
    EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
    EVP_DigestUpdate(ctx, pass, len);
    EVP_DigestFinal_ex(ctx, privkey, &olen);
    EVP_MD_CTX_free(ctx);
    return true;
}

static void eccdiag_print_u256(const char* label, const bw_u256& x) {
    std::cout << "  " << label << " = 0x";
    for (int i = 3; i >= 0; --i) {
        std::cout << std::hex << std::setw(16) << std::setfill('0') << x.v[i];
    }
    std::cout << std::dec << std::setfill(' ') << "\n";
}

static int run_eccdiag_mode() {
    std::cout << "======== HYDRA ECC DIAG =============================\n";
    const int w = 14;
    uint64_t* d_comb_GX = nullptr;
    uint64_t* d_comb_GY = nullptr;
    int comb_cols = 0, comb_stride = 0;
    if (!brain_gen_table(w, &d_comb_GX, &d_comb_GY, &comb_cols, &comb_stride)) return 1;

    const std::vector<int> sizes = {
        1, 2, 3, 4, 7, 8, 9, 31, 32, 33, 63, 64, 65,
        127, 128, 129, 255, 256, 257, 511, 512, 513,
        1024, 4096, 16337, 16549, 262143, 262333
    };
    const int maxN = *std::max_element(sizes.begin(), sizes.end());
    const int THREADS = 256;
    const int max_blk_inv = (maxN + BW_THREADS_INV - 1) / BW_THREADS_INV;

    uint8_t* d_nodes = nullptr;
    EccAffinePoint* d_local = nullptr;
    EccAffinePoint* d_batch = nullptr;
    BrainJacobian* d_jac = nullptr;
    bw_u256* d_local_except = nullptr;
    bw_u256* d_block_prods = nullptr;
    bw_u256* d_block_inv = nullptr;
    uint8_t* d_z_zero = nullptr;
    EccDiagResult* d_result = nullptr;

    cudaMalloc(&d_nodes, (size_t)maxN * 64);
    cudaMalloc(&d_local, (size_t)maxN * sizeof(EccAffinePoint));
    cudaMalloc(&d_batch, (size_t)maxN * sizeof(EccAffinePoint));
    cudaMalloc(&d_jac, (size_t)maxN * sizeof(BrainJacobian));
    cudaMalloc(&d_local_except, (size_t)maxN * sizeof(bw_u256));
    cudaMalloc(&d_block_prods, (size_t)max_blk_inv * sizeof(bw_u256));
    cudaMalloc(&d_block_inv, (size_t)max_blk_inv * sizeof(bw_u256));
    cudaMalloc(&d_z_zero, (size_t)maxN);
    cudaMalloc(&d_result, sizeof(EccDiagResult));

    int failures = 0;
    for (int N : sizes) {
        const int blk_ecc = (N + BW_THREADS_ECC - 1) / BW_THREADS_ECC;
        const int blk_inv = (N + BW_THREADS_INV - 1) / BW_THREADS_INV;
        const int blk_inv2 = (blk_inv + 127) / 128;
        const int blk_init = (N + THREADS - 1) / THREADS;
        const size_t smem_local = 2 * BW_THREADS_INV * sizeof(bw_u256);
        const size_t smem_inv = (128 + 4) * sizeof(bw_u256);

        cudaMemset(d_result, 0, sizeof(EccDiagResult));
        kernel_eccdiag_make_nodes<<<blk_init, THREADS>>>(d_nodes, N);
        kernel_ecc_nodes_to_jac_t<14><<<blk_ecc, BW_THREADS_ECC>>>(
            d_nodes, N, d_comb_GX, d_comb_GY, d_jac);
        kernel_ecc_affine_local_from_jac<<<blk_inv, BW_THREADS_INV>>>(
            d_jac, N, d_local);
        kernel_bw_local_prod<<<blk_inv, BW_THREADS_INV, smem_local>>>(
            d_jac, N, d_local_except, d_block_prods, d_z_zero);
        kernel_bw_invert_blocks<<<blk_inv2, 128, smem_inv>>>(
            d_block_prods, d_block_inv, blk_inv);
        kernel_ecc_affine_from_jac<<<blk_inv, BW_THREADS_INV>>>(
            d_jac, d_local_except, d_block_inv, d_z_zero, N, d_batch);
        kernel_eccdiag_compare_affine<<<blk_init, THREADS>>>(
            d_local, d_batch, d_jac, d_local_except, d_block_prods, d_block_inv, d_z_zero, N, d_result);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "[eccdiag] CUDA error for N=" << N << ": "
                      << cudaGetErrorString(err) << "\n";
            failures++;
            break;
        }

        EccDiagResult h = {};
        cudaMemcpy(&h, d_result, sizeof(EccDiagResult), cudaMemcpyDeviceToHost);
        if (!h.mismatch) {
            std::cout << "[eccdiag] N=" << N << " OK\n";
            continue;
        }

        failures++;
        std::cout << "[eccdiag] N=" << N << " MISMATCH"
                  << " idx=" << h.idx
                  << " block=" << (h.idx / BW_THREADS_INV)
                  << " lane=" << (h.idx % BW_THREADS_INV)
                  << " kind=" << h.kind << "\n";
        std::cout << "  local.valid=" << h.local.isValid
                  << " batch.valid=" << h.batch.isValid
                  << " jac.valid=" << h.jac.isValid
                  << " z_zero=" << (int)h.z_zero << "\n";
        eccdiag_print_u256("local.x", h.local.x);
        eccdiag_print_u256("batch.x", h.batch.x);
        eccdiag_print_u256("local.y", h.local.y);
        eccdiag_print_u256("batch.y", h.batch.y);
        eccdiag_print_u256("J.x", h.jac.jx);
        eccdiag_print_u256("J.y", h.jac.jy);
        eccdiag_print_u256("J.z", h.jac.jz);
        eccdiag_print_u256("local_except", h.local_except);
        eccdiag_print_u256("block_prod", h.block_prod);
        eccdiag_print_u256("block_inv", h.block_inv);
        eccdiag_print_u256("direct_z_inv", h.direct_z_inv);
        eccdiag_print_u256("batch_z_inv", h.batch_z_inv);
        eccdiag_print_u256("block_check", h.block_check);
        eccdiag_print_u256("z_check", h.z_check);
        break;
    }

    cudaFree(d_nodes); cudaFree(d_local); cudaFree(d_batch); cudaFree(d_jac);
    cudaFree(d_local_except); cudaFree(d_block_prods); cudaFree(d_block_inv);
    cudaFree(d_z_zero); cudaFree(d_result);
    cudaFree(d_comb_GX); cudaFree(d_comb_GY);

    if (failures == 0) {
        std::cout << "[eccdiag] local affine == batch affine for all tested sizes\n";
        return 0;
    }
    return 2;
}

static bool cpu_precompute_C() {
    // M mod N en big-endian bytes
    static const uint8_t M_MOD_N_BE[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        0x45, 0x51, 0x23, 0x19, 0x50, 0xb7, 0x5f, 0xc4,
        0x40, 0x2d, 0xa1, 0x73, 0x2f, 0xc9, 0xbe, 0xbe
    };

    EC_GROUP* grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    if (!grp) { std::cerr << "[C=M·G] EC_GROUP failed\n"; return false; }

    BIGNUM* scalar = BN_bin2bn(M_MOD_N_BE, 32, nullptr);
    if (!scalar) { EC_GROUP_free(grp); std::cerr << "[C=M·G] BN failed\n"; return false; }

    EC_POINT* C = EC_POINT_new(grp);
    EC_POINT_mul(grp, C, scalar, nullptr, nullptr, nullptr);

    BIGNUM *cx = BN_new(), *cy = BN_new();
    BN_CTX* ctx = BN_CTX_new();
    EC_POINT_get_affine_coordinates(grp, C, cx, cy, ctx);

    // Convertir en uint64_t Little-Endian (4 limbs)
    auto bn_to_limbs_le = [](BIGNUM* b, uint64_t limbs[4]) {
        uint8_t buf[32] = {}; BN_bn2binpad(b, buf, 32);
        uint64_t tmp[4];
        for (int i = 0; i < 4; i++) {
            tmp[i] = 0;
            for (int j = 0; j < 8; j++)
                tmp[i] = (tmp[i] << 8) | buf[i*8+j];
        }
        // Inversion LE
        limbs[0] = tmp[3]; limbs[1] = tmp[2];
        limbs[2] = tmp[1]; limbs[3] = tmp[0];
    };

    uint64_t h_Cx[4], h_Cy[4];
    bn_to_limbs_le(cx, h_Cx);
    bn_to_limbs_le(cy, h_Cy);

    BN_free(cx); BN_free(cy); BN_CTX_free(ctx);
    EC_POINT_free(C); BN_free(scalar); EC_GROUP_free(grp);

    // Upload dans constant memory GPU
    if (cudaMemcpyToSymbol(d_Cx, h_Cx, 32) != cudaSuccess) return false;
    if (cudaMemcpyToSymbol(d_Cy, h_Cy, 32) != cudaSuccess) return false;

    std::cout << "[C=M·G] C precomputed and uploaded to constant memory (CP Trick active)\n";
    return true;
}


static int run_brainwallet_mode(const std::string& addr_str, const std::string& rules_file = "", bool use_cp = false, const ResumeState* resume = nullptr) {
    // ── Parse target ─────────────────────────────────────────────────────────
    TargetData target = {};

    if (use_cp) {
        if (!cpu_precompute_C()) {
            std::cerr << "Error: failed to precompute C for the CP Trick.\n";
            return 1;
        }
    }

    if (is_bloom_arg(addr_str)) {
        target.type = get_bloom_type(addr_str);
        if (!load_bloom_to_target(target)) return 1;
        std::cout << "Mode : BRAINWALLET + Bloom\n";
    } else if (try_resolve_any_pubkey_target_for_scheduler(addr_str, target)) {
        if (target.type == TargetType::ETH_PUBKEY) {
            std::cout << "Mode : BRAINWALLET / ETH PubKey\n";
        } else {
            std::cout << "Mode : BRAINWALLET / BTC PubKey\n";
        }
    } else if (addr_str.size()>=2 && addr_str[0]=='0' && (addr_str[1]=='x'||addr_str[1]=='X')) {
        target.type = TargetType::ETH;
        if (!ethAddrToBytes(addr_str, target.hash20)) {
            std::cerr << "Error: invalid ETH address.\n"; return 1;
        }
        std::cout << "Mode : BRAINWALLET / ETH\n";
    } else {
        target.type = TargetType::BTC;
        if (!addrToHash160Any(addr_str, target.hash20)) {
            std::cerr << "Error: invalid BTC address.\n"; return 1;
        }
        std::cout << "Mode : BRAINWALLET / BTC\n";
    }

    // ── Choose wCOMB width and generate table ─────────────────────────────────
    const int w = brain_choose_w();
    uint64_t* d_comb_GX = nullptr;
    uint64_t* d_comb_GY = nullptr;
    int comb_cols = 0, comb_stride = 0;
    if (!brain_gen_table(w, &d_comb_GX, &d_comb_GY, &comb_cols, &comb_stride)) return 1;

    FILE* f = fopen(BRAIN_FILE, "rb");
    if (!f) {
        std::cerr << "Error: cannot open " << BRAIN_FILE << "\n";
        cudaFree(d_comb_GX); cudaFree(d_comb_GY);
        return 1;
    }
    // No line count needed for text stream. We will estimate ETA by bytes.
    fseek(f, 0, SEEK_END);
    uint64_t total_bytes = (uint64_t)ftell(f);
    rewind(f);
    std::cout << "Dictionary : " << BRAIN_FILE << " (" << total_bytes << " bytes)\n";

    // ── Parse rules ──────────────────────────────────────────────────────────
    std::vector<GpuRule> h_rules;
    std::vector<std::string> h_rule_strings;
    if (!rules_file.empty()) {
        if (!RuleParser::parse_rule_file(rules_file, h_rules, &h_rule_strings)) return 1;
    }
    int num_rules = (int)h_rules.size();
    GpuRule* d_rules = nullptr;
    if (num_rules > 0) {
        cudaError_t err = cudaMalloc(&d_rules, num_rules * sizeof(GpuRule));
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: cudaMalloc failed for rules: " << cudaGetErrorString(err) << "\n";
            fclose(f);
            cudaFree(d_comb_GX); cudaFree(d_comb_GY);
            return 1;
        }
        err = cudaMemcpy(d_rules, h_rules.data(), num_rules * sizeof(GpuRule), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "CUDA Error: cudaMemcpy failed for rules: " << cudaGetErrorString(err) << "\n";
            fclose(f);
            cudaFree(d_rules);
            cudaFree(d_comb_GX); cudaFree(d_comb_GY);
            return 1;
        }
    }

    // ── GPU setup ─────────────────────────────────────────────────────────────
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    // kernel_bw_ecc has 0 smem → 8 blocks/SM at 256t → full 2048t/SM occupancy
    const int occupancy_wave = std::max(131072, (int)(prop.multiProcessorCount * 2048));
    
    size_t free_vram, total_vram;
    cudaMemGetInfo(&free_vram, &total_vram);
    size_t usable_vram = (free_vram > 512*1024*1024) ? (free_vram - 512*1024*1024) : 0;
    size_t max_items_from_vram = usable_vram / 66; // ~65 bytes per item + margin
    
    // Si --CP est actif, chaque candidat génère 4 variantes
    int cp_multiplier = use_cp ? 2 : 1;
    max_items_from_vram /= cp_multiplier;

    int max_allowed_items = std::max(occupancy_wave, 8388608);
    if ((size_t)max_allowed_items > max_items_from_vram) {
        max_allowed_items = (int)max_items_from_vram;
    }
    
    int effective_wave = max_allowed_items / std::max(1, num_rules);
    if (effective_wave < 1) effective_wave = 1;
    
    if (effective_wave < 256) {
        int target_items = 256 * std::max(1, num_rules);
        if ((size_t)target_items <= max_items_from_vram) {
            effective_wave = 256;
        } else {
            effective_wave = std::max(1, (int)(max_items_from_vram / std::max(1, num_rules)));
        }
    }
    const int max_items = effective_wave * std::max(1, num_rules) * cp_multiplier;
    const int max_blk_inv = (max_items + BW_THREADS_INV - 1) / BW_THREADS_INV;

    const size_t max_chunk_size = 16 * 1024 * 1024; // 16 MB chunk

    uint8_t*       d_chunk        = nullptr;
    uint32_t*      d_offsets      = nullptr;
    TargetData*    d_target       = nullptr;
    HydraResult*   d_result       = nullptr;
    BrainJacobian* d_jacobian     = nullptr;
    bw_u256*       d_local_except = nullptr;
    bw_u256*       d_block_prods  = nullptr;
    bw_u256*       d_block_inv    = nullptr;
    uint8_t*       d_z_zero       = nullptr;

    cudaMalloc(&d_chunk,        max_chunk_size);
    cudaMalloc(&d_offsets,      (size_t)effective_wave * sizeof(uint32_t));
    cudaMalloc(&d_target,       sizeof(TargetData));
    cudaMalloc(&d_result,       sizeof(HydraResult));
    cudaMalloc(&d_jacobian,     (size_t)max_items * sizeof(BrainJacobian));
    cudaMalloc(&d_local_except, (size_t)max_items * sizeof(bw_u256));
    cudaMalloc(&d_block_prods,  (size_t)max_blk_inv * sizeof(bw_u256));
    cudaMalloc(&d_block_inv,    (size_t)max_blk_inv * sizeof(bw_u256));
    cudaMalloc(&d_z_zero,       (size_t)max_items);
    cudaMemcpy(d_target, &target, sizeof(TargetData), cudaMemcpyHostToDevice);

    HydraResult h_res = {};
    cudaMemcpy(d_result, &h_res, sizeof(HydraResult), cudaMemcpyHostToDevice);

    std::cout << "======== HYDRA (BRAINWALLET MODE) ========================\n";
    std::cout << "GPU    : " << prop.name << " (" << prop.multiProcessorCount << " SM)\n";
    std::cout << "wCOMB  : w=" << w << "  cols=" << comb_cols << "  stride=" << comb_stride << "\n";
    std::cout << "Wave   : " << max_items << " (eff: " << effective_wave << " × " << std::max(1, num_rules) << " rules" << (use_cp ? " × 2 CP" : "") << ")\n";
    std::cout << "Target : " << addr_str << "\n";

    // Resume state
    ResumeState rs = make_resume_state("brainwallet", "", addr_str);

    // Host buffers: double-buffered and pinned so parsing overlaps async H2D copies.
    uint8_t*  h_chunk[2]   = {nullptr, nullptr};
    uint32_t* h_offsets[2] = {nullptr, nullptr};
    const size_t h_chunk_bytes   = max_chunk_size;
    const size_t h_offsets_bytes = (size_t)effective_wave * sizeof(uint32_t);
    for (int b = 0; b < 2; ++b) {
        cudaError_t e1 = cudaMallocHost((void**)&h_chunk[b], h_chunk_bytes);
        cudaError_t e2 = cudaMallocHost((void**)&h_offsets[b], h_offsets_bytes);
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
            std::cerr << "CUDA Error: cudaMallocHost failed for brainwallet buffers\n";
            for (int i = 0; i < 2; ++i) {
                if (h_chunk[i]) cudaFreeHost(h_chunk[i]);
                if (h_offsets[i]) cudaFreeHost(h_offsets[i]);
            }
            if (d_rules) cudaFree(d_rules);
            fclose(f);
            cudaFree(d_comb_GX); cudaFree(d_comb_GY);
            cudaFree(d_chunk); cudaFree(d_offsets);
            cudaFree(d_jacobian); cudaFree(d_local_except);
            cudaFree(d_block_prods); cudaFree(d_block_inv); cudaFree(d_z_zero);
            cudaFree(d_target); cudaFree(d_result);
            return 1;
        }
    }
    int      h_count[2] = {0, 0};
    uint64_t h_start_pos[2] = {0, 0};
    uint64_t h_end_pos[2] = {0, 0};
    size_t   h_chunk_size[2] = {0, 0};
    bool     h_eof[2] = {false, false};
    double   h_read_ms[2] = {0.0, 0.0};

    TextBatchReader reader(f);

    auto fill_brain_batch = [&](int slot) {
        auto read_begin = std::chrono::high_resolution_clock::now();
        int count = reader.fill_batch(h_chunk[slot], h_offsets[slot], effective_wave, max_chunk_size,
                                      h_start_pos[slot], h_end_pos[slot], h_eof[slot], h_chunk_size[slot]);
        auto read_end = std::chrono::high_resolution_clock::now();
        h_count[slot] = count;
        h_read_ms[slot] = std::chrono::duration<double, std::milli>(read_end - read_begin).count();
    };

    // ── Main loop ─────────────────────────────────────────────────────────────
    auto t0         = std::chrono::high_resolution_clock::now();
    auto t_last     = t0;
    auto t_resume   = t0;
    uint64_t total_tested    = 0;
    uint64_t tested_this_run = 0;
    uint64_t since_last      = 0;
    int      found           = 0;
    bool     eof             = false;
    int      last_batch_slot = 0;
    int      last_batch_count = 0;
    uint64_t prof_candidates = 0;
    uint64_t prof_batches    = 0;
    double prof_read_ms = 0.0, prof_h2d_ms = 0.0, prof_k1_ms = 0.0;
    double prof_k2_ms = 0.0, prof_k3_ms = 0.0, prof_k4_ms = 0.0, prof_d2h_ms = 0.0;
    cudaStream_t bw_stream = nullptr;
    cudaStreamCreateWithFlags(&bw_stream, cudaStreamNonBlocking);
    cudaEvent_t ev_h2d0, ev_h2d1, ev_k10, ev_k11, ev_k20, ev_k21, ev_k30, ev_k31, ev_k40, ev_k41;
    cudaEventCreate(&ev_h2d0); cudaEventCreate(&ev_h2d1);
    cudaEventCreate(&ev_k10);  cudaEventCreate(&ev_k11);
    cudaEventCreate(&ev_k20);  cudaEventCreate(&ev_k21);
    cudaEventCreate(&ev_k30);  cudaEventCreate(&ev_k31);
    cudaEventCreate(&ev_k40);  cudaEventCreate(&ev_k41);

    if (resume && resume->active) {
        total_tested        = resume->tested;
        rs.tested           = total_tested;
        rs.dict_byte_offset = resume->dict_byte_offset;
        if (!fseek(f, resume->dict_byte_offset, SEEK_SET)) {
            std::cerr << "[Resume] Cannot seek, restarting.\n";
            rewind(f); total_tested = 0; rs.tested = 0; rs.dict_byte_offset = 0;
        } else {
            std::cout << "[Resume] BRAINWALLET at byte " << resume->dict_byte_offset
                      << " | tested " << total_tested << " candidates\n";
        }
    }
    write_resume_snapshot(rs);

    fill_brain_batch(0);
    int cur_slot = 0;
    int next_slot = 1;
    int batch_count = h_count[cur_slot];
    while (!g_sigint && !found && batch_count > 0) {
        last_batch_slot = cur_slot;
        last_batch_count = batch_count;
        const uint64_t batch_end_pos = h_end_pos[cur_slot];
        prof_read_ms += h_read_ms[cur_slot];

        std::thread reader_thread;
        bool reader_started = false;
        if (!h_eof[cur_slot]) {
            reader_started = true;
            reader_thread = std::thread(fill_brain_batch, next_slot);
        } else {
            eof = true;
        }

        cudaEventRecord(ev_h2d0, bw_stream);
        cudaMemcpyAsync(d_chunk, h_chunk[cur_slot],
                        h_chunk_size[cur_slot], cudaMemcpyHostToDevice, bw_stream);
        cudaMemcpyAsync(d_offsets, h_offsets[cur_slot], (size_t)batch_count * sizeof(uint32_t),
                        cudaMemcpyHostToDevice, bw_stream);
        cudaEventRecord(ev_h2d1, bw_stream);

        int base_batch_count = batch_count * std::max(1, num_rules);
        int total_batch_count = base_batch_count * cp_multiplier;
        const int blk_ecc  = (base_batch_count + BW_THREADS_ECC - 1) / BW_THREADS_ECC;
        const int blk_inv  = (total_batch_count + BW_THREADS_INV - 1) / BW_THREADS_INV;
        const int blk_inv2 = (blk_inv + 127) / 128;
        const size_t smem2 = 2 * BW_THREADS_INV * sizeof(bw_u256);
        const size_t smem3 = 132 * sizeof(bw_u256);

        cudaEventRecord(ev_k10, bw_stream);
        if (use_cp) {
            switch (w) {
                case 19: kernel_bw_ecc_t<19, true><<<blk_ecc, BW_THREADS_ECC, 0, bw_stream>>>(d_chunk, d_offsets, base_batch_count, batch_count, d_comb_GX, d_comb_GY, d_jacobian, d_rules, num_rules); break;
                case 22: kernel_bw_ecc_t<22, true><<<blk_ecc, BW_THREADS_ECC, 0, bw_stream>>>(d_chunk, d_offsets, base_batch_count, batch_count, d_comb_GX, d_comb_GY, d_jacobian, d_rules, num_rules); break;
                case 24: kernel_bw_ecc_t<24, true><<<blk_ecc, BW_THREADS_ECC, 0, bw_stream>>>(d_chunk, d_offsets, base_batch_count, batch_count, d_comb_GX, d_comb_GY, d_jacobian, d_rules, num_rules); break;
                case 26: kernel_bw_ecc_t<26, true><<<blk_ecc, BW_THREADS_ECC, 0, bw_stream>>>(d_chunk, d_offsets, base_batch_count, batch_count, d_comb_GX, d_comb_GY, d_jacobian, d_rules, num_rules); break;
                default: kernel_bw_ecc<true><<<blk_ecc, BW_THREADS_ECC, 0, bw_stream>>>(d_chunk, d_offsets, base_batch_count, batch_count, d_comb_GX, d_comb_GY, w, comb_cols, comb_stride, d_jacobian, d_rules, num_rules); break;
            }
        } else {
            switch (w) {
                case 19: kernel_bw_ecc_t<19, false><<<blk_ecc, BW_THREADS_ECC, 0, bw_stream>>>(d_chunk, d_offsets, base_batch_count, batch_count, d_comb_GX, d_comb_GY, d_jacobian, d_rules, num_rules); break;
                case 22: kernel_bw_ecc_t<22, false><<<blk_ecc, BW_THREADS_ECC, 0, bw_stream>>>(d_chunk, d_offsets, base_batch_count, batch_count, d_comb_GX, d_comb_GY, d_jacobian, d_rules, num_rules); break;
                case 24: kernel_bw_ecc_t<24, false><<<blk_ecc, BW_THREADS_ECC, 0, bw_stream>>>(d_chunk, d_offsets, base_batch_count, batch_count, d_comb_GX, d_comb_GY, d_jacobian, d_rules, num_rules); break;
                case 26: kernel_bw_ecc_t<26, false><<<blk_ecc, BW_THREADS_ECC, 0, bw_stream>>>(d_chunk, d_offsets, base_batch_count, batch_count, d_comb_GX, d_comb_GY, d_jacobian, d_rules, num_rules); break;
                default: kernel_bw_ecc<false><<<blk_ecc, BW_THREADS_ECC, 0, bw_stream>>>(d_chunk, d_offsets, base_batch_count, batch_count, d_comb_GX, d_comb_GY, w, comb_cols, comb_stride, d_jacobian, d_rules, num_rules); break;
            }
        }
        cudaError_t err_last = cudaGetLastError();
        cudaError_t err_sync = cudaStreamSynchronize(bw_stream);
        if (err_last != cudaSuccess || err_sync != cudaSuccess) {
            cudaError_t err = (err_last != cudaSuccess) ? err_last : err_sync;
            std::cerr << "CUDA Error (bw_ecc): " << cudaGetErrorString(err) << "\n";
            if (reader_started) reader_thread.join();
            break;
        }
        cudaEventRecord(ev_k11, bw_stream);
        cudaEventRecord(ev_k20, bw_stream);
        kernel_bw_local_prod<<<blk_inv, BW_THREADS_INV, smem2, bw_stream>>>(
            d_jacobian, total_batch_count, d_local_except, d_block_prods, d_z_zero);
        cudaEventRecord(ev_k21, bw_stream);
        cudaEventRecord(ev_k30, bw_stream);
        kernel_bw_invert_blocks<<<blk_inv2, 128, smem3, bw_stream>>>(
            d_block_prods, d_block_inv, blk_inv);
        cudaEventRecord(ev_k31, bw_stream);
        cudaEventRecord(ev_k40, bw_stream);
        kernel_bw_finalize<<<blk_inv, BW_THREADS_INV, 0, bw_stream>>>(
            d_jacobian, d_local_except, d_block_inv, d_z_zero,
            total_batch_count, d_target, d_result);
        cudaEventRecord(ev_k41, bw_stream);

        if (cudaGetLastError()!=cudaSuccess || cudaStreamSynchronize(bw_stream)!=cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(cudaGetLastError()) << "\n";
            if (reader_started) reader_thread.join();
            break;
        }
        float stage_ms = 0.0f;
        cudaEventElapsedTime(&stage_ms, ev_h2d0, ev_h2d1); prof_h2d_ms += stage_ms;
        cudaEventElapsedTime(&stage_ms, ev_k10,  ev_k11);  prof_k1_ms  += stage_ms;
        cudaEventElapsedTime(&stage_ms, ev_k20,  ev_k21);  prof_k2_ms  += stage_ms;
        cudaEventElapsedTime(&stage_ms, ev_k30,  ev_k31);  prof_k3_ms  += stage_ms;
        cudaEventElapsedTime(&stage_ms, ev_k40,  ev_k41);  prof_k4_ms  += stage_ms;
        prof_candidates += (uint64_t)total_batch_count;
        prof_batches++;

        auto d2h_begin = std::chrono::high_resolution_clock::now();
        cudaMemcpyAsync(&found, &d_result->found, sizeof(int), cudaMemcpyDeviceToHost, bw_stream);
        cudaStreamSynchronize(bw_stream);
        auto d2h_end = std::chrono::high_resolution_clock::now();
        prof_d2h_ms += std::chrono::duration<double, std::milli>(d2h_end - d2h_begin).count();

        // Bloom hit → CPU verification
        if (found && (target.type==TargetType::BLOOM||target.type==TargetType::BLOOM_BTC
                      ||target.type==TargetType::BLOOM_ETH)) {
            cudaMemcpyAsync(&h_res, d_result, sizeof(HydraResult), cudaMemcpyDeviceToHost, bw_stream);
            cudaStreamSynchronize(bw_stream);
            uint64_t total_local_idx = h_res.index;
            int rules_count = std::max(1, num_rules);
            int base_N = batch_count * rules_count;
            int cp_variant = use_cp ? (total_local_idx / base_N) : 0;
            uint64_t base_idx = use_cp ? (total_local_idx % base_N) : total_local_idx;
            uint64_t word_idx = num_rules > 0 ? (base_idx % batch_count) : base_idx;
            int rule_idx = num_rules > 0 ? (base_idx / batch_count) : 0;

            std::string hit_pass;
            std::string hit_rule;
            if (word_idx < (uint64_t)batch_count) {
                uint32_t offset = h_offsets[cur_slot][word_idx];
                uint32_t pass_len = h_chunk[cur_slot][offset];
                const uint8_t* p = h_chunk[cur_slot] + offset + 1;
                
                uint8_t local_pass[MAX_BRAIN_LEN];
                uint32_t out_len = pass_len;
                if (num_rules > 0) {
                    bool valid_rule = apply_rule(p, pass_len, local_pass, out_len, &h_rules[rule_idx], MAX_BRAIN_LEN);
                    if (!valid_rule) out_len = 0;
                    hit_rule = h_rule_strings[rule_idx];
                } else {
                    for (uint32_t i = 0; i < pass_len; ++i) local_pass[i] = p[i];
                }
                hit_pass = std::string((const char*)local_pass, out_len);
                if (use_cp) {
                    if (cp_variant == 1) hit_pass += " (INVERSE)";
                }
            }
            std::cout << "\n!!! BLOOM HIT !!! passphrase=\"" << hit_pass << "\"\n";
            if (!hit_rule.empty()) std::cout << "  Rule        : " << hit_rule << "\n";
            std::cout << "  [CPU] Verifying...\n";

            uint8_t privkey[32];
            cpu_brain_privkey(hit_pass.c_str(), (uint32_t)hit_pass.size(), privkey);
            std::string al, as_, ae;
            key_to_addresses(privkey, al, as_, ae);
            bool victory = false;
            if (target.type != TargetType::BLOOM_ETH)
                victory = check_balances_and_notify(privkey, al, as_, ae);
            if (!victory && target.type != TargetType::BLOOM_BTC) {
                // Derive ETH address from same key
                victory = check_balances_and_notify(privkey, al, as_, ae);
            }
            if (!victory) {
                int zero=0;
                cudaMemcpyAsync(&d_result->found, &zero, sizeof(int), cudaMemcpyHostToDevice, bw_stream);
                cudaStreamSynchronize(bw_stream);
                found = 0;
            }
        }

        int total_mutations = batch_count * std::max(1, num_rules) * cp_multiplier;
        total_tested     += total_mutations;
        tested_this_run  += total_mutations;
        rs.tested         = total_tested;
        rs.dict_byte_offset = batch_end_pos;
        since_last       += total_mutations;

        auto now = std::chrono::high_resolution_clock::now();
        const bool force_resume = found || g_sigint || eof;
        if (should_write_resume_snapshot(t_resume, now, force_resume)) {
            write_resume_snapshot(rs); t_resume = now;
        }
        double dt = std::chrono::duration<double>(now - t_last).count();
        if (dt >= 1.0) {
            double spd     = since_last / dt / 1e6;
            double elapsed = std::chrono::duration<double>(now - t0).count();
            double prog    = (total_bytes>0) ? 100.0 * (double)rs.dict_byte_offset / (double)total_bytes : 0.0;
            double eta     = (elapsed > 0 && rs.dict_byte_offset > 0 && total_bytes > 0)
                ? (double)(total_bytes - rs.dict_byte_offset) / ((double)rs.dict_byte_offset / elapsed) : 0;
            int eh=(int)(eta/3600), em=(int)((eta-eh*3600)/60), es=(int)((long long)eta%60);
            std::cout << "\r[" << std::fixed << std::setprecision(1) << prog << "%] "
                      << std::setprecision(2) << (spd / cp_multiplier) << " MKey/s"
                      << " | ETA " << std::setfill('0')
                      << std::setw(2)<<eh<<":"<<std::setw(2)<<em<<":"<<std::setw(2)<<es
                      << std::setfill(' ') << "    " << std::flush;
            t_last = now; since_last = 0;
        }

        if (reader_started) {
            reader_thread.join();
            if (!found && !g_sigint) {
                cur_slot = next_slot;
                next_slot = 1 - cur_slot;
                batch_count = h_count[cur_slot];
                if (batch_count == 0) eof = true;
            }
        } else {
            eof = true;
            batch_count = 0;
        }
    }
    fclose(f);

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(t_end - t0).count();
    double avg_spd = (total_elapsed>0) ? (double)total_tested/total_elapsed/1e6 : 0;
    std::cout << "\nTime    : " << std::fixed << std::setprecision(2) << total_elapsed << " s"
              << " | " << std::setprecision(2) << avg_spd << " MKey/s avg"
              << " | Tested  : " << total_tested << "\n";
    if (prof_batches > 0 && prof_candidates > 0) {
        auto stage_rate = [&](double ms) -> double {
            return (ms > 0.0) ? ((double)prof_candidates / ms / 1000.0) : 0.0;
        };
        const double prof_gpu_ms = prof_h2d_ms + prof_k1_ms + prof_k2_ms + prof_k3_ms + prof_k4_ms + prof_d2h_ms;
        std::cout << "[Profile] batches=" << prof_batches
                  << " candidates=" << prof_candidates
                  << " gpu+copy=" << std::fixed << std::setprecision(2) << prof_gpu_ms << " ms\n";
        std::cout << "  read/fill : " << std::setw(9) << prof_read_ms << " ms"
                  << " | " << std::setw(7) << stage_rate(prof_read_ms) << " MKey/s\n";
        std::cout << "  H2D copy  : " << std::setw(9) << prof_h2d_ms << " ms"
                  << " | " << std::setw(7) << stage_rate(prof_h2d_ms) << " MKey/s\n";
        std::cout << "  K1 SHA+ECC: " << std::setw(9) << prof_k1_ms << " ms"
                  << " | " << std::setw(7) << stage_rate(prof_k1_ms) << " MKey/s\n";
        std::cout << "  K2 local  : " << std::setw(9) << prof_k2_ms << " ms"
                  << " | " << std::setw(7) << stage_rate(prof_k2_ms) << " MKey/s\n";
        std::cout << "  K3 invert : " << std::setw(9) << prof_k3_ms << " ms"
                  << " | " << std::setw(7) << stage_rate(prof_k3_ms) << " MKey/s\n";
        std::cout << "  K4 hash   : " << std::setw(9) << prof_k4_ms << " ms"
                  << " | " << std::setw(7) << stage_rate(prof_k4_ms) << " MKey/s\n";
        std::cout << "  D2H found : " << std::setw(9) << prof_d2h_ms << " ms"
                  << " | " << std::setw(7) << stage_rate(prof_d2h_ms) << " MKey/s\n";
    }

    if (found) {
        clear_resume_snapshot();
        cudaMemcpyAsync(&h_res, d_result, sizeof(HydraResult), cudaMemcpyDeviceToHost, bw_stream);
        cudaStreamSynchronize(bw_stream);
        const uint64_t total_local_idx = h_res.index;
        int rules_count = std::max(1, num_rules);
        int base_N = last_batch_count * rules_count;
        int cp_variant = total_local_idx / base_N;
        uint64_t base_idx = total_local_idx % base_N;
        uint64_t word_idx = num_rules > 0 ? (base_idx % last_batch_count) : base_idx;
        int rule_idx = num_rules > 0 ? (base_idx / last_batch_count) : 0;

        std::string found_pass;
        std::string hit_rule;
        if (word_idx < (uint64_t)last_batch_count) {
            uint32_t offset = h_offsets[last_batch_slot][word_idx];
            uint32_t pass_len = h_chunk[last_batch_slot][offset];
            const uint8_t* p = h_chunk[last_batch_slot] + offset + 1;
            
            uint8_t local_pass[MAX_BRAIN_LEN];
            uint32_t out_len = pass_len;
            if (num_rules > 0) {
                bool valid_rule = apply_rule(p, pass_len, local_pass, out_len, &h_rules[rule_idx], MAX_BRAIN_LEN);
                if (!valid_rule) out_len = 0;
                hit_rule = h_rule_strings[rule_idx];
            } else {
                for (uint32_t i = 0; i < pass_len; ++i) local_pass[i] = p[i];
            }
            found_pass = std::string((const char*)local_pass, out_len);
            if (cp_variant == 1) found_pass += " (INVERSE)";
        }
        std::cout << "\n======== VICTORY ! BRAINWALLET FOUND ==================\n";
        std::cout << "Passphrase : \"" << found_pass << "\"\n";
        if (!hit_rule.empty()) std::cout << "Rule       : " << hit_rule << "\n";

        uint8_t privkey[32];
        cpu_brain_privkey(found_pass.c_str(), (uint32_t)found_pass.size(), privkey);
        std::string al, as_, ae;
        key_to_addresses(privkey, al, as_, ae);

        std::cout << "BTC legacy : " << al << "\n";
        std::cout << "BTC segwit : " << as_ << "\n";
        std::cout << "ETH        : " << ae << "\n";
        // privkey hex
        std::cout << "PrivKey    : ";
        for (int i=0;i<32;i++) std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)privkey[i];
        std::cout << std::dec << "\n";
        std::cout << "=======================================================\n";

        std::string key_info = "*Passphrase:*\n`" + found_pass + "`";
        std::string addr_info = is_bloom_arg(addr_str) ? ("*BTC:* `"+al+"`\n*ETH:* `"+ae+"`")
                                                       : ("*Address:* `"+addr_str+"`");
        notify_victory("BRAINWALLET FOUND \xF0\x9F\x94\x91", key_info, addr_info);
    } else if (!g_sigint) {
        clear_resume_snapshot();
        std::cout << "Passphrase not found in " << BRAIN_FILE << "\n";
    } else {
        write_resume_snapshot(rs);
        print_resume_hint();
    }
    print_search_summary(found != 0);

    cudaFree(d_comb_GX); cudaFree(d_comb_GY);
    cudaFree(d_chunk); cudaFree(d_offsets);
    cudaFree(d_jacobian); cudaFree(d_local_except);
    cudaFree(d_block_prods); cudaFree(d_block_inv); cudaFree(d_z_zero);
    cudaFree(d_rules);
    if (target.d_bloom_filter &&
        (target.type==TargetType::BLOOM||target.type==TargetType::BLOOM_BTC||target.type==TargetType::BLOOM_ETH))
        cudaFree((void*)target.d_bloom_filter);
    cudaFree(d_target); cudaFree(d_result);
    cudaFreeHost(h_chunk[0]); cudaFreeHost(h_chunk[1]);
    cudaFreeHost(h_offsets[0]); cudaFreeHost(h_offsets[1]);
    cudaEventDestroy(ev_h2d0); cudaEventDestroy(ev_h2d1);
    cudaEventDestroy(ev_k10);  cudaEventDestroy(ev_k11);
    cudaEventDestroy(ev_k20);  cudaEventDestroy(ev_k21);
    cudaEventDestroy(ev_k30);  cudaEventDestroy(ev_k31);
    cudaEventDestroy(ev_k40);  cudaEventDestroy(ev_k41);
    cudaStreamDestroy(bw_stream);
    return found ? 0 : 1;
}

int main(int argc, char* argv[]) {
    load_tokens();

    if (argc < 2) {
        std::cerr << "Usage:\n";
        std::cerr << "  Hex mode        : ./Hydra <64hex_mask_#> <address|pubkey|bloom>\n";
        std::cerr << "  Seed mode       : ./Hydra \"<phrase with # for unknowns>\" <address|bloom>\n";
        std::cerr << "  Electrum V2     : ./Hydra \"<phrase with # for unknowns>\" <address|bloom> --electrumV2\n";
        std::cerr << "                    same BIP39 wordlist, Electrum checksum + m/0/0 derivation\n";
        std::cerr << "                    auto-detects standard ('01') or segwit ('100') from address\n";
        std::cerr << "  Electrum V1     : ./Hydra \"<old Electrum phrase with #>\" <address|bloom> --electrumV1 [--path m/0/0|--gap N]\n";
        std::cerr << "                    legacy 1626-word mode, default m/0/0..19 + m/1/0..19\n";
        std::cerr << "  Passphrase mode : ./Hydra \"<12 full words>\" <address|bloom>\n";
        std::cerr << "                    (reads dictionary.txt)\n";
        std::cerr << "  Brainwallet     : ./Hydra brainwallet <address|bloom>\n";
        std::cerr << "                    SHA256(line) → privkey for each line in brainwallet.txt\n";
        std::cerr << "  WIF mode        : ./Hydra <wif_mask_#> <address|pubkey|bloom>\n";
        std::cerr << "                    known pubkey targets auto-select BSGS for larger masks\n";
        std::cerr << "  Resume          : ./Hydra resume\n";
        std::cerr << "  Bloom filter    : replace <address> with bloom/bloombtc/bloometh\n";
        std::cerr << "                    bloom    = search BTC + ETH in bloom.bin\n";
        std::cerr << "                    bloombtc = BTC only (faster)\n";
        std::cerr << "                    bloometh = ETH only (faster)\n";
        std::cerr << "  Shell note      : quote masks that start with #, e.g. './Hydra \"##...\" <target>'\n";
        std::cerr << "  BSGS split      : experimental opt-ins: --baby=7.5-vram, --baby=8, --baby=5-vram, --baby=5.5\n";
        return 1;
    }

    std::string rules_file = "";
    if (std::ifstream("resources/rule.txt").good()) {
        rules_file = "resources/rule.txt";
    }

    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--CP") {
            // --CP is consumed silently (CP trick is always enabled)
        } else {
            args.push_back(argv[i]);
        }
    }
    
    if (args.empty()) {
        std::cerr << "Usage:\n";
        return 1;
    }
    
    std::string arg1 = args[0];

    if (args.size() == 1 && arg1 == "eccdiag") {
        return run_eccdiag_mode();
    }

    if (args.size() == 1 && arg1 == "resume") {
        ResumeState rs;
        if (!load_resume_snapshot(rs)) {
            std::cerr << "Error: no valid resume snapshot in " << HYDRA_RESUME_FILE << "\n";
            return 1;
        }

        std::cout << "[Resume] Mode=" << rs.mode << "\n";
        if (rs.mode == "hex")          return run_hex_mode(rs.arg1, rs.arg2, &rs);
        if (rs.mode == "hex_bsgs")     return run_hex_bsgs_mode(rs.arg1, rs.arg2, &rs);
        if (rs.mode == "wif_bsgs")     return run_wif_bsgs_mode(rs.arg1, rs.arg2, &rs);
        if (rs.mode == "seed")         return run_seed_mode(rs.arg1, rs.arg2, &rs);
        if (rs.mode == "electrumv2")   return run_electrumv2_mode(rs.arg1, rs.arg2, &rs);
        if (rs.mode == "electrumv1")   return run_electrumv1_mode(rs.arg1, rs.arg2, &rs);
        if (rs.mode == "wif")          return run_wif_mode(rs.arg1, rs.arg2, &rs);
        if (rs.mode == "passphrase")   return run_passphrase_mode(rs.arg1, rs.arg2, rules_file, &rs);
        if (rs.mode == "brainwallet")  return run_brainwallet_mode(rs.arg2, rules_file, true, &rs);

        std::cerr << "Error: unsupported resume mode '" << rs.mode << "'\n";
        return 1;
    }

    if (args.size() >= 2 && looks_like_hex_mask(arg1)) {
        BsgsBabyOverride baby_override = BsgsBabyOverride::AUTO;
        std::vector<std::string> bsgs_opts;
        for (size_t i = 2; i < args.size(); ++i) bsgs_opts.emplace_back(args[i]);
        if (!parse_bsgs_baby_override(bsgs_opts, false, baby_override)) return 1;
        TargetData bsgs_target = {};
        const uint32_t unknowns = bsgs_count_hex_unknowns(arg1);
        if (unknowns >= HYDRA_HEX_BSGS_AUTO_MIN_UNKNOWN &&
            try_resolve_any_pubkey_target_for_scheduler(args[1].c_str(), bsgs_target)) {
            std::cout << "[Scheduler] HEX + known pubkey + >= "
                      << HYDRA_HEX_BSGS_AUTO_MIN_UNKNOWN
                      << " nibbles: using BSGS\n";
            return run_hex_bsgs_mode(arg1, args[1], nullptr, &bsgs_target, baby_override);
        }
        if (baby_override != BsgsBabyOverride::AUTO) {
            std::cerr << "Error: --baby requires HEX BSGS, so the target public key must be known/resolvable.\n";
            return 1;
        }
        return run_hex_mode(arg1, args[1]);
    } else if (args.size() == 2 && is_passphrase_mode(arg1)) {
        return run_passphrase_mode(arg1, args[1], rules_file);
    } else if (args.size() == 3 && looks_like_seed(arg1) && is_electrum_v2_flag(args[2].c_str())) {
        return run_electrumv2_mode(arg1, args[1]);
    } else if (args.size() >= 3 && is_electrum_v1_flag(args[2].c_str())) {
        std::vector<std::string> ev1_opts;
        for (size_t i = 3; i < args.size(); ++i) ev1_opts.emplace_back(args[i]);
        return run_electrumv1_mode(arg1, args[1], nullptr, ev1_opts);
    } else if (args.size() == 2 && looks_like_seed(arg1)) {
        return run_seed_mode(arg1, args[1]);
    } else if (args.size() >= 2 && looks_like_wif_mask(arg1)) {
        BsgsBabyOverride baby_override = BsgsBabyOverride::AUTO;
        std::vector<std::string> bsgs_opts;
        for (size_t i = 2; i < args.size(); ++i) bsgs_opts.emplace_back(args[i]);
        if (!parse_bsgs_baby_override(bsgs_opts, true, baby_override)) return 1;
        TargetData bsgs_target = {};
        const uint32_t unknowns = bsgs_count_wif_unknowns(arg1);
        if (unknowns >= HYDRA_WIF_BSGS_AUTO_MIN_UNKNOWN &&
            try_resolve_btc_pubkey_target_for_scheduler(args[1].c_str(), bsgs_target)) {
            std::cout << "[Scheduler] WIF + known pubkey + >= "
                      << HYDRA_WIF_BSGS_AUTO_MIN_UNKNOWN
                      << " unknowns: using BSGS\n";
            return run_wif_bsgs_mode(arg1, args[1], nullptr, &bsgs_target, baby_override);
        }
        if (baby_override != BsgsBabyOverride::AUTO) {
            std::cerr << "Error: --baby requires WIF BSGS, so the BTC public key must be known/resolvable.\n";
            return 1;
        }
        return run_wif_mode(arg1, args[1]);
    } else if (args.size() == 2 && arg1 == "brainwallet") {
        return run_brainwallet_mode(args[1], rules_file, true);
    } else {
        std::cerr << "Error: unrecognized argument or missing address.\n\n";
        std::cerr << "Usage:\n";
        std::cerr << "  Hex mode        : ./Hydra <64hex_mask_#> <address|pubkey|bloom>\n";
        std::cerr << "  Seed mode       : ./Hydra \"<phrase with # for unknowns>\" <address|bloom>\n";
        std::cerr << "  Electrum V2     : ./Hydra \"<phrase with # for unknowns>\" <address|bloom> --electrumV2\n";
        std::cerr << "                    same BIP39 wordlist, Electrum checksum + m/0/0 derivation\n";
        std::cerr << "                    auto-detects standard ('01') or segwit ('100') from address\n";
        std::cerr << "  Electrum V1     : ./Hydra \"<old Electrum phrase with #>\" <address|bloom> --electrumV1 [--path m/0/0|--gap N]\n";
        std::cerr << "                    legacy 1626-word mode, default m/0/0..19 + m/1/0..19\n";
        std::cerr << "  Passphrase mode : ./Hydra \"<12 full words>\" <address|bloom>\n";
        std::cerr << "                    (reads dictionary.txt)\n";
        std::cerr << "  Brainwallet     : ./Hydra brainwallet <address|bloom>\n";
        std::cerr << "                    SHA256(line) → privkey for each line in brainwallet.txt\n";
        std::cerr << "  WIF mode        : ./Hydra <wif_mask_#> <address|pubkey|bloom>\n";
        std::cerr << "                    known pubkey targets auto-select BSGS for larger masks\n";
        std::cerr << "  Resume          : ./Hydra resume\n";
        std::cerr << "  Bloom filter    : replace <address> with bloom/bloombtc/bloometh\n";
        std::cerr << "                    bloom    = search BTC + ETH in bloom.bin\n";
        std::cerr << "                    bloombtc = BTC only (faster)\n";
        std::cerr << "                    bloometh = ETH only (faster)\n";
        std::cerr << "  Shell note      : quote masks that start with #, e.g. './Hydra \"##...\" <target>'\n";
        std::cerr << "  BSGS split      : experimental opt-ins: --baby=7.5-vram, --baby=8, --baby=5-vram, --baby=5.5\n";
        return 1;
    }
}
