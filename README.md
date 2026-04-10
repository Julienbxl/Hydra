# 🐍 Hydra V4.0 — GPU Private Key, Seed & Passphrase Recovery Tool

> **CUDA-accelerated brute-force recovery for partial Bitcoin and Ethereum keys.**  
> Designed for users who have lost part of a private key, WIF, BIP39 seed phrase, or BIP39 passphrase and need to recover access to their own wallet.

---

## ⚠️ Legal Disclaimer

This tool is intended **exclusively** for recovering access to wallets you own or have legal authorization to access. The authors accept no liability for misuse.

---

## Compilation

Hydra now uses `CMake` as the primary build system on both Linux and Windows.  
The default release build targets three NVIDIA architectures in one binary:

- `sm_86` — RTX 30xx
- `sm_89` — RTX 40xx
- `sm_120` — RTX 50xx

### Linux / WSL

Requirements:

- CUDA Toolkit 13.1
- `cmake` 3.24+
- `g++`
- OpenSSL development files (`libssl-dev`)

```bash
git clone https://github.com/Julienbxl/hydra.git
cd hydra

export CUDA_HOME=/usr/local/cuda-13.1
export CUDAToolkit_ROOT=/usr/local/cuda-13.1
export CUDACXX=/usr/local/cuda-13.1/bin/nvcc
export PATH=/usr/local/cuda-13.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH

cmake --preset linux-release
cmake --build --preset linux-release
```

The executable is emitted at the project root:

```bash
./Hydra
```

### Windows

Requirements:

- CUDA Toolkit 13.1 for Windows
- Visual Studio Build Tools 2022 with `Desktop development with C++`
- CMake
- `vcpkg` with OpenSSL installed:

```bat
vcpkg install openssl:x64-windows
```

Open an `x64 Native Tools Command Prompt for VS 2022`, then:

```bat
cd C:\dev\Hydra
cmake --preset windows-release
cmake --build --preset windows-release
```

The executable is emitted at the project root:

```bat
Hydra.exe
```

### Notes

- The legacy `Makefile` is no longer the recommended path for portable builds.
- If you want to override GPU targets, edit [`CMakeLists.txt`](./CMakeLists.txt) and adjust the `HYDRA_TARGET_SM*` options.
- The presets assume:
  - Linux CUDA at `/usr/local/cuda-13.1`
  - Windows CUDA at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`
  - Windows `vcpkg` at `C:\vcpkg`

---

## Configuration — `token.txt`

Credentials are read from a `token.txt` file placed next to the binary — **never hardcoded in source**. This keeps your keys out of version control and makes updates painless.

Create `token.txt` with one value per line:

```
YOUR_ETHERSCAN_API_KEY
YOUR_TELEGRAM_BOT_TOKEN
YOUR_TELEGRAM_CHAT_ID
```

- **Line 1 — Etherscan API key** — free tier at [etherscan.io/apis](https://etherscan.io/apis). Required for ETH balance verification on bloom hits.
- **Line 2 — Telegram bot token** — create a bot via [@BotFather](https://t.me/BotFather).
- **Line 3 — Telegram chat ID** — retrieve from `https://api.telegram.org/bot<TOKEN>/getUpdates`.

All three lines are optional. If `token.txt` is missing or a line is blank, the corresponding feature is silently disabled — ETH bloom verification falls back to hash-only mode, and Telegram notifications are skipped (hits are written to `errors.json` instead).

---

## Usage

```bash
./Hydra <key_or_phrase> <target>
```

The `<target>` can be:

- A **BTC legacy address** — `1ABC...`
- A **BTC SegWit address** — `bc1q...`
- An **ETH address** — `0x1234...abcd`
- A **bloom keyword** — `bloom`, `bloombtc`, or `bloometh`

---

## Mode 1 — Hex Private Key

Recover a 256-bit private key where some **nibbles** (hex characters) are unknown. Use `#` as the wildcard for each unknown nibble.

```bash
# 4 unknown nibbles = 16 bits — BTC legacy
./Hydra 7cb5da6f7757##14a59#f40dc45739eda5e532804f24af675e3339f1fe9c4 1AddressBTC

# BTC SegWit
./Hydra 7cb5da6f7757##14a59#f40dc45739eda5e532804f24af675e3339f1fe9c4 bc1qYourAddress

# ETH
./Hydra 7cb5da6f7757##14a59#f40dc45739eda5e532804f24af675e3339f1fe9c4 0x1234...abcd

# Bloom — BTC only
./Hydra 7cb5da6f7757##14a59#f40dc45739eda5e532804f24af675e3339f1fe9c4 bloombtc

# Bloom — ETH only
./Hydra 7cb5da6f7757##14a59#f40dc45739eda5e532804f24af675e3339f1fe9c4 bloometh

# Bloom — BTC + ETH
./Hydra 7cb5da6f7757##14a59#f40dc45739eda5e532804f24af675e3339f1fe9c4 bloom
```

Each `#` represents **4 unknown bits**.

### Pubkey bypass (automatic)

When the target address has at least one outgoing transaction, its public key is already revealed on-chain. Hydra automatically fetches it (BTC via mempool.space, ETH via Etherscan + ecrecover) and bypasses the hash step entirely — comparing the ECC point directly. This roughly **doubles throughput** for those addresses with no change in correctness.

**How it works:** the fixed part of the key is precomputed as `P_base = k_fixed × G` on CPU (OpenSSL). An affine dictionary of `2^LOW_BITS` precomputed increments covers the low-order bits on GPU at zero ECC cost; the high-order bits are enumerated via Gray code (one point addition per step).

---

## Mode 2 — BIP39 Seed Phrase

Recover a 12 or 24-word BIP39 phrase where some words are unknown. Use `#` as a placeholder for each missing word.

```bash
# 2 unknown words out of 12
./Hydra "word1 word2 # word4 # word6 word7 word8 word9 word10 word11 word12" 1AddressBTC

# BTC SegWit
./Hydra "word1 word2 # word4 ..." bc1qYourAddress

# ETH — derives via BIP44 m/44'/60'/0'/0/0
./Hydra "word1 word2 # word4 ..." 0x1234...abcd

# Bloom
./Hydra "word1 word2 # word4 ..." bloom
```

- If the **last word** is `#`, Hydra enforces the correct BIP39 checksum automatically — no wasted candidates.
- BTC path: `m/44'/0'/0'/0/0`. ETH path: `m/44'/60'/0'/0/0`.
- The GPU pipeline is 3-stage: **K1** filters on BIP39 checksum (eliminates 15/16 candidates instantly), **K2a** runs PBKDF2-HMAC-SHA512, **K2b/c** handle BIP32 derivation + ECC + address comparison.

---

## Mode 3 — BIP39 Passphrase (25th word)

All words are known but the passphrase is unknown. Hydra brute-forces it from a dictionary file.

```bash
./Hydra "word1 word2 ... word12" 1AddressBTC
./Hydra "word1 word2 ... word12" 0x1234...abcd
./Hydra "word1 word2 ... word12" bloom
```

Place candidates in `dictionary.txt`, one per line. On a bloom hit, Hydra derives the full BIP44 key CPU-side and verifies the live balance before confirming.

---

## Mode 4 — WIF (Wallet Import Format)

Recover a compressed WIF private key (52 characters, starting with `K` or `L`) with unknown characters. Use `#` for each unknown Base58 character.

```bash
# 3 unknown WIF characters
./Hydra KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qY#rFej#um7Wt#CRUx 1AddressBTC

# BTC SegWit
./Hydra KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qY#rFej#um7Wt#CRUx bc1qYourAddress
```

A SHA256×2 checksum pre-filter eliminates the vast majority of candidates before the ECC step.

---

## Bloom Filter Mode

Instead of a single address, Hydra can test every candidate against a bloom filter loaded in VRAM — scanning millions of addresses simultaneously at no per-address overhead.

| Keyword | Behavior |
|---|---|
| `bloombtc` | BTC legacy + SegWit only |
| `bloometh` | ETH only |
| `bloom` | BTC + ETH |

### Building a bloom filter

```bash
python3 create_bloom.py addresses.txt bloom.bin
python3 create_bloom.py btc.txt eth.txt bloom.bin   # multiple files
```

Accepts BTC legacy, BTC SegWit, and ETH addresses mixed in the same file.

```bash
pip install mmh3 base58 bech32 bitarray tqdm
```

### Verifying a filter

```bash
python3 check_bloom.py <address>
```

### VRAM tuning

The default filter targets 110 million addresses and produces a 2 GB file (`2^34` bits). Reduce `TARGET_SIZE_GB` in `create_bloom.py` for lower VRAM:

| `TARGET_SIZE_GB` | Filter bits | FP / Gkey (110M addrs) |
|---|---|---|
| 0.5 GB | `2^32` | ~27 |
| 1 GB | `2^33` | ~0.002 |
| **2 GB** *(default)* | **`2^34`** | **< 0.0001** |

> `TARGET_SIZE_GB` must be a power of two (0.5, 1, 2, 4…).

---

## Resume / Checkpoint

Hydra saves its progress automatically every 5 seconds, allowing you to safely interrupt and resume long searches without losing work.

```bash
# Interrupt at any time with Ctrl+C — progress is saved
./Hydra <mask> <target>
^C
[Resume] Checkpoint saved — resume with: ./Hydra resume

# Resume exactly where you left off
./Hydra resume
```

The checkpoint file (`hydra_resume.bin`) stores the mode, mask, target, and current offset. It is written atomically (temp file + rename) so a crash or power loss cannot corrupt it. All four modes support resume.

---

## Performance Tuning — `LOW_BITS`

Hydra splits the search space into two parts. High bits are enumerated via Gray code (one ECC point addition per step). Low bits are handled by a precomputed affine dictionary in constant memory at zero ECC cost.

```c
// HydraCommon.h
#define LOW_BITS 9   // 512-entry dictionary, ~24 KB constant memory
```

| `LOW_BITS` | Dictionary | Constant memory |
|---|---|---|
| `7` | 128 entries | ~6 KB |
| `8` | 256 entries | ~12 KB |
| `9` *(default)* | 512 entries | ~24 KB |
| `10` | 1024 entries | ~48 KB |

Higher `LOW_BITS` = more work per Gray step = higher throughput. If performance drops on older GPUs, try stepping down by 1. Recompile after any change.

---

## Performance Benchmarks

*Measured on RTX 5060 (30 SM, Blackwell sm_120)*

| Mode | Throughput |
|---|---|
| Hex — BTC legacy / SegWit / Bloom BTC | ~1,300 MK/s |
| Hex — ETH / Bloom ETH | ~800 MK/s |
| Hex — Bloom BTC+ETH | ~550 MK/s |
| Hex — BTC or ETH with known pubkey | ~2,500 MK/s |
| Seed | ~1,700,000 seeds/s |
| Passphrase | ~100,000 pass/s |
| WIF | ~2,800 MK/s |

The pubkey bypass activates automatically when the target address has at least one outgoing transaction — no flags needed.

---

## Testing

Self-contained test scripts (stdlib only, no dependencies):

```bash
python3 testhex.py    # 10 tests — random key, 8 unknown nibbles, BTC/ETH targets
python3 testwif.py    # 10 tests — random WIF, 5 unknown characters
python3 testseed.py   # 10 tests — random BIP39 phrase, unknown words
python3 testpass.py   # 10 tests — known mnemonic, brute-forces passphrase
python3 testbloom.py  # 5 tests — temporary bloom.bin, HEX/SEED/WIF in BTC+ETH bloom modes
```

Run the full suite after every recompile:

```bash
python3 testhex.py && python3 testwif.py && python3 testseed.py && python3 testpass.py && python3 testbloom.py
```

`testbloom.py` is destructive only to `bloom.bin` in the project directory for the duration of the test:

- if a real `bloom.bin` exists, it is renamed to a temporary backup
- a small deterministic test filter is written in its place
- the original `bloom.bin` is restored automatically at the end

Windows uses the same scripts:

```bat
python testhex.py
python testwif.py
python testseed.py
python testpass.py
python testbloom.py
```

Full Windows suite:

```bat
python testhex.py && python testwif.py && python testseed.py && python testpass.py && python testbloom.py
```

---

## Output & Notifications

When a match is found, Hydra:

1. Prints the private key (hex) and the matched address to stdout.
2. Sends a Telegram notification (if configured in `token.txt`).
3. In bloom mode: verifies the live balance via blockchain API — zero-balance hits are discarded and the search continues automatically.
4. Network errors during verification are written to `errors.json` for manual review.

---

## Support the Project

If Hydra helped you recover your funds, consider a donation.

**BTC:** `bc1qsn23hyqhwkw4775ssykdtegxqgmwpe9qns3y0m`  
**ETH / ERC-20:** `0x8f00CbC520876a62eE07b54c2266d988fE61cD86`

---

## License

MIT License — Copyright (c) 2026 Julienbxl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: the above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
