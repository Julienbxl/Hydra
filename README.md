# Hydra V5

CUDA-accelerated wallet recovery for Bitcoin and Ethereum.

Hydra is built for legitimate recovery cases: partial private keys, WIF keys, BIP39 seeds, Electrum seeds, BIP39 passphrases, and brainwallet candidates. It can target one address/pubkey or a large Bloom filter.

## Modes At A Glance

Benchmarks below are indicative values measured on an RTX 5060-class GPU. Real speed depends on GPU, mask shape, target type, Bloom size, VRAM, and whether the target is an address or a known public key.

| Mode | Typical Speed | What It Recovers | Best Target | Example |
|---|---:|---|---|---|
| HEX | ~1,250 MKey/s | Missing hex nibbles in a 32-byte private key | address, pubkey, Bloom | `./Hydra <64hex_mask> bloombtc` |
| HEX BSGS | ~640 PKey/s equivalent | Larger HEX gaps when the public key is known | pubkey | `./Hydra <64hex_mask> <pubkey>` |
| WIF | ~3,700 MKey/s | Missing Base58 characters in WIF keys | BTC address, pubkey, Bloom | `./Hydra <wif_mask> <btc_target>` |
| WIF BSGS | ~200 PKey/s equivalent | Larger WIF gaps when the public key is known | pubkey | `./Hydra <wif_mask> <pubkey>` |
| BIP39 Seed | ~5 MKey/s raw | Missing words in 12/24-word BIP39 mnemonics | address, pubkey, Bloom | `./Hydra "word # word ..." bloombtc` |
| Electrum V2 | ~50 MKey/s raw | Missing words in Electrum V2 seeds | address, pubkey, Bloom | `./Hydra "word # word ..." bloombtc --electrumV2` |
| Electrum V1 | ~8,000 cand/s | Old Electrum V1 seeds | BTC legacy address, Bloom | `./Hydra "old # seed ..." <target> --electrumV1` |
| Brainwallet | ~60 MKey/s | Lines from `resources/brainwallet.txt` | address, pubkey, Bloom | `./Hydra brainwallet bloombtc` |
| Passphrase | ~0.2 MKey/s | BIP39 optional passphrase from `resources/dictionary.txt` | address, pubkey, Bloom | `./Hydra "full seed phrase" <target>` |

BSGS speeds are expressed as equivalent searched keyspace. For example, HEX BSGS at 600 Msteps/s with a baby-7.5 split is about `600e6 * 16^7.5 = 640 PKey/s`. WIF BSGS at 305 Msteps/s with baby-5 is about `305e6 * 58^5 = 200 PKey/s`.

Larger BSGS splits are useful on high-end GPU/RAM configurations:

| Mode | Split | Approximate Requirement | Notes |
|---|---:|---|---|
| HEX BSGS | baby7.5 | ~4.4 GiB VRAM + ~6.4 GiB RAM | Safe large-mask RAM backend |
| HEX BSGS | baby7.5 VRAM | ~13 GiB VRAM | Experimental opt-in only with `--baby=7.5-vram` |
| HEX BSGS | baby8 | ~4.4 GiB VRAM + ~24.4 GiB RAM | Experimental opt-in only with `--baby=8` |
| WIF BSGS | baby5 RAM | ~7.5 GiB RAM | Current large WIF path |
| WIF BSGS | baby5 VRAM | ~41 GiB VRAM fully GPU-resident | Experimental opt-in only with `--baby=5-vram` |
| WIF BSGS | baby5.5 | ~64 GiB RAM in experimental RAM-baby mode | Experimental opt-in only with `--baby=5.5` |

The automatic scheduler only selects tested production splits: HEX baby7.5 RAM and WIF baby5 RAM. HEX baby7.5 VRAM, WIF baby5 VRAM, HEX baby8 RAM, and WIF baby5.5 RAM are deliberately opt-in until they get real hardware feedback.

## Quick Start

Hydra uses simple positional masks. Put `#` where characters or words are unknown.

```bash
./Hydra "<mask>" <address|pubkey|bloom|bloombtc|bloometh>
```

Targets:

| Target | Meaning |
|---|---|
| BTC address | Legacy or SegWit address |
| ETH address | Ethereum address |
| compressed/uncompressed pubkey | Enables pubkey-only paths and BSGS scheduling |
| `bloom` | Search the loaded Bloom filter on BTC and ETH paths where the mode supports both |
| `bloombtc` | Bloom search, BTC only |
| `bloometh` | Bloom search, ETH only |

Hydra auto-selects BSGS for HEX/WIF when the mask is large enough and the target public key is known. If you give an address, Hydra tries to recover the public key automatically from public chain data when possible. For fully offline recovery, provide the public key directly.

WIF and Electrum modes are Bitcoin-only formats. Use `bloombtc` for those modes; generic `bloom` is accepted where supported, but it is routed to the BTC Bloom path and never scans Ethereum. ETH addresses, ETH pubkeys, and `bloometh` are rejected for Electrum modes.

## Usage By Mode

### HEX Private Key

Recover missing hex nibbles in a 64-character private key.

```bash
./Hydra 7cb5da6f7757##14a59#f40dc45739eda5e532804f24af675e3339f1fe9c4 1Address
```

For larger gaps, Hydra can use BSGS if the public key is known. With an address target, Hydra tries to fetch the public key automatically when it is available on-chain. For offline use, pass the public key directly:

```bash
./Hydra ffffffffffffffffffffffffffffffffffffffffffffffffffffff########## 0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
```

### WIF

Recover missing Base58 characters in compressed or uncompressed Bitcoin WIF keys.

```bash
./Hydra KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qY#rFej#um7Wt#CRUx 1Address
```

With a known public key, large WIF masks use the BSGS engine.

```bash
./Hydra KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9N9w######### <compressed_or_uncompressed_pubkey>
```

### BIP39 Seed

Recover missing words from a 12-word or 24-word BIP39 mnemonic. The checksum is enforced on GPU before PBKDF2.

```bash
./Hydra "rally # # bracket know opera produce jar # expire # solar" bloombtc
```

Default BTC path is `m/44'/0'/0'/0/0`. ETH uses `m/44'/60'/0'/0/0`.

### Electrum V2

Electrum V2 uses the BIP39 wordlist but Electrum's own seed-version checksum and derivation.

```bash
./Hydra "rally # # bracket know opera produce jar raccoon expire # solar" bloombtc --electrumV2
```

Hydra supports standard and SegWit Electrum V2 prefixes and derives `m/0/0`.

### Electrum V1

Old Electrum V1 uses its own 1626-word list and a much heavier legacy stretch.

```bash
./Hydra "like just # know # want time out there make look #" 1Address --electrumV1
```

By default Hydra scans `m/0/0..19` and `m/1/0..19`. You can restrict the scan:

```bash
./Hydra "<old electrum mask>" <target> --electrumV1 --path m/0/0
./Hydra "<old electrum mask>" <target> --electrumV1 --gap 50
```

### BIP39 Passphrase

Use this when the mnemonic is known but the optional BIP39 passphrase is not. Candidates are read from `resources/dictionary.txt`.

```bash
./Hydra "rally baby bracket know opera produce jar raccoon expire solar dog cat" bloombtc
```

If `resources/rule.txt` exists, rules are applied on GPU to each dictionary line.

### Brainwallet

Candidates are read from `resources/brainwallet.txt`. Each line is hashed as `SHA256(line)` and used as a private key.

```bash
./Hydra brainwallet bloombtc
```

If `resources/rule.txt` exists, Hydra applies the supported rule set to each line. Brainwallet mode also checks the inverse CP-trick variant.

## Resource Files

| File | Used By | Purpose |
|---|---|---|
| `resources/bloom.bin` | Bloom modes | Target address filter |
| `resources/dictionary.txt` | Passphrase | Candidate passphrases |
| `resources/brainwallet.txt` | Brainwallet | Candidate brainwallet lines |
| `resources/rule.txt` | Passphrase, Brainwallet | Hashcat-style mutation rules |
| `telegram.txt` | All modes | Optional Telegram notifications: line 1 is the bot token, line 2 is the chat id |
| `errors.json` | Bloom/API/Telegram | Append-only log for unverified Bloom hits and Telegram delivery failures |

Create a Bloom filter from text files:

```bash
python3 tools/create_bloom.py btc_addresses.txt eth_addresses.txt resources/bloom.bin
```

`telegram.txt` format:

```text
123456789:telegram_bot_token_here
123456789
```

Bloom filters can produce false positives. When a Bloom hit cannot be verified because the balance API is unavailable, Hydra writes the private key and derived addresses to `errors.json` so the candidate can be checked later. If a real victory is found but Telegram delivery fails, that message is also saved to `errors.json`.

## Rule Engine

Hydra supports the common Hashcat rule operations needed by typical rule files such as `best64`, `nsa64`, and small custom mutation sets. Rules are parsed from `resources/rule.txt` and executed on GPU in passphrase and brainwallet modes.

Supported families include case transforms, append/prepend, replace, insert/delete, truncate/extract, duplicate, rotate, purge, and reject filters. Advanced Hashcat memory rules and character-class rules are not currently implemented.

## Resume

Hydra periodically writes a resume snapshot. Stop a run with `Ctrl+C`, then continue with:

```bash
./Hydra resume
```

Resume is supported across the main long-running modes, including BSGS, seed, Electrum, WIF, passphrase, and brainwallet.

## Build

Linux / WSL:

```bash
cmake --preset linux-release
cmake --build --preset linux-release
```

The default release build targets `sm_86`, `sm_89`, and `sm_120`. You can disable individual targets with CMake options such as `-DHYDRA_TARGET_SM120=OFF`.

Windows:

```bat
vcpkg install openssl:x64-windows
cmake --preset windows-release
cmake --build --preset windows-release
```

## Tests

The repository includes smoke tests for the main recovery modes:

```bash
python3 tests/testhex.py
python3 tests/testwif.py
python3 tests/testseed.py
python3 tests/testelectrumv1.py
python3 tests/testelectrumv2.py
python3 tests/testbloom.py
python3 tests/testbrainwallet.py
python3 tests/testpass.py
python3 tests/testbsgshex.py
python3 tests/testbsgswif.py
```

## Disclaimer

Hydra is intended only for recovering wallets you own or are explicitly authorized to audit. Do not use it against other people's wallets, addresses, or systems.

If Hydra helped you recover funds:

- BTC: `bc1qsn23hyqhwkw4775ssykdtegxqgmwpe9qns3y0m`
- ETH: `0x8f00CbC520876a62eE07b54c2266d988fE61cD86`

## License

Hydra is licensed under the PolyForm Noncommercial License 1.0.0.

Noncommercial use, modification, and redistribution are permitted under the license terms. Commercial use, including wallet-recovery services or commercial redistribution, requires explicit written permission from the author.

Required Notice: Copyright (c) 2026 Julienbxl.

See [LICENSE.md](LICENSE.md) for the full license text.
