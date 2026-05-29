import subprocess
from testpass import gen_phrase, gen_passphrase, pub, master, mn2seed, bip44, eth_addr

phrase = ' '.join(gen_phrase(12))
base = gen_passphrase()
print('Testing:', base, phrase)
with open('dictionary.txt', 'w') as f:
    f.write(base + '\n')
with open('rule.txt', 'w') as f:
    f.write(':\n')

mp, mc = master(mn2seed(phrase, base))
k = int.from_bytes(bip44(mp, mc, 60), 'big')
addr = eth_addr(k)
print('ETH addr:', addr)

# We use the known fetchable address '0xb40f147b02748133a573c71b3afbd885718dde79' to test the fetch mechanism!
cmd = ['./Hydra', phrase, '0xb40f147b02748133a573c71b3afbd885718dde79']
res = subprocess.run(cmd, cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'), capture_output=True, text=True)
print(res.stdout[:1000])
