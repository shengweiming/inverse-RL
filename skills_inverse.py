"""
25 INVERTIBLE string-transformation skills for the inverse-RL experiment.
Each skill is INJECTIVE on its declared input domain, so the preimage is UNIQUE.
We provide forward `f` and reference inverse `f_inv`, plus per-skill input samplers
that only draw inputs on which f is injective. Round-trip tested at bottom.

Reused-from-original (Yuan et al. RL-Compositionality, string_data.py): marked [R]
New for this experiment: marked [N]
"""
import random, string
from math import gcd

# ---------- helpers ----------
LOWER = string.ascii_lowercase
ALNUM = string.ascii_letters + string.digits

def _letters_only(n):  return ''.join(random.choices(LOWER, k=n))
def _alnum(n):         return ''.join(random.choices(ALNUM, k=n))
def _words(nw):        return ' '.join(_letters_only(random.randint(2,5)) for _ in range(nw))

# ======================================================================
# TIER 1 — easy, local / char-wise (mostly involutions or simple maps)
# ======================================================================

# [R] recursive_reverse : involution
def reverse(s):            return s[::-1]
def reverse_inv(s):        return s[::-1]

# [N] swap_case : involution
def swap_case(s):          return s.swapcase()
def swap_case_inv(s):      return s.swapcase()

# [N] atbash (letters reflected a<->z); involution; non-letters fixed
def _atbash_ch(c):
    if 'a'<=c<='z': return chr(ord('z')-(ord(c)-ord('a')))
    if 'A'<=c<='Z': return chr(ord('Z')-(ord(c)-ord('A')))
    return c
def atbash(s):             return ''.join(_atbash_ch(c) for c in s)
def atbash_inv(s):         return ''.join(_atbash_ch(c) for c in s)

# [N] complement_digits d->9-d ; involution on digits; here inputs are digit strings
def complement_digits(s):  return ''.join(str(9-int(c)) if c.isdigit() else c for c in s)
def complement_digits_inv(s): return complement_digits(s)

# [R] shift_chars (Caesar) ; bijection on letters
def shift_chars(s, k=3):
    def sh(c):
        if 'a'<=c<='z': return chr((ord(c)-97+k)%26+97)
        if 'A'<=c<='Z': return chr((ord(c)-65+k)%26+65)
        return c
    return ''.join(sh(c) for c in s)
def shift_chars_inv(s, k=3): return shift_chars(s, -k)

# [N] shift_digits d->(d+k)%10 ; bijection on digits
def shift_digits(s, k=4):   return ''.join(str((int(c)+k)%10) if c.isdigit() else c for c in s)
def shift_digits_inv(s, k=4): return shift_digits(s, -k%10)

# [R] duplicate_every_char ; injective
def duplicate_every_char(s): return ''.join(c*2 for c in s)
def duplicate_every_char_inv(s): return s[::2]

# [R] fancy_brackets «c» per char ; injective
def fancy_brackets(s):     return ''.join("«"+c+"»" for c in s)
def fancy_brackets_inv(s): return s[1::3]  # «,c,» repeating

# [N] wrap_tag whole-string delimiters ; injective (strip fixed affixes)
def wrap_tag(s):           return "<<"+s+">>"
def wrap_tag_inv(s):       return s[2:-2]

# [R] add_prefix ; injective
def add_prefix(s, pre="pre_"): return pre+s
def add_prefix_inv(s, pre="pre_"): return s[len(pre):]

# ======================================================================
# TIER 2 — medium, global structure / parametric bijections
# ======================================================================

# [R] add_suffix ; injective
def add_suffix(s, suf="_end"): return s+suf
def add_suffix_inv(s, suf="_end"): return s[:-len(suf)]

# [R] rotate_str by n ; bijection
def rotate_str(s, n=2):
    if not s: return s
    n%=len(s); return s[n:]+s[:n]
def rotate_str_inv(s, n=2):
    if not s: return s
    n%=len(s); return s[-n:]+s[:-n] if n else s

# [N] rotate_words by n ; bijection on word list
def rotate_words(s, n=1):
    w=s.split(' ');
    if not w: return s
    n%=len(w); return ' '.join(w[n:]+w[:n])
def rotate_words_inv(s, n=1):
    w=s.split(' ')
    if not w: return s
    n%=len(w); return ' '.join(w[-n:]+w[:-n]) if n else s

# [R] repeat_str n times ; injective (n known)
def repeat_str(s, n=3):     return s*n
def repeat_str_inv(s, n=3): return s[:len(s)//n]

# [R] mirror_str s+reverse(s) ; injective (take first half)
def mirror_str(s):          return s+s[::-1]
def mirror_str_inv(s):      return s[:len(s)//2]

# [N] swap_halves (even length) ; = rotate by L/2, involution on even L
def swap_halves(s):         h=len(s)//2; return s[h:]+s[:h]
def swap_halves_inv(s):     h=len(s)//2; return s[h:]+s[:h]  # even-length -> involution

# [N] swap_pairs adjacent (0<->1,...) ; involution; odd leaves last
def swap_pairs(s):
    l=list(s)
    for i in range(0,len(l)-1,2): l[i],l[i+1]=l[i+1],l[i]
    return ''.join(l)
def swap_pairs_inv(s):      return swap_pairs(s)

# [R] insert_separator join chars with sep ; injective (sep absent from input)
def insert_separator(s, sep="-"): return sep.join(s)
def insert_separator_inv(s, sep="-"): return s.replace(sep,"")

# [R] reverse_words ; involution on word order (single spaces)
def reverse_words(s):       return ' '.join(reversed(s.split(' ')))
def reverse_words_inv(s):   return ' '.join(reversed(s.split(' ')))

# [N] succ_char : +k on codepoint within fixed printable band [32,126]; bijection
_LO,_HI = 32,126; _SPAN=_HI-_LO+1
def succ_char(s, k=1):      return ''.join(chr((ord(c)-_LO+k)%_SPAN+_LO) if _LO<=ord(c)<=_HI else c for c in s)
def succ_char_inv(s, k=1):  return succ_char(s, -k)

# ======================================================================
# TIER 3 — hard, position-dependent / permutation / classical ciphers
# ======================================================================

# [R] deterministic_shuffle : fixed multiplier permutation (coprime to L) ; bijection
def _mult(L):
    m=3
    while gcd(m,L)!=1: m+=2
    return m
def deterministic_shuffle(s):
    L=len(s)
    if L==0: return s
    m=_mult(L)
    return ''.join(s[(i*m)%L] for i in range(L))
def deterministic_shuffle_inv(s):
    L=len(s)
    if L==0: return s
    m=_mult(L); out=[None]*L
    for i in range(L): out[(i*m)%L]=s[i]
    return ''.join(out)

# [N] positional_shift : char i shifted by (i mod 26) over letters ; bijection
def positional_shift(s):
    out=[]
    for i,c in enumerate(s):
        k=i%26
        if 'a'<=c<='z': out.append(chr((ord(c)-97+k)%26+97))
        elif 'A'<=c<='Z': out.append(chr((ord(c)-65+k)%26+65))
        else: out.append(c)
    return ''.join(out)
def positional_shift_inv(s):
    out=[]
    for i,c in enumerate(s):
        k=i%26
        if 'a'<=c<='z': out.append(chr((ord(c)-97-k)%26+97))
        elif 'A'<=c<='Z': out.append(chr((ord(c)-65-k)%26+65))
        else: out.append(c)
    return ''.join(out)

# [N] vigenere repeating-key Caesar over lowercase letters ; bijection
def _vig(s, key, sign):
    out=[]; j=0
    for c in s:
        if 'a'<=c<='z':
            k=(ord(key[j%len(key)])-97)*sign
            out.append(chr((ord(c)-97+k)%26+97)); j+=1
        else: out.append(c)
    return ''.join(out)
def vigenere(s, key="key"):     return _vig(s, key, +1)
def vigenere_inv(s, key="key"): return _vig(s, key, -1)

# [N] rail_fence_2 : 2-rail zigzag transposition ; bijection
def rail_fence_2(s):
    a=s[0::2]; b=s[1::2]; return a+b
def rail_fence_2_inv(s):
    L=len(s); na=(L+1)//2
    a,b=s[:na],s[na:]; out=[]
    for i in range(L):
        out.append(a[i//2] if i%2==0 else b[i//2])
    return ''.join(out)

# [N] riffle_shuffle : perfect interleave of two halves (even L) ; bijection
def riffle_shuffle(s):
    h=len(s)//2; a,b=s[:h],s[h:]
    return ''.join(x+y for x,y in zip(a,b))
def riffle_shuffle_inv(s):
    a=s[0::2]; b=s[1::2]; return a+b

# ======================================================================
# REGISTRY:  name -> (forward, inverse, sampler, default-kwargs, tier, origin)
# ======================================================================
def even(n_lo,n_hi):
    n=random.randint(n_lo,n_hi); return n - (n%2)

SKILLS = {
 # tier 1
 "reverse":              (reverse, reverse_inv, lambda: _letters_only(random.randint(4,9)), {}, 1,"R"),
 "swap_case":            (swap_case, swap_case_inv, lambda: ''.join(random.choice([c,c.upper()]) for c in _letters_only(random.randint(4,9))), {},1,"N"),
 "atbash":               (atbash, atbash_inv, lambda: _letters_only(random.randint(4,9)), {},1,"N"),
 "complement_digits":    (complement_digits, complement_digits_inv, lambda: ''.join(random.choices(string.digits,k=random.randint(4,8))), {},1,"N"),
 "shift_chars":          (shift_chars, shift_chars_inv, lambda: _letters_only(random.randint(4,9)), {"k":3},1,"R"),
 "shift_digits":         (shift_digits, shift_digits_inv, lambda: ''.join(random.choices(string.digits,k=random.randint(4,8))), {"k":4},1,"N"),
 "duplicate_every_char": (duplicate_every_char, duplicate_every_char_inv, lambda: _letters_only(random.randint(4,8)), {},1,"R"),
 "fancy_brackets":       (fancy_brackets, fancy_brackets_inv, lambda: _letters_only(random.randint(4,8)), {},1,"R"),
 "wrap_tag":             (wrap_tag, wrap_tag_inv, lambda: _alnum(random.randint(4,9)), {},1,"N"),
 "add_prefix":           (add_prefix, add_prefix_inv, lambda: _alnum(random.randint(4,9)), {"pre":"pre_"},1,"R"),
 # tier 2
 "add_suffix":           (add_suffix, add_suffix_inv, lambda: _alnum(random.randint(4,9)), {"suf":"_end"},2,"R"),
 "rotate_str":           (rotate_str, rotate_str_inv, lambda: _letters_only(random.randint(5,9)), {"n":2},2,"R"),
 "rotate_words":         (rotate_words, rotate_words_inv, lambda: _words(random.randint(3,5)), {"n":1},2,"N"),
 "repeat_str":           (repeat_str, repeat_str_inv, lambda: _letters_only(random.randint(3,6)), {"n":3},2,"R"),
 "mirror_str":           (mirror_str, mirror_str_inv, lambda: _letters_only(random.randint(4,8)), {},2,"R"),
 "swap_halves":          (swap_halves, swap_halves_inv, lambda: _letters_only(even(6,10)), {},2,"N"),
 "swap_pairs":           (swap_pairs, swap_pairs_inv, lambda: _letters_only(even(6,10)), {},2,"N"),
 "insert_separator":     (insert_separator, insert_separator_inv, lambda: _letters_only(random.randint(4,8)), {"sep":"-"},2,"R"),
 "reverse_words":        (reverse_words, reverse_words_inv, lambda: _words(random.randint(3,5)), {},2,"R"),
 "succ_char":            (succ_char, succ_char_inv, lambda: _letters_only(random.randint(4,9)), {"k":1},2,"N"),
 # tier 3
 "deterministic_shuffle":(deterministic_shuffle, deterministic_shuffle_inv, lambda: _letters_only(random.randint(5,9)), {},3,"R"),
 "positional_shift":     (positional_shift, positional_shift_inv, lambda: _letters_only(random.randint(5,9)), {},3,"N"),
 "vigenere":             (vigenere, vigenere_inv, lambda: _letters_only(random.randint(5,9)), {"key":"key"},3,"N"),
 "rail_fence_2":         (rail_fence_2, rail_fence_2_inv, lambda: _letters_only(random.randint(5,9)), {},3,"N"),
 "riffle_shuffle":       (riffle_shuffle, riffle_shuffle_inv, lambda: _letters_only(even(6,10)), {},3,"N"),
}

if __name__ == "__main__":
    random.seed(0)
    print(f"Total skills: {len(SKILLS)}  (target 25)")
    n_reused = sum(1 for v in SKILLS.values() if v[5]=="R")
    print(f"  reused [R]: {n_reused}   new [N]: {len(SKILLS)-n_reused}")
    print("\nRound-trip & injectivity test (2000 samples each):")
    all_ok=True
    for name,(f,finv,samp,kw,tier,origin) in SKILLS.items():
        seen={}; rt_ok=True; inj_ok=True
        for _ in range(2000):
            x=samp()
            y=f(x,**kw)
            xr=finv(y,**kw)
            if xr!=x: rt_ok=False; bad=(x,y,xr); break
            if y in seen and seen[y]!=x: inj_ok=False; bad=(x,y,seen[y]); break
            seen[y]=x
        flag="OK" if (rt_ok and inj_ok) else "FAIL"
        if flag=="FAIL": all_ok=False
        extra="" if (rt_ok and inj_ok) else f"  <-- rt={rt_ok} inj={inj_ok} {bad}"
        print(f"  [{flag}] T{tier} {origin} {name}{extra}")
    print("\nALL GOOD" if all_ok else "\nSOME FAILED")
