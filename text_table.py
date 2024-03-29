_values = '''
$4F    
$57  #
$51  *
$52  A1
$53  A2
$54  POKé
$55  +
$58  $
$75  …
$7F   
$80  A
$81  B
$82  C
$83  D
$84  E
$85  F
$86  G
$87  H
$88  I
$89  J
$8A  K
$8B  L
$8C  M
$8D  N
$8E  O
$8F  P
$90  Q
$91  R
$92  S
$93  T
$94  U
$95  V
$96  W
$97  X
$98  Y
$99  Z
$9A  (
$9B  )
$9C  :
$9D  ;
$9E  [
$9F  ]
$A0  a
$A1  b
$A2  c
$A3  d
$A4  e
$A5  f
$A6  g
$A7  h
$A8  i
$A9  j
$AA  k
$AB  l
$AC  m
$AD  n
$AE  o
$AF  p
$B0  q
$B1  r
$B2  s
$B3  t
$B4  u
$B5  v
$B6  w
$B7  x
$B8  y
$B9  z
$BA  é
$BB  'd
$BC  'l
$BD  's
$BE  't
$BF  'v
$E0  '
$E1  PK
$E2  MN
$E3  -
$E4  'r
$E5  'm
$E6  ?
$E7  !
$E8  .
$ED  →
$EE  ↓
$EF  ♂
$F0  ¥
$F1  ×
$F3  /
$F4  ,
$F5  ♀
$F6  0
$F7  1
$F8  2
$F9  3
$FA  4
$FB  5
$FC  6
$FD  7
$FE  8
$FF  9
'''

def _init_text_table():
    _splits = _values.split('\n$')

    del _splits[0]
    _splits[-1] = _splits[-1][:-1]
    _splits[0] = _splits[0][:-1]

    table = {}
    for s in _splits:
        k, v = int(s[:2], 16), s[4:]
        table[k] = v

    return table

text_table = _init_text_table()