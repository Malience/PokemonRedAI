_values='''001	0x01	Master Ball
002	0x02	Ultra Ball
003	0x03	Great Ball
004	0x04	Poké Ball
005	0x05	Town Map
006	0x06	Bicycle
007	0x07	?????
008	0x08	Safari Ball
009	0x09	Pokédex
010	0x0A	Moon Stone
011	0x0B	Antidote
012	0x0C	Burn Heal
013	0x0D	Ice Heal
014	0x0E	Awakening
015	0x0F	Parlyz Heal
016	0x10	Full Restore
017	0x11	Max Potion
018	0x12	Hyper Potion
019	0x13	Super Potion
020	0x14	Potion
021	0x15	BoulderBadge
022	0x16	CascadeBadge
023	0x17	ThunderBadge
024	0x18	RainbowBadge
025	0x19	SoulBadge
026	0x1A	MarshBadge
027	0x1B	VolcanoBadge
028	0x1C	EarthBadge
029	0x1D	Escape Rope
030	0x1E	Repel
031	0x1F	Old Amber
032	0x20	Fire Stone
033	0x21	Thunderstone
034	0x22	Water Stone
035	0x23	HP Up
036	0x24	Protein
037	0x25	Iron
038	0x26	Carbos
039	0x27	Calcium
040	0x28	Rare Candy
041	0x29	Dome Fossil
042	0x2A	Helix Fossil
043	0x2B	Secret Key
044	0x2C	?????
045	0x2D	Bike Voucher
046	0x2E	X Accuracy
047	0x2F	Leaf Stone
048	0x30	Card Key
049	0x31	Nugget
050	0x32	PP Up*
051	0x33	Poké Doll
052	0x34	Full Heal
053	0x35	Revive
054	0x36	Max Revive
055	0x37	Guard Spec.
056	0x38	Super Repel
057	0x39	Max Repel
058	0x3A	Dire Hit
059	0x3B	Coin
060	0x3C	Fresh Water
061	0x3D	Soda Pop
062	0x3E	Lemonade
063	0x3F	S.S. Ticket
064	0x40	Gold Teeth
065	0x41	X Attack
066	0x42	X Defend
067	0x43	X Speed
068	0x44	X Special
069	0x45	Coin Case
070	0x46	Oak's Parcel
071	0x47	Itemfinder
072	0x48	Silph Scope
073	0x49	Poké Flute
074	0x4A	Lift Key
075	0x4B	Exp. All
076	0x4C	Old Rod
077	0x4D	Good Rod
078	0x4E	Super Rod
079	0x4F	PP Up
080	0x50	Ether
081	0x51	Max Ether
082	0x52	Elixer
083	0x53	Max Elixer
196	0xC4	HM01
197	0xC5	HM02
198	0xC6	HM03
199	0xC7	HM04
200	0xC8	HM05
201	0xC9	TM01
202	0xCA	TM02
203	0xCB	TM03
204	0xCC	TM04
205	0xCD	TM05
206	0xCE	TM06
207	0xCF	TM07
208	0xD0	TM08
209	0xD1	TM09
210	0xD2	TM10
211	0xD3	TM11
212	0xD4	TM12
213	0xD5	TM13
214	0xD6	TM14
215	0xD7	TM15
216	0xD8	TM16
217	0xD9	TM17
218	0xDA	TM18
219	0xDB	TM19
220	0xDC	TM20
221	0xDD	TM21
222	0xDE	TM22
223	0xDF	TM23
224	0xE0	TM24
225	0xE1	TM25
226	0xE2	TM26
227	0xE3	TM27
228	0xE4	TM28
229	0xE5	TM29
230	0xE6	TM30
231	0xE7	TM31
232	0xE8	TM32
233	0xE9	TM33
234	0xEA	TM34
235	0xEB	TM35
236	0xEC	TM36
237	0xED	TM37
238	0xEE	TM38
239	0xEF	TM39
240	0xF0	TM40
241	0xF1	TM41
242	0xF2	TM42
243	0xF3	TM43
244	0xF4	TM44
245	0xF5	TM45
246	0xF6	TM46
247	0xF7	TM47
248	0xF8	TM48
249	0xF9	TM49
250	0xFA	TM50
251	0xFB	TM51
252	0xFC	TM52
253	0xFD	TM53
254	0xFE	TM54
255	0xFF	TM55'''

def _init_item_table():
    _splits = _values.split('\n')
    dct = {}
    for splt in _splits:
        dct[int(splt[:3])] = splt[9:]
    return dct

item_table = _init_item_table()