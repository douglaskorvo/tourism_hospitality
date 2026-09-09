# -*- coding: utf-8 -*-
"""
Lexicon v2 -- destination-level hospitality analysis of Google Maps reviews
served in Brazilian Portuguese (native + machine-translated).

Design rule (attribution-first coding)
--------------------------------------
Every marker is declared either UNAMBIGUOUS (it can only refer to one
referent class -- e.g. 'hospitaleiro' refers to a host; 'patrimonio' refers
to a place) or AMBIGUOUS (its referent depends on context -- e.g. 'limpo',
'lindo', 'historia').  Ambiguous markers are only credited to a construct
when the nearest referent anchor inside a +/-12-token window is of the
matching class.  The strict variants used in the analysis apply this rule;
the permissive variants are retained for sensitivity analysis.

Constructs
----------
HOSP   hospitableness -- host-guest relational conduct
       (Lashley 2000; Blain & Lashley 2014; Ariffin & Maghzi 2012;
        Tasci & Semrad 2016).  Sub-dimensions: warmth, welcome,
        attentiveness, generosity, personal bond.
SQ     commercial service quality, SERVQUAL-style provider performance
       (Parasuraman, Zeithaml & Berry 1988).  Sub-dimensions: reliability,
       responsiveness, competence/assurance, tangibles, value.  Valenced.
DQ     perceived destination quality -- the place itself
       (Echtner & Ritchie 1993; Chen & Tsai 2007).  Sub-dimensions:
       scenery, heritage, environment, safety, atmosphere.  Valenced.
EXP    generic experiential affect -- discriminant-validity control that
       absorbs undirected enthusiasm ('inesquecivel', 'incrivel').
INT    behavioural intention -- recommendation and revisit.  Outcome.

'~' at the end of a stem means 'any suffix'.  Matching is performed on
casefolded, diacritic-stripped text with word boundaries.
"""

# =========================================================================
# 1. HOSPITABLENESS  (host is the provider / its staff)
# =========================================================================
HOSP = {
    # unambiguous: describe a person's relational conduct
    "warmth": [
        r"simpatic~", r"simpatiss?im~", r"simpatia", r"gentil", r"gentileza",
        r"gentilissim~", r"amavel", r"amaveis", r"amabilidade", r"cordial~",
        r"cortes", r"cortesia", r"educad~", r"bem.?educad~", r"carinhos~",
        r"carinho", r"afetuos~", r"afavel", r"amigavel", r"amigaveis",
        r"sorriso", r"sorridente", r"sorrindo", r"bom humor",
        r"bem.?humorad~", r"caloros~", r"calor human~", r"docura", r"meig~",
        r"receptiv~", r"simpaticissim~",
    ],
    "welcome": [
        r"acolhedor~", r"acolhid~", r"acolhiment~", r"acolheram", r"acolheu",
        r"hospitalidade", r"hospitaleir~", r"bem.?vind~", r"bem recebid~",
        r"nos receberam", r"me receberam", r"nos recebeu", r"me recebeu",
        r"recepc~ caloros~", r"boas.?vindas", r"aconchegante", r"aconchego",
        r"como em casa", r"sentir em casa", r"nos sentimos em casa",
        r"me senti em casa", r"segunda casa", r"de brac~ abert~",
    ],
    "attentiveness": [
        r"atencios~", r"atenciosissim~", r"prestativ~", r"solicit~",
        r"disponivel", r"disponibilidade", r"paciente", r"paciencia",
        r"cuidados~", r"zelos~", r"zelo", r"atent~ a", r"atent~ as",
        r"atent~ ao", r"nos ajud~", r"me ajud~", r"ajudou", r"ajudaram",
        r"preocupad~ com", r"nos tratou", r"nos trataram", r"me tratou",
        r"me trataram", r"bem tratad~", r"nos acompanh~",
        r"cuidaram de (?:nos|mim|tudo)", r"cuidou de (?:nos|mim|tudo)",
        r"cuidad~ com tudo", r"amparo", r"nos deu suporte", r"suporte total",
    ],
    "generosity": [
        r"generos~", r"generosidade", r"foi alem", r"foram alem",
        r"alem do esperad~", r"alem do combinad~", r"alem das expectativas",
        r"sem cobrar", r"sem custo (?:adicional|extra|algum)",
        r"cortesia da casa", r"nos ofereceu", r"nos ofereceram",
        r"nos presente~", r"mimo~", r"nos surpreendeu", r"surpresa agradavel",
        r"se desdobr~", r"fez questao", r"fizeram questao",
        r"nao mediu esforc~", r"nao mediram esforc~", r"nada era demais",
    ],
    "personal_bond": [
        r"como famili~", r"da famili~", r"parte da famili~",
        r"nos tratou como", r"nos trataram como", r"como amig~",
        r"virou amig~", r"viramos amig~", r"amizade", r"personalizad~",
        r"toque pessoal", r"atenc~ personalizad~", r"nos fez sentir",
        r"nos fizeram sentir", r"me fez sentir", r"me fizeram sentir",
        r"nos sentimos especiai?s", r"sentimos especiai?s", r"humaniz~",
        r"empati~", r"de coracao", r"do coracao", r"fundo do coracao",
        r"todo o coracao", r"como se fosse",
    ],
}
HOSP_AMBIGUOUS = set()          # HOSP markers are host-referring by construction  # noqa: E501

# =========================================================================
# 2. SERVICE QUALITY  (provider performance)
# =========================================================================
SQ_POS = {
    "reliability": [
        r"pontual~", r"pontualidade", r"no horario", r"em horario",
        r"dentro do prazo", r"antes do prazo", r"cumpriu", r"cumpriram",
        r"conforme combinad~", r"como combinad~", r"conforme prometid~",
        r"organizad~", r"organizac~", r"bem planejad~", r"planejament~",
        r"confiavel", r"confiabilidade", r"de confianc~",
        r"passa confianc~", r"transmite confianc~", r"sem atras~",
        r"sem contratempo~", r"sem problema~", r"tudo certo",
        r"tudo ocorreu bem", r"correu tudo bem", r"deu tudo certo",
        r"impecavel", r"impecaveis", r"honest~", r"honestidade",
        r"transparent~", r"transparenci~", r"seriedade", r"pontualissim~",
    ],
    "responsiveness": [
        r"agil", r"agilidade", r"prontamente", r"pronto atendiment~",
        r"resposta rapid~", r"respostas rapid~", r"respondeu", r"responderam",
        r"retorno rapid~", r"resolveu", r"resolveram", r"resolvid~",
        r"solucion~", r"sempre disponivel", r"pronto para ajudar",
        r"flexivel", r"flexibilidade", r"sem burocraci~", r"pratico",
        r"praticidade", r"rapidez", r"resposta imediat~", r"atendeu rapido",
    ],
    "competence": [
        r"profissional~", r"profissionalismo", r"competent~", r"competenci~",
        r"conhecedor~", r"experient~", r"qualificad~", r"capacitad~",
        r"bem preparad~", r"dominio", r"explicou", r"explicaram",
        r"explicac~", r"informativ~", r"esclarec~", r"bem informad~",
        r"expertise", r"eficient~", r"eficienci~", r"eficaz",
        r"muito conhecimento", r"amplo conhecimento", r"pleno conhecimento",
        r"vasto conhecimento", r"bem versad~", r"sabe tudo sobre",
    ],
    "tangibles": [
        r"confortavel", r"confortaveis", r"conforto", r"novo em folha",
        r"novissim~", r"bem conservad~", r"climatizad~", r"ar.?condicionad~",
        r"bem equipad~", r"boa estrutura", r"otima estrutura",
        r"excelente estrutura", r"otimas instalac~", r"veiculo novo",
        r"carro novo", r"van nova", r"onibus novo", r"bem mantid~",
        r"higieniz~", r"boa manutenc~", r"excelente manutenc~",
    ],
    "value": [
        r"preco just~", r"preco razoavel", r"precos razoavei?s",
        r"preco bo~", r"precos bo~", r"preco acessivel", r"precos acessivei?s",
        r"custo.?beneficio", r"vale cada centavo", r"vale o preco",
        r"vale o investiment~", r"vale a pena", r"valeu a pena",
        r"economic~", r"preco justo", r"sem cobranc~ extra",
        r"sem taxa~ escondid~", r"otimo preco", r"bom preco",
    ],
}
SQ_NEG = {
    "reliability_neg": [
        r"atras~", r"atrasad~", r"nao cumpriu", r"nao cumpriram",
        r"descumpriu", r"nao apareceu", r"nao apareceram", r"cancelou",
        r"cancelaram", r"foi cancelad~", r"foram cancelad~",
        r"desorganizad~", r"desorganizac~", r"bagunc~", r"confus~",
        r"mal planejad~", r"sem planejament~", r"nao era o combinad~",
        r"diferente do combinad~", r"diferente do prometid~", r"nao entregou",
        r"nao entregaram", r"deixou na mao", r"deixaram na mao",
        r"perdemos o (?:voo|passeio|onibus)", r"perdi o (?:voo|passeio)",
        r"nao honr~", r"calote",
    ],
    "responsiveness_neg": [
        r"nao respond~", r"sem resposta", r"nunca respond~", r"ignorou",
        r"ignoraram", r"demorou", r"demoraram", r"demorad~", r"lentid~",
        r"esperei horas", r"esperamos horas", r"ficamos esperando",
        r"nao resolveu", r"nao resolveram", r"descaso", r"pouco caso",
        r"nao deram retorno", r"sem retorno",
    ],
    "competence_neg": [
        r"despreparad~", r"desprepar~", r"amador~", r"incompetent~",
        r"nao sabia", r"nao sabiam", r"nao soube", r"desinformad~",
        r"sem conhecimento", r"nao explic~", r"nao domina",
    ],
    "conduct_neg": [
        r"grosseir~", r"grosseria", r"mal.?educad~", r"rude", r"rudeza",
        r"arrogant~", r"antipatic~", r"antipatia", r"agressiv~",
        r"gritou", r"gritaram", r"desrespeit~", r"humilh~", r"constrang~",
        r"mal atendid~", r"pessim~ atendiment~", r"pessim~ servic~",
        r"mau atendiment~", r"destrat~", r"mal recebid~",
    ],
    "value_neg": [
        r"car~ demais", r"muito car~", r"super.?faturad~", r"abusiv~",
        r"cobrou a mais", r"cobraram a mais", r"cobranc~ indevid~",
        r"taxa~ escondid~", r"golpe", r"enganad~", r"enganaram",
        r"engan~ o cliente", r"roub~ no preco", r"extorsao",
        r"nao vale o preco", r"nao vale o que", r"nao vale a pena",
        r"dinheiro jogad~ fora", r"prejuiz~", r"estelionat~", r"fraude",
        r"picaretagem", r"trapac~", r"propaganda enganos~",
    ],
    "tangibles_neg": [
        r"mal conservad~", r"quebrad~", r"estragad~", r"desconfortavel",
        r"desconfortaveis", r"apertad~", r"sem ar.?condicionad~",
        r"mau cheir~", r"cheiro ruim", r"fedor", r"sucat~", r"caindo aos peda",
    ],
}
# 'limpo/sujo' can describe a vehicle/room OR a city -> resolve by anchor
SQ_AMBIGUOUS_POS = {"tangibles": [
    r"limp~", r"limpeza", r"modern~", r"espacos~"]}
SQ_AMBIGUOUS_NEG = {"tangibles_neg": [r"suj~", r"sujeira"],
                    "conduct_neg": [r"assedi~", r"importun~"]}

# =========================================================================
# 3. PERCEIVED DESTINATION QUALITY
# =========================================================================
DQ_POS = {
    # unambiguous place referents
    "heritage": [
        r"patrimoni~", r"monument~", r"museu~", r"arquitetur~", r"ruina~",
        r"templo~", r"palacio~", r"castelo~", r"tumba~", r"faraonic~",
        r"colonial", r"milenar~", r"sitio arqueologic~", r"arqueologic~",
        r"medina", r"mesquita~", r"catedral", r"igreja historic~",
        r"fortaleza", r"pirami?de~", r"obelisc~", r"necropole", r"souk",
        r"bazar", r"centro historic~", r"cidade velha", r"casario",
        r"acropole", r"hieroglif~", r"artefato~", r"antiguidade~",
    ],
    "scenery": [
        r"paisage~", r"cenari~", r"natureza", r"exuberant~",
        r"paradisiac~", r"panoram~", r"por do sol", r"nascer do sol",
        r"de tirar o folego", r"vista maravilhos~", r"vista lind~",
        r"vista espetacular~", r"vistas deslumbrant~", r"vista incrivel",
        r"vista privilegiad~", r"esplendid~", r"magnific~ (?:vista|paisage)",
        r"belez~ natural", r"belez~ da (?:cidade|regiao|natureza)",
        r"maravilh~ natural",
    ],
    "environment": [
        r"cidade limp~", r"limpeza da cidade", r"cidade bem cuidad~",
        r"bem cuidad~ a cidade", r"infraestrutura", r"bem sinalizad~",
        r"facil de circular", r"facil locomoc~", r"transporte publico bo~",
        r"cidade organizad~", r"cidade arboriz~", r"cidade preservad~",
        r"bem preservad~", r"bem conservad~ a cidade", r"cidade verde",
    ],
    "safety": [
        r"cidade segur~", r"lugar segur~", r"me senti segur~",
        r"nos sentimos segur~", r"sentimos segur~", r"muito segur~",
        r"totalmente segur~", r"sem medo", r"pode andar tranquil~",
        r"seguranc~ da cidade", r"pais segur~", r"regiao segur~",
        r"anda tranquil~", r"tranquil~ para andar",
    ],
    "atmosphere": [
        r"charmos~", r"charme", r"encantador~", r"cidade magic~",
        r"apaixon~ pela cidade", r"apaixon~ pelo (?:lugar|pais)",
        r"amei a cidade", r"amamos a cidade", r"cidade incrivel",
        r"cidade maravilhos~", r"lugar magic~", r"atmosfera da cidade",
        r"clima da cidade", r"energi~ da cidade", r"vibe da cidade",
        r"alma da cidade",
    ],
}
DQ_NEG = {
    # unambiguous: the place is named inside the expression itself
    "environment_neg": [
        r"cidade suj~", r"suj~ a cidade", r"sujeira na", r"lixo na",
        r"lixo pela", r"poluic~", r"poluid~", r"transito caotic~",
        r"transito ruim", r"engarrafament~", r"cidade mal cuidad~",
        r"cidade feia", r"esgoto", r"esgoto a ceu abert~",
        r"ruas esburacad~", r"cidade degradad~", r"sem infraestrutura",
        r"falta de infraestrutura",
    ],
    "safety_neg": [
        r"cidade insegur~", r"cidade perigos~", r"regiao perigos~",
        r"bairro perigos~", r"pais insegur~", r"lugar perigos~",
        r"nao me senti segur~", r"nao nos sentimos segur~",
        r"evite a regiao", r"cidade violent~", r"nao ande sozinh~",
        r"assalt~ na rua", r"roub~ na rua", r"nao e segur~ andar",
    ],
    "harassment": [
        r"armadilha para turista~", r"explora~ o turista",
        r"exploram o turista", r"exploracao do turista",
        r"tentam te enganar", r"golpe para turista~", r"turista~ e alvo",
        r"cambist~", r"pedint~",
    ],
    "crowding": [
        r"cheio de turista~", r"turistificad~", r"turismo de massa",
        r"excesso de turista~", r"massificad~",
    ],
}
# ambiguous negatives: only count when the referent is the destination
DQ_AMBIGUOUS_NEG = {
    "environment_neg": [r"suj~", r"sujeira", r"lixo", r"barulhent~",
                        r"barulho", r"caotic~", r"abandonad~", r"degradad~",
                        r"mau cheiro", r"fedor"],
    "safety_neg": [r"insegur~", r"perigos~", r"assalt~", r"furt~",
                   r"violenci~", r"violent~", r"com medo"],
    "harassment": [r"assedi~", r"insistent~", r"importun~", r"mendig~"],
    "crowding": [r"lotad~", r"superlotad~", r"muito cheio", r"cheio demais",
                 r"aglomerac~", r"multidao", r"fila enorme",
                 r"filas enormes", r"fila gigantesc~"],
}

# these need a destination anchor to count
DQ_AMBIGUOUS_POS = {
    "heritage": [r"histori~", r"historic~", r"cultur~", r"tradic~",
                 r"folclor~", r"artesanat~ local", r"gastronomi~ local",
                 r"culinari~ local", r"comida local", r"folclor~ local"],
    "scenery": [r"lind~", r"bel~", r"beleza", r"belissim~", r"vista",
                r"vistas", r"espetacular~", r"deslumbrant~"],
    "atmosphere": [r"atmosfera", r"vibe", r"vibrant~", r"magic~", r"unic~"],
}

# =========================================================================
# 4. GENERIC EXPERIENTIAL AFFECT  (discriminant control)
# =========================================================================
EXP = {
    "affect": [
        r"inesquecivel", r"inesqueciveis", r"memoravel", r"memoraveis",
        r"marcante", r"incrivel", r"incriveis", r"maravilhos~", r"excelent~",
        r"otim~", r"perfeit~", r"espetacular", r"fantastic~", r"sensacional",
        r"extraordinari~", r"excepcional", r"agradavel", r"tranquil~",
        r"superou (?:as |minhas |nossas )?expectativas", r"acima das expectativas",  # noqa: E501
        r"melhor experienci~", r"experienci~ unic~", r"nota (?:10|1000|dez)",
        r"show de bola", r"simplesmente demais", r"adorei", r"amei",
    ],
}

# =========================================================================
# 5. BEHAVIOURAL INTENTION  (outcome)
# =========================================================================
INT = {
    "recommend": [
        r"recomend~", r"indico", r"indicamos", r"super indico",
        r"vale muito a pena", r"aconselh~",
    ],
    "revisit": [
        r"voltarei", r"voltaremos", r"quero voltar", r"queremos voltar",
        r"certeza voltarei", r"certeza voltaremos", r"voltar mais vezes",
        r"vou voltar", r"vamos voltar", r"pretendo voltar", r"volto sempre",
        r"sempre volto", r"ja e a (?:segunda|terceira|quarta) vez",
        r"proxima viagem", r"com certeza volt~",
    ],
}
INT_NEG = {
    "not_recommend": [
        r"nao recomend~", r"jamais recomend~", r"nunca mais", r"nao indico",
        r"nao voltarei", r"nao volto", r"fujam", r"fuja", r"passe longe",
        r"evitem", r"nao percam tempo", r"nao contrat~",
    ],
}

# =========================================================================
# REFERENT ANCHORS
# =========================================================================
PROVIDER_ANCHORS = [
    r"guia~", r"motorist~", r"equipe", r"staff", r"funcionari~", r"atendent~",
    r"atendiment~", r"empresa", r"agenci~", r"operador~", r"anfitri~",
    r"recepcionist~", r"garcon~", r"chef", r"cozinheir~", r"vendedor~",
    r"proprietari~", r"dono", r"dona do", r"gerent~", r"instrutor~",
    r"professor~", r"condutor~", r"servic~", r"atendeu", r"atenderam",
    r"hotel", r"pousada", r"hostel", r"restaurant~", r"loja", r"estudio",
    r"escola", r"tour", r"passei~", r"excursao", r"transfer", r"van",
    r"onibus", r"quarto~", r"apartament~", r"acomodac~", r"veiculo",
    r"carro", r"academia", r"aula~",
]
DESTINATION_ANCHORS = [
    r"cidade~", r"pais", r"regiao", r"destino", r"bairro", r"rua~",
    r"praia~", r"montanh~", r"deserto", r"ilha~", r"povo", r"populac~",
    r"habitante~", r"morador~", r"nativ~", r"as pessoas", r"a populacao",
    r"cultura local", r"turism~", r"pontos turistic~", r"atrac~",
    r"paisage~", r"centro", r"capital", r"metropole", r"vila", r"aldeia",
    r"litoral", r"interior", r"praca", r"avenida", r"orla", r"vielas",
    r"deserto", r"rio nilo", r"medina", r"souk",
] + [
    # the 30 sampled destinations and their countries / demonyms:
    # a toponym is the strongest possible destination anchor
    r"rio de janeiro", r"cariocas?", r"sao paulo", r"paulistan~",
    r"salvador", r"bahia", r"baian~", r"foz do iguacu", r"iguacu",
    r"pequim", r"beijing", r"xangai", r"shanghai", r"xi an", r"xian",
    r"china", r"chines~", r"cairo", r"gize", r"luxor", r"assua~",
    r"aswan", r"egito", r"egipci~", r"adis abeba", r"addis", r"lalibela",
    r"etiopia", r"etiope~", r"agra", r"delhi", r"deli", r"jaipur",
    r"mumbai", r"bombaim", r"india", r"indian~", r"rajastao",
    r"isfahan", r"esfahan", r"shiraz", r"teera~", r"tehran", r"ira",
    r"iranian~", r"moscou", r"moscow", r"sao petersburgo",
    r"petersburgo", r"russia", r"russ~", r"al ula", r"alula", r"jeddah",
    r"jedah", r"riade", r"riyadh", r"arabia saudita", r"saudit~",
    r"cidade do cabo", r"cape town", r"joanesburgo", r"johannesburgo",
    r"joburg", r"africa do sul", r"sul.?african~", r"abu dhabi",
    r"dubai", r"sharjah", r"emirados", r"emirad~", r"arabe~",
]

# Destination-level hospitality: hospitableness predicated of the PLACE or
# its RESIDENTS rather than of the commercial host.
DEST_HOSP_MARKERS = [
    r"povo (?:\w+ ){0,3}?(?:acolhedor|hospitaleir|receptiv|simpatic|gentil|amave)",  # noqa: E501
    r"(?:as )?pessoas (?:da cidade|do pais|locais|de la|daqui) (?:\w+ ){0,3}?(?:acolhedor|hospitaleir|receptiv|simpatic|gentil|amave)",  # noqa: E501
    r"populac~ (?:\w+ ){0,3}?(?:acolhedor|hospitaleir|receptiv|simpatic|gentil|amave)",  # noqa: E501
    r"morador~ (?:\w+ ){0,3}?(?:acolhedor|hospitaleir|receptiv|simpatic|gentil|amave)",  # noqa: E501
    r"nativ~ (?:\w+ ){0,3}?(?:acolhedor|hospitaleir|receptiv|simpatic|gentil|amave)",  # noqa: E501
    r"local~ (?:sao|eram) (?:\w+ ){0,2}?(?:acolhedor|hospitaleir|receptiv|simpatic|gentil|amave)",  # noqa: E501
    r"cidade (?:\w+ ){0,2}?(?:acolhedora|hospitaleira|receptiva|acolhe)",
    r"hospitalidade (?:do povo|da cidade|do pais|dos "
    r"locais|local|arabe|brasileir|indian|egipci|african|russ|iranian|saudit|etiope|sul.?african)",  # noqa: E501
    r"(?:povo|gente) (?:muito |super |bem "
    r")?(?:acolhedor|hospitaleir|receptiv|simpatic|gentil)",
    r"acolhid~ (?:pela cidade|pelo pais|pelo povo|pelos? local)",
    r"calor human~ d[oa] (?:povo|cidade|pais|gente)",
    r"recebid~ de brac~ abert~ (?:pel[oa]s? (?:povo|cidade|local|morador))",
]

# =========================================================================
# NEGATION AND INTENSIFIERS
# =========================================================================
NEGATORS = [
    "nao", "nunca", "jamais", "nenhum", "nenhuma", "nada", "sem",
    "pouco", "pouca", "faltou", "nem", "tampouco", "zero", "deixou",
]
INTENSIFIERS = [
    "muito", "super", "extremamente", "incrivelmente", "absolutamente",
    "totalmente", "completamente", "realmente", "bastante", "altamente",
    "demais", "extraordinariamente", "excepcionalmente", "verdadeiramente",
    "mega", "hiper", "sumamente",
]

CONSTRUCTS = {
    "HOSP": HOSP,
    "SQ_POS": SQ_POS,
    "SQ_NEG": SQ_NEG,
    "DQ_POS": DQ_POS,
    "DQ_NEG": DQ_NEG,
    "EXP": EXP,
    "INT": INT,
    "INT_NEG": INT_NEG,
}

# construct -> {subdim: [ambiguous stems]} , plus the anchor class required
AMBIGUOUS = {
    "SQ_POS": (SQ_AMBIGUOUS_POS, "provider"),
    "SQ_NEG": (SQ_AMBIGUOUS_NEG, "provider"),
    "DQ_POS": (DQ_AMBIGUOUS_POS, "destination"),
    "DQ_NEG": (DQ_AMBIGUOUS_NEG, "destination"),
}
