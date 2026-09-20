"""The language ids OmniVoice understands (the manual language picker).

OmniVoice detects the language from the text on its own, and most of the time
that is exactly right.  On short lines it is not: a one-word sentence in a
language close to a neighbour (Spanish and Portuguese, Hindi and Marathi, a
Chinese line carrying a Latin brand name) can come back with the wrong accent
- or as gibberish.  OmniVoice also accepts an explicit language hint, and this
module is the list of hints it was trained on: the 646 languages of
``k2-fsa/OmniVoice`` (581k hours of training audio).

* **Auto** (:data:`AUTO_LABEL`) is always the first entry of the picker.  Its
  value is :data:`AUTO` (``""``), which normalises to "no hint at all" and
  therefore keeps OmniVoice's own detection - the behaviour of every release
  before the picker existed.
* Every other entry carries the upstream **OmniVoice language id** (``en``,
  ``zh``, ``hi`` ...), which is exactly the value ``spec.clean_language``
  passes to the engine, plus the English name for the label.  The ISO 639-3
  code is kept alongside so that a code typed or pasted by hand (``eng``,
  ``cmn``, ``deu``) still lands on the right language.

The table below is a verbatim copy of upstream ``docs/languages.md``
(:data:`SOURCE_URL`), in the file's own alphabetical-by-name order, which is
also the order the combo box shows.  It is code rather than a data file on
purpose: the dialogs, the frozen build and the tests then need no extra
resource and no packaging rule, and a wrong hint fails loudly at import time
against :func:`_check` instead of silently at synthesis time.

Regenerate it with ``python tools/fetch_omnivoice_languages.py``.

Nothing here imports the engine, so it is safe in the GUI thread and in tests.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

#: The picker's first entry - no hint, OmniVoice detects the language itself.
AUTO = ""
AUTO_LABEL = "Auto (detect from the text)"

#: Where the table came from, so a future upgrade has an obvious starting point.
SOURCE_URL = "https://github.com/k2-fsa/OmniVoice/blob/master/docs/languages.md"

#: ``(English name, OmniVoice language id, ISO 639-3 code)`` for every
#: language OmniVoice was trained on, as published upstream.
LANGUAGES: Tuple[Tuple[str, str, str], ...] = (
    ("Abadi", "kbt", "kbt"),
    ("Abkhazian", "ab", "abk"),
    ("Abron", "abr", "abr"),
    ("Abua", "abn", "abn"),
    ("Adamawa Fulfulde", "fub", "fub"),
    ("Adyghe", "ady", "ady"),
    ("Afade", "aal", "aal"),
    ("Afrikaans", "af", "afr"),
    ("Agwagwune", "yay", "yay"),
    ("Aja (Benin)", "ajg", "ajg"),
    ("Akebu", "keu", "keu"),
    ("Alago", "ala", "ala"),
    ("Albanian", "sq", "sqi"),
    ("Algerian Arabic", "arq", "arq"),
    ("Algerian Saharan Arabic", "aao", "aao"),
    ("Ambo-Pasco Quechua", "qva", "qva"),
    ("Ambonese Malay", "abs", "abs"),
    ("Amdo Tibetan", "adx", "adx"),
    ("Amharic", "am", "amh"),
    ("Anaang", "anw", "anw"),
    ("Angika", "anp", "anp"),
    ("Antankarana Malagasy", "xmv", "xmv"),
    ("Aragonese", "an", "arg"),
    ("Arbëreshë Albanian", "aae", "aae"),
    ("Arequipa-La Unión Quechua", "qxu", "qxu"),
    ("Armenian", "hy", "hye"),
    ("Ashe", "ahs", "ahs"),
    ("Ashéninka Perené", "prq", "prq"),
    ("Askopan", "eiv", "eiv"),
    ("Assamese", "as", "asm"),
    ("Asturian", "ast", "ast"),
    ("Atayal", "tay", "tay"),
    ("Awak", "awo", "awo"),
    ("Ayacucho Quechua", "quy", "quy"),
    ("Azerbaijani", "az", "aze"),
    ("Baatonum", "bba", "bba"),
    ("Bacama", "bcy", "bcy"),
    ("Bade", "bde", "bde"),
    ("Bafia", "ksf", "ksf"),
    ("Bafut", "bfd", "bfd"),
    ("Bagirmi Fulfulde", "fui", "fui"),
    ("Bago-Kusuntu", "bqg", "bqg"),
    ("Baharna Arabic", "abv", "abv"),
    ("Bakoko", "bkh", "bkh"),
    ("Balanta-Ganja", "bjt", "bjt"),
    ("Balti", "bft", "bft"),
    ("Bamenyam", "bce", "bce"),
    ("Bamun", "bax", "bax"),
    ("Bangwinji", "bsj", "bsj"),
    ("Banjar", "bjn", "bjn"),
    ("Bankon", "abb", "abb"),
    ("Baoulé", "bci", "bci"),
    ("Bara Malagasy", "bhr", "bhr"),
    ("Barok", "bjk", "bjk"),
    ("Basa (Cameroon)", "bas", "bas"),
    ("Basa (Nigeria)", "bzw", "bzw"),
    ("Bashkir", "ba", "bak"),
    ("Basque", "eu", "eus"),
    ("Batak Mandailing", "btm", "btm"),
    ("Batanga", "bnm", "bnm"),
    ("Bateri", "btv", "btv"),
    ("Bats", "bbl", "bbl"),
    ("Bayot", "bda", "bda"),
    ("Bebele", "beb", "beb"),
    ("Belarusian", "be", "bel"),
    ("Bengali", "bn", "ben"),
    ("Betawi", "bew", "bew"),
    ("Bhili", "bhb", "bhb"),
    ("Bhojpuri", "bho", "bho"),
    ("Bilur", "bxf", "bxf"),
    ("Bima", "bhp", "bhp"),
    ("Bodo", "brx", "brx"),
    ("Boghom", "bux", "bux"),
    ("Bokyi", "bky", "bky"),
    ("Bomu", "bmq", "bmq"),
    ("Bondei", "bou", "bou"),
    ("Borgu Fulfulde", "fue", "fue"),
    ("Bosnian", "bs", "bos"),
    ("Brahui", "brh", "brh"),
    ("Braj", "bra", "bra"),
    ("Breton", "br", "bre"),
    ("Buduma", "bdm", "bdm"),
    ("Buginese", "bug", "bug"),
    ("Bukharic", "bhh", "bhh"),
    ("Bulgarian", "bg", "bul"),
    ("Bulu (Cameroon)", "bum", "bum"),
    ("Bundeli", "bns", "bns"),
    ("Bunun", "bnn", "bnn"),
    ("Bura-Pabir", "bwr", "bwr"),
    ("Burak", "bys", "bys"),
    ("Burmese", "my", "mya"),
    ("Burushaski", "bsk", "bsk"),
    ("Cacaloxtepec Mixtec", "miu", "miu"),
    ("Cajatambo North Lima Quechua", "qvl", "qvl"),
    ("Cakfem-Mushere", "cky", "cky"),
    ("Cameroon Pidgin", "wes", "wes"),
    ("Campidanese Sardinian", "sro", "sro"),
    ("Cantonese", "yue", "yue"),
    ("Catalan", "ca", "cat"),
    ("Cebuano", "ceb", "ceb"),
    ("Cen", "cen", "cen"),
    ("Central Kurdish", "ckb", "ckb"),
    ("Central Nahuatl", "nhn", "nhn"),
    ("Central Pame", "pbs", "pbs"),
    ("Central Pashto", "pst", "pst"),
    ("Central Puebla Nahuatl", "ncx", "ncx"),
    ("Central Tarahumara", "tar", "tar"),
    ("Central Yupik", "esu", "esu"),
    ("Central-Eastern Niger Fulfulde", "fuq", "fuq"),
    ("Chadian Arabic", "shu", "shu"),
    ("Chichewa", "ny", "nya"),
    ("Chichicapan Zapotec", "zpv", "zpv"),
    ("Chiga", "cgg", "cgg"),
    ("Chimalapa Zoque", "zoh", "zoh"),
    ("Chimborazo Highland Quichua", "qug", "qug"),
    ("Chinese", "zh", "cmn"),
    ("Chiquián Ancash Quechua", "qxa", "qxa"),
    ("Chitwania Tharu", "the", "the"),
    ("Chokwe", "cjk", "cjk"),
    ("Chuvash", "cv", "chv"),
    ("Cibak", "ckl", "ckl"),
    ("Coastal Konjo", "kjc", "kjc"),
    ("Copainalá Zoque", "zoc", "zoc"),
    ("Cornish", "kw", "cor"),
    ("Corongo Ancash Quechua", "qwa", "qwa"),
    ("Croatian", "hr", "hrv"),
    ("Cross River Mbembe", "mfn", "mfn"),
    ("Cuyamecalco Mixtec", "xtu", "xtu"),
    ("Czech", "cs", "ces"),
    ("Dadiya", "dbd", "dbd"),
    ("Dagbani", "dag", "dag"),
    ("Dameli", "dml", "dml"),
    ("Danish", "da", "dan"),
    ("Dargwa", "dar", "dar"),
    ("Dazaga", "dzg", "dzg"),
    ("Deccan", "dcc", "dcc"),
    ("Degema", "deg", "deg"),
    ("Dera (Nigeria)", "kna", "kna"),
    ("Dghwede", "dgh", "dgh"),
    ("Dhatki", "mki", "mki"),
    ("Dhivehi", "dv", "div"),
    ("Dhofari Arabic", "adf", "adf"),
    ("Dijim-Bwilim", "cfa", "cfa"),
    ("Dogri", "dgo", "dgo"),
    ("Domaaki", "dmk", "dmk"),
    ("Dotyali", "dty", "dty"),
    ("Duala", "dua", "dua"),
    ("Dutch", "nl", "nld"),
    ("Dũya", "ldb", "ldb"),
    ("Dyula", "dyu", "dyu"),
    ("Eastern Balochi", "bgp", "bgp"),
    ("Eastern Bolivian Guaraní", "gui", "gui"),
    ("Eastern Egyptian Bedawi Arabic", "avl", "avl"),
    ("Eastern Krahn", "kqo", "kqo"),
    ("Eastern Mari", "mhr", "mhr"),
    ("Eastern Yiddish", "ydd", "ydd"),
    ("Ebrié", "ebr", "ebr"),
    ("Eggon", "ego", "ego"),
    ("Egyptian Arabic", "arz", "arz"),
    ("Ejagham", "etu", "etu"),
    ("Eleme", "elm", "elm"),
    ("Eloyi", "afo", "afo"),
    ("Embu", "ebu", "ebu"),
    ("English", "en", "eng"),
    ("Erzya", "myv", "myv"),
    ("Esan", "ish", "ish"),
    ("Esperanto", "eo", "epo"),
    ("Estonian", "et", "est"),
    ("Eton (Cameroon)", "eto", "eto"),
    ("Ewondo", "ewo", "ewo"),
    ("Extremaduran", "ext", "ext"),
    ("Fang (Equatorial Guinea)", "fan", "fan"),
    ("Fanti", "fat", "fat"),
    ("Farefare", "gur", "gur"),
    ("Fe'fe'", "fmp", "fmp"),
    ("Filipino", "fil", "fil"),
    ("Filomena Mata-Coahuitlán Totonac", "tlp", "tlp"),
    ("Finnish", "fi", "fin"),
    ("Fipa", "fip", "fip"),
    ("French", "fr", "fra"),
    ("Fulah", "ff", "ful"),
    ("Galician", "gl", "glg"),
    ("Gambian Wolof", "wof", "wof"),
    ("Ganda", "lg", "lug"),
    ("Garhwali", "gbm", "gbm"),
    ("Gawar-Bati", "gwt", "gwt"),
    ("Gawri", "gwc", "gwc"),
    ("Gbagyi", "gbr", "gbr"),
    ("Gbari", "gby", "gby"),
    ("Geji", "gyz", "gyz"),
    ("Gen", "gej", "gej"),
    ("Georgian", "ka", "kat"),
    ("German", "de", "deu"),
    ("Geser-Gorom", "ges", "ges"),
    ("Gheg Albanian", "aln", "aln"),
    ("Ghomálá'", "bbj", "bbj"),
    ("Gidar", "gid", "gid"),
    ("Glavda", "glw", "glw"),
    ("Goan Konkani", "gom", "gom"),
    ("Goaria", "gig", "gig"),
    ("Goemai", "ank", "ank"),
    ("Gola", "gol", "gol"),
    ("Greek", "el", "ell"),
    ("Guarani", "gn", "grn"),
    ("Guduf-Gava", "gdf", "gdf"),
    ("Guerrero Amuzgo", "amu", "amu"),
    ("Gujarati", "gu", "guj"),
    ("Gujari", "gju", "gju"),
    ("Gulf Arabic", "afb", "afb"),
    ("Gurgula", "ggg", "ggg"),
    ("Gusii", "guz", "guz"),
    ("Gusilay", "gsl", "gsl"),
    ("Gweno", "gwe", "gwe"),
    ("Güilá Zapotec", "ztu", "ztu"),
    ("Hadothi", "hoj", "hoj"),
    ("Hahon", "hah", "hah"),
    ("Haitian", "ht", "hat"),
    ("Hakha Chin", "cnh", "cnh"),
    ("Hakö", "hao", "hao"),
    ("Halia", "hla", "hla"),
    ("Hausa", "ha", "hau"),
    ("Hawaiian", "haw", "haw"),
    ("Hazaragi", "haz", "haz"),
    ("Hebrew", "he", "heb"),
    ("Hemba", "hem", "hem"),
    ("Herero", "hz", "her"),
    ("Highland Konjo", "kjk", "kjk"),
    ("Hijazi Arabic", "acw", "acw"),
    ("Hindi", "hi", "hin"),
    ("Huarijio", "var", "var"),
    ("Huautla Mazatec", "mau", "mau"),
    ("Huaxcaleca Nahuatl", "nhq", "nhq"),
    ("Huba", "hbb", "hbb"),
    ("Huitepec Mixtec", "mxs", "mxs"),
    ("Hula", "hul", "hul"),
    ("Hungarian", "hu", "hun"),
    ("Hunjara-Kaina Ke", "hkk", "hkk"),
    ("Hwana", "hwo", "hwo"),
    ("Ibibio", "ibb", "ibb"),
    ("Icelandic", "is", "isl"),
    ("Idakho-Isukha-Tiriki", "ida", "ida"),
    ("Idoma", "idu", "idu"),
    ("Igbo", "ig", "ibo"),
    ("Igo", "ahl", "ahl"),
    ("Ikposo", "kpo", "kpo"),
    ("Ikwere", "ikw", "ikw"),
    ("Imbabura Highland Quichua", "qvi", "qvi"),
    ("Indonesian", "id", "ind"),
    ("Indus Kohistani", "mvy", "mvy"),
    ("Interlingua (International Auxiliary Language Association)", "ia", "ina"),
    ("Inupiaq", "ik", "ipk"),
    ("Irish", "ga", "gle"),
    ("Iron Ossetic", "os", "oss"),
    ("Isekiri", "its", "its"),
    ("Isoko", "iso", "iso"),
    ("Italian", "it", "ita"),
    ("Ito", "itw", "itw"),
    ("Itzá", "itz", "itz"),
    ("Ixtayutla Mixtec", "vmj", "vmj"),
    ("Izon", "ijc", "ijc"),
    ("Jambi Malay", "jax", "jax"),
    ("Japanese", "ja", "jpn"),
    ("Jaqaru", "jqr", "jqr"),
    ("Jauja Wanca Quechua", "qxw", "qxw"),
    ("Jaunsari", "jns", "jns"),
    ("Javanese", "jv", "jav"),
    ("Jiba", "juo", "juo"),
    ("Jju", "kaj", "kaj"),
    ("Judeo-Moroccan Arabic", "aju", "aju"),
    ("Juxtlahuaca Mixtec", "vmc", "vmc"),
    ("Kabardian", "kbd", "kbd"),
    ("Kabras", "lkb", "lkb"),
    ("Kabuverdianu", "kea", "kea"),
    ("Kabyle", "kab", "kab"),
    ("Kachi Koli", "gjk", "gjk"),
    ("Kairak", "ckr", "ckr"),
    ("Kalabari", "ijn", "ijn"),
    ("Kalasha", "kls", "kls"),
    ("Kalenjin", "kln", "kln"),
    ("Kalkoti", "xka", "xka"),
    ("Kamba", "kam", "kam"),
    ("Kamo", "kcq", "kcq"),
    ("Kanauji", "bjj", "bjj"),
    ("Kanembu", "kbl", "kbl"),
    ("Kannada", "kn", "kan"),
    ("Karekare", "kai", "kai"),
    ("Kashmiri", "ks", "kas"),
    ("Kathoriya Tharu", "tkt", "tkt"),
    ("Kati", "bsh", "bsh"),
    ("Kazakh", "kk", "kaz"),
    ("Keiyo", "eyo", "eyo"),
    ("Khams Tibetan", "khg", "khg"),
    ("Khana", "ogo", "ogo"),
    ("Khetrani", "xhe", "xhe"),
    ("Khmer", "km", "khm"),
    ("Khowar", "khw", "khw"),
    ("Kinga", "zga", "zga"),
    ("Kinnauri", "kfk", "kfk"),
    ("Kinyarwanda", "rw", "kin"),
    ("Kirghiz", "ky", "kir"),
    ("Kirya-Konzəl", "fkk", "fkk"),
    ("Kochila Tharu", "thq", "thq"),
    ("Kohistani Shina", "plk", "plk"),
    ("Kohumono", "bcs", "bcs"),
    ("Kok Borok", "trp", "trp"),
    ("Kol (Papua New Guinea)", "kol", "kol"),
    ("Kom (Cameroon)", "bkm", "bkm"),
    ("Koma", "kmy", "kmy"),
    ("Konkani", "knn", "knn"),
    ("Konzo", "koo", "koo"),
    ("Korean", "ko", "kor"),
    ("Korwa", "kfp", "kfp"),
    ("Kota (India)", "kfe", "kfe"),
    ("Koti", "eko", "eko"),
    ("Kuanua", "ksd", "ksd"),
    ("Kuanyama", "kj", "kua"),
    ("Kui (India)", "uki", "uki"),
    ("Kulung (Nigeria)", "bbu", "bbu"),
    ("Kuot", "kto", "kto"),
    ("Kushi", "kuh", "kuh"),
    ("Kwambi", "kwm", "kwm"),
    ("Kwasio", "nmg", "nmg"),
    ("Lala-Roba", "lla", "lla"),
    ("Lamang", "hia", "hia"),
    ("Lao", "lo", "lao"),
    ("Larike-Wakasihu", "alo", "alo"),
    ("Lasi", "lss", "lss"),
    ("Latgalian", "ltg", "ltg"),
    ("Latvian", "lv", "lav"),
    ("Levantine Arabic", "apc", "apc"),
    ("Liana-Seti", "ste", "ste"),
    ("Liberia Kpelle", "xpe", "xpe"),
    ("Liberian English", "lir", "lir"),
    ("Libyan Arabic", "ayl", "ayl"),
    ("Ligurian", "lij", "lij"),
    ("Lijili", "mgi", "mgi"),
    ("Lingala", "ln", "lin"),
    ("Lithuanian", "lt", "lit"),
    ("Loarki", "lrk", "lrk"),
    ("Logooli", "rag", "rag"),
    ("Logudorese Sardinian", "src", "src"),
    ("Loja Highland Quichua", "qvj", "qvj"),
    ("Loloda", "loa", "loa"),
    ("Longuda", "lnu", "lnu"),
    ("Loxicha Zapotec", "ztp", "ztp"),
    ("Luba-Lulua", "lua", "lua"),
    ("Luo", "luo", "luo"),
    ("Lushai", "lus", "lus"),
    ("Luxembourgish", "lb", "ltz"),
    ("Maasina Fulfulde", "ffm", "ffm"),
    ("Maba (Chad)", "mde", "mde"),
    ("Macedo-Romanian", "rup", "rup"),
    ("Macedonian", "mk", "mkd"),
    ("Mada (Cameroon)", "mxu", "mxu"),
    ("Mafa", "maf", "maf"),
    ("Maithili", "mai", "mai"),
    ("Malay", "ms", "msa"),
    ("Malayalam", "ml", "mal"),
    ("Mali", "gcc", "gcc"),
    ("Malinaltepec Me'phaa", "tcf", "tcf"),
    ("Maltese", "mt", "mlt"),
    ("Mandara", "tbf", "tbf"),
    ("Mandjak", "mfv", "mfv"),
    ("Manggarai", "mqy", "mqy"),
    ("Manipuri", "mni", "mni"),
    ("Mansoanka", "msw", "msw"),
    ("Manx", "gv", "glv"),
    ("Maori", "mi", "mri"),
    ("Marathi", "mr", "mar"),
    ("Marghi Central", "mrt", "mrt"),
    ("Marghi South", "mfm", "mfm"),
    ("Maria (India)", "mrr", "mrr"),
    ("Marwari (Pakistan)", "mve", "mve"),
    ("Masana", "mcn", "mcn"),
    ("Masikoro Malagasy", "msh", "msh"),
    ("Matsés", "mcf", "mcf"),
    ("Mazaltepec Zapotec", "zpy", "zpy"),
    ("Mazatlán Mazatec", "vmz", "vmz"),
    ("Mazatlán Mixe", "mzl", "mzl"),
    ("Mbe", "mfo", "mfo"),
    ("Mbo (Cameroon)", "mbo", "mbo"),
    ("Mbum", "mdd", "mdd"),
    ("Medumba", "byv", "byv"),
    ("Mekeo", "mek", "mek"),
    ("Meru", "mer", "mer"),
    ("Mesopotamian Arabic", "acm", "acm"),
    ("Mewari", "mtr", "mtr"),
    ("Min Nan Chinese", "nan", "nan"),
    ("Mingrelian", "xmf", "xmf"),
    ("Mitlatongo Mixtec", "vmm", "vmm"),
    ("Miya", "mkf", "mkf"),
    ("Mokpwe", "bri", "bri"),
    ("Moksha", "mdf", "mdf"),
    ("Mom Jango", "ver", "ver"),
    ("Mongolian", "mn", "mon"),
    ("Moroccan Arabic", "ary", "ary"),
    ("Motu", "meu", "meu"),
    ("Mpiemo", "mcx", "mcx"),
    ("Mpumpong", "mgg", "mgg"),
    ("Mundang", "mua", "mua"),
    ("Mungaka", "mhk", "mhk"),
    ("Musey", "mse", "mse"),
    ("Musgu", "mug", "mug"),
    ("Musi", "mui", "mui"),
    ("Naba", "mne", "mne"),
    ("Najdi Arabic", "ars", "ars"),
    ("Nalik", "nal", "nal"),
    ("Nawdm", "nmz", "nmz"),
    ("Ndonga", "ng", "ndo"),
    ("Neapolitan", "nap", "nap"),
    ("Nepali", "npi", "npi"),
    ("Ngamo", "nbh", "nbh"),
    ("Ngas", "anc", "anc"),
    ("Ngiemboon", "nnh", "nnh"),
    ("Ngizim", "ngi", "ngi"),
    ("Ngomba", "jgo", "jgo"),
    ("Ngombale", "nla", "nla"),
    ("Nigerian Fulfulde", "fuv", "fuv"),
    ("Nigerian Pidgin", "pcm", "pcm"),
    ("Nimadi", "noe", "noe"),
    ("Nobiin", "fia", "fia"),
    ("North Mesopotamian Arabic", "ayp", "ayp"),
    ("North Moluccan Malay", "max", "max"),
    ("Northern Betsimisaraka Malagasy", "bmm", "bmm"),
    ("Northern Hindko", "hno", "hno"),
    ("Northern Kurdish", "kmr", "kmr"),
    ("Northern Pame", "pmq", "pmq"),
    ("Northern Pashto", "pbu", "pbu"),
    ("Northern Uzbek", "uzn", "uzn"),
    ("Northwest Gbaya", "gya", "gya"),
    ("Norwegian", "no", "nor"),
    ("Norwegian Bokmål", "nb", "nob"),
    ("Norwegian Nynorsk", "nn", "nno"),
    ("Notsi", "ncf", "ncf"),
    ("Nyankpa", "yes", "yes"),
    ("Nyungwe", "nyu", "nyu"),
    ("Nzanyi", "nja", "nja"),
    ("Nüpode Huitoto", "hux", "hux"),
    ("Occitan", "oc", "oci"),
    ("Od", "odk", "odk"),
    ("Odia", "ory", "ory"),
    ("Odual", "odu", "odu"),
    ("Omani Arabic", "acx", "acx"),
    ("Orizaba Nahuatl", "nlv", "nlv"),
    ("Orma", "orc", "orc"),
    ("Ormuri", "oru", "oru"),
    ("Oromo", "om", "orm"),
    ("Pahari-Potwari", "phr", "phr"),
    ("Paiwan", "pwn", "pwn"),
    ("Panjabi", "pa", "pan"),
    ("Papuan Malay", "pmy", "pmy"),
    ("Parkari Koli", "kvx", "kvx"),
    ("Pedi", "nso", "nso"),
    ("Pero", "pip", "pip"),
    ("Persian", "fa", "fas"),
    ("Petats", "pex", "pex"),
    ("Phalura", "phl", "phl"),
    ("Piemontese", "pms", "pms"),
    ("Piya-Kwonci", "piy", "piy"),
    ("Plateau Malagasy", "plt", "plt"),
    ("Polish", "pl", "pol"),
    ("Poqomam", "poc", "poc"),
    ("Portuguese", "pt", "por"),
    ("Pulaar", "fuc", "fuc"),
    ("Pular", "fuf", "fuf"),
    ("Puno Quechua", "qxp", "qxp"),
    ("Pushto", "ps", "pus"),
    ("Pökoot", "pko", "pko"),
    ("Qaqet", "byx", "byx"),
    ("Quiotepec Chinantec", "chq", "chq"),
    ("Rana Tharu", "thr", "thr"),
    ("Rangi", "lag", "lag"),
    ("Rapoisi", "kyx", "kyx"),
    ("Ratahan", "rth", "rth"),
    ("Rayón Zoque", "zor", "zor"),
    ("Romanian", "ro", "ron"),
    ("Romansh", "rm", "roh"),
    ("Rombo", "rof", "rof"),
    ("Rotokas", "roo", "roo"),
    ("Rukai", "dru", "dru"),
    ("Russian", "ru", "rus"),
    ("Sacapulteco", "quv", "quv"),
    ("Saidi Arabic", "aec", "aec"),
    ("Sakalava Malagasy", "skg", "skg"),
    ("Sakizaya", "szy", "szy"),
    ("Saleman", "sau", "sau"),
    ("Samba Daka", "ccg", "ccg"),
    ("Samba Leko", "ndi", "ndi"),
    ("San Felipe Otlaltepec Popoloca", "pow", "pow"),
    ("San Francisco Del Mar Huave", "hue", "hue"),
    ("San Juan Atzingo Popoloca", "poe", "poe"),
    ("San Martín Itunyoso Triqui", "trq", "trq"),
    ("San Miguel El Grande Mixtec", "mig", "mig"),
    ("Sansi", "ssi", "ssi"),
    ("Sanskrit", "sa", "san"),
    ("Santa Ana de Tusi Pasco Quechua", "qxt", "qxt"),
    ("Santa Catarina Albarradas Zapotec", "ztn", "ztn"),
    ("Santali", "sat", "sat"),
    ("Santiago del Estero Quichua", "qus", "qus"),
    ("Saposa", "sps", "sps"),
    ("Saraiki", "skr", "skr"),
    ("Sardinian", "sc", "srd"),
    ("Saya", "say", "say"),
    ("Sediq", "trv", "trv"),
    ("Serbian", "sr", "srp"),
    ("Seri", "sei", "sei"),
    ("Shina", "scl", "scl"),
    ("Shona", "sn", "sna"),
    ("Siar-Lak", "sjr", "sjr"),
    ("Sibe", "nco", "nco"),
    ("Sicilian", "scn", "scn"),
    ("Sihuas Ancash Quechua", "qws", "qws"),
    ("Sikkimese", "sip", "sip"),
    ("Sinaugoro", "snc", "snc"),
    ("Sindhi", "sd", "snd"),
    ("Sindhi Bhil", "sbn", "sbn"),
    ("Sinhala", "si", "sin"),
    ("Sinicahua Mixtec", "xti", "xti"),
    ("Sipacapense", "qum", "qum"),
    ("Siwai", "siw", "siw"),
    ("Slovak", "sk", "slk"),
    ("Slovenian", "sl", "slv"),
    ("Solos", "sol", "sol"),
    ("Somali", "so", "som"),
    ("Soninke", "snk", "snk"),
    ("South Giziga", "giz", "giz"),
    ("South Ucayali Ashéninka", "cpy", "cpy"),
    ("Southeastern Nochixtlán Mixtec", "mxy", "mxy"),
    ("Southern Betsimisaraka Malagasy", "bzc", "bzc"),
    ("Southern Pashto", "pbt", "pbt"),
    ("Southern Pastaza Quechua", "qup", "qup"),
    ("Soyaltepec Mazatec", "vmp", "vmp"),
    ("Spanish", "es", "spa"),
    ("Standard Arabic", "arb", "arb"),
    ("Standard Moroccan Tamazight", "zgh", "zgh"),
    ("Sudanese Arabic", "apd", "apd"),
    ("Sulka", "sua", "sua"),
    ("Svan", "sva", "sva"),
    ("Swahili", "sw", "swa"),
    ("Swedish", "sv", "swe"),
    ("Tae'", "rob", "rob"),
    ("Tahaggart Tamahaq", "thv", "thv"),
    ("Taita", "dav", "dav"),
    ("Tajik", "tg", "tgk"),
    ("Tamil", "ta", "tam"),
    ("Tandroy-Mahafaly Malagasy", "tdx", "tdx"),
    ("Tangale", "tan", "tan"),
    ("Tanosy Malagasy", "txy", "txy"),
    ("Tarok", "yer", "yer"),
    ("Tatar", "tt", "tat"),
    ("Tedaga", "tuq", "tuq"),
    ("Telugu", "te", "tel"),
    ("Tem", "kdh", "kdh"),
    ("Teop", "tio", "tio"),
    ("Tepeuxila Cuicatec", "cux", "cux"),
    ("Tepinapa Chinantec", "cte", "cte"),
    ("Tera", "ttr", "ttr"),
    ("Terei", "buo", "buo"),
    ("Termanu", "twu", "twu"),
    ("Tesaka Malagasy", "tkg", "tkg"),
    ("Tetelcingo Nahuatl", "nhg", "nhg"),
    ("Teutila Cuicatec", "cut", "cut"),
    ("Thai", "th", "tha"),
    ("Tibetan", "bo", "bod"),
    ("Tidaá Mixtec", "mtx", "mtx"),
    ("Tidore", "tvo", "tvo"),
    ("Tigak", "tgc", "tgc"),
    ("Tigre", "tig", "tig"),
    ("Tigrinya", "ti", "tir"),
    ("Tilquiapan Zapotec", "zts", "zts"),
    ("Tinputz", "tpz", "tpz"),
    ("Tlacoapa Me'phaa", "tpl", "tpl"),
    ("Tlacoatzintepec Chinantec", "ctl", "ctl"),
    ("Tlingit", "tli", "tli"),
    ("Toki Pona", "tok", "tok"),
    ("Tomoip", "tqp", "tqp"),
    ("Tondano", "tdn", "tdn"),
    ("Tonsea", "txs", "txs"),
    ("Tooro", "ttj", "ttj"),
    ("Torau", "ttu", "ttu"),
    ("Torwali", "trw", "trw"),
    ("Tsimihety Malagasy", "xmw", "xmw"),
    ("Tsotso", "lto", "lto"),
    ("Tswana", "tn", "tsn"),
    ("Tugen", "tuy", "tuy"),
    ("Tuki", "bag", "bag"),
    ("Tula", "tul", "tul"),
    ("Tulu", "tcy", "tcy"),
    ("Tunen", "tvu", "tvu"),
    ("Tungag", "lcm", "lcm"),
    ("Tunisian Arabic", "aeb", "aeb"),
    ("Tupuri", "tui", "tui"),
    ("Turkana", "tuv", "tuv"),
    ("Turkish", "tr", "tur"),
    ("Turkmen", "tk", "tuk"),
    ("Tututepec Mixtec", "mtu", "mtu"),
    ("Twi", "tw", "twi"),
    ("Ubaghara", "byc", "byc"),
    ("Uighur", "ug", "uig"),
    ("Ukrainian", "uk", "ukr"),
    ("Umbundu", "umb", "umb"),
    ("Upper Sorbian", "hsb", "hsb"),
    ("Urdu", "ur", "urd"),
    ("Ushojo", "ush", "ush"),
    ("Uzbek", "uz", "uzb"),
    ("Vai", "vai", "vai"),
    ("Vietnamese", "vi", "vie"),
    ("Votic", "vot", "vot"),
    ("Võro", "vro", "vro"),
    ("Waci Gbe", "wci", "wci"),
    ("Wadiyara Koli", "kxp", "kxp"),
    ("Waja", "wja", "wja"),
    ("Wakhi", "wbl", "wbl"),
    ("Wanga", "lwg", "lwg"),
    ("Wapan", "juk", "juk"),
    ("Warji", "wji", "wji"),
    ("Welsh", "cy", "cym"),
    ("Wemale", "weo", "weo"),
    ("Western Frisian", "fy", "fry"),
    ("Western Highland Purepecha", "pua", "pua"),
    ("Western Juxtlahuaca Mixtec", "jmx", "jmx"),
    ("Western Maninkakan", "mlq", "mlq"),
    ("Western Mari", "mrj", "mrj"),
    ("Western Niger Fulfulde", "fuh", "fuh"),
    ("Western Panjabi", "pnb", "pnb"),
    ("Wolof", "wo", "wol"),
    ("Wuzlam", "udl", "udl"),
    ("Xanaguía Zapotec", "ztg", "ztg"),
    ("Xhosa", "xh", "xho"),
    ("Yace", "ekr", "ekr"),
    ("Yakut", "sah", "sah"),
    ("Yalahatan", "jal", "jal"),
    ("Yanahuanca Pasco Quechua", "qur", "qur"),
    ("Yangben", "yav", "yav"),
    ("Yaqui", "yaq", "yaq"),
    ("Yauyos Quechua", "qux", "qux"),
    ("Yekhee", "ets", "ets"),
    ("Yiddish", "yi", "yid"),
    ("Yidgha", "ydg", "ydg"),
    ("Yoruba", "yo", "yor"),
    ("Yutanduchi Mixtec", "mab", "mab"),
    ("Zacatlán-Ahuacatlán-Tepetzintla Nahuatl", "nhi", "nhi"),
    ("Zarma", "dje", "dje"),
    ("Zaza", "zza", "zza"),
    ("Zulu", "zu", "zul"),
    ("Ömie", "aom", "aom"),
)


def _check() -> None:
    """Fail at import time if the generated table was ever edited by hand."""
    seen_ids: Dict[str, str] = {}
    for name, omni_id, iso in LANGUAGES:
        if not name or not omni_id or not iso:
            raise ValueError(f"incomplete language row: {(name, omni_id, iso)!r}")
        if not omni_id.islower() or not iso.islower():
            raise ValueError(f"language ids must be lower case: {omni_id!r}")
        if omni_id in seen_ids:
            raise ValueError(
                f"duplicate OmniVoice id {omni_id!r} ({seen_ids[omni_id]!r} and {name!r})"
            )
        seen_ids[omni_id] = name


_check()

#: Number of languages OmniVoice was trained on (upstream: 646, 581k hours).
COUNT = len(LANGUAGES)

_BY_ID: Dict[str, Tuple[str, str]] = {
    omni_id: (name, iso) for name, omni_id, iso in LANGUAGES
}
_BY_ISO: Dict[str, str] = {iso: omni_id for _name, omni_id, iso in LANGUAGES}
_BY_NAME: Dict[str, str] = {name.lower(): omni_id for name, omni_id, _iso in LANGUAGES}


def language_ids() -> Tuple[str, ...]:
    """Every OmniVoice language id, in the picker's order."""
    return tuple(omni_id for _name, omni_id, _iso in LANGUAGES)


def names() -> Tuple[str, ...]:
    """Every English language name, in the picker's order."""
    return tuple(name for name, _omni_id, _iso in LANGUAGES)


def count() -> int:
    """How many languages the table holds (``COUNT``, as a function)."""
    return len(LANGUAGES)


def is_known(value: Optional[str]) -> bool:
    """True when ``value`` is a language id this table lists."""
    return bool(value) and str(value).strip().lower() in _BY_ID


def display_name(value: Optional[str]) -> str:
    """The English name of a language id, or the raw text when unknown.

    Used by the dialogs to describe a stored hint without a lookup table of
    their own; an unknown value is returned unchanged so nothing is lost.
    """
    text = ("" if value is None else str(value)).strip()
    if not text:
        return ""
    entry = _BY_ID.get(text.lower())
    return entry[0] if entry else text


def label(value: Optional[str]) -> str:
    """The combo-box label of a language: ``"English (en)"``.

    The auto entry gets its own wording; an unknown value is shown as typed so
    a hint from a newer engine version is never hidden from the user.
    """
    text = ("" if value is None else str(value)).strip()
    if not text:
        return AUTO_LABEL
    lower = text.lower()
    entry = _BY_ID.get(lower)
    if entry:
        return f"{entry[0]} ({lower})"
    return text


def normalise(value: Optional[str]) -> Optional[str]:
    """Map free text to a canonical OmniVoice language id, or ``None``.

    Accepts what a user may reasonably type or paste:

    * ``""`` / ``"auto"`` / ``"automatic"`` / ``"any"`` / the picker's own
      :data:`AUTO_LABEL` -> ``None`` (no hint),
    * a language id (``en``, ``zh``, ``kbt``) -> itself, lower-cased,
    * an ISO 639-3 code (``eng``, ``cmn``, ``deu``) -> its OmniVoice id,
    * an English name (``English``, ``Chinese``) -> its OmniVoice id,
    * a picker label (``English (en)``) -> its OmniVoice id,
    * anything else -> the trimmed text, lower-cased, untouched.

    The last rule matters: an engine newer than this table may know languages
    it does not, and refusing that hint here would break a working project.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    if lower in ("auto", "automatic", "any") or lower.startswith("auto ("):
        return None
    if lower in _BY_ID:
        return lower
    if lower in _BY_ISO:
        return _BY_ISO[lower]
    if lower in _BY_NAME:
        return _BY_NAME[lower]
    # "English (en)" / "English - en" / "en (English)" -> the bracketed code.
    for part in re.split(r"[()\[\],/]| - ", lower):
        part = part.strip()
        if part in _BY_ID:
            return part
        if part in _BY_ISO:
            return _BY_ISO[part]
        if part in _BY_NAME:
            return _BY_NAME[part]
    return lower


def choices() -> List[Tuple[str, str]]:
    """``(label, value)`` pairs for the picker: auto first, then alphabetical.

    The list is built once per call and is cheap; callers that fill a combo box
    call it exactly once.
    """
    out: List[Tuple[str, str]] = [(AUTO_LABEL, AUTO)]
    out.extend((f"{name} ({omni_id})", omni_id) for name, omni_id, _iso in LANGUAGES)
    return out


def selection_index(value: Optional[str]) -> int:
    """The :func:`choices` index of a stored hint (0 = auto, 0 when unknown)."""
    normalised = normalise(value)
    if not normalised:
        return 0
    for index, (_label, item_value) in enumerate(choices()):
        if item_value == normalised:
            return index
    return 0
